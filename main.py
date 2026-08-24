"""
main.py — Entry point for the Gaze-Controlled Virtual Keyboard

Modes:
    --mode mouse          : GUI only, gaze simulated by mouse position
    --mode webcam_async   : Real gaze via Algorithm 1 (Async selector)
    --mode webcam_sync    : Real gaze via Algorithm 2 (Sync selector)
    --mode calibrate      : Run calibration (alias for calibrate.py)

Dwell time (mouse mode):
    User must keep mouse in a zone for --dwell_sec seconds (default 1.5s)
    without moving to an adjacent zone. A progress bar fills up in each cell.
    If the mouse leaves the zone, the bar resets.

Usage:
    python main.py --mode mouse
    python main.py --mode mouse --dwell_sec 2.0
    python main.py --mode webcam_async --user_id alice --camera 0
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.keyboard_gui import KeyboardGUI
from src.predictor import Predictor
from src.command_selector import AsyncSelector, SyncSelector


# ── Shared: prediction handler ─────────────────────────────────────────────────

def make_text_change_handler(gui: KeyboardGUI, predictor: Predictor):
    def handler(text: str):
        completions = predictor.get_completions(text, top_k=3)
        gui.update_predictions(completions)
    return handler


# ── Mouse / dwell mode ─────────────────────────────────────────────────────────

def _mouse_zone(screen_w: int, screen_h: int) -> int:
    """Map current mouse position to gaze zone 1–9."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        mx = root.winfo_pointerx()
        my = root.winfo_pointery()
        root.destroy()
    except Exception:
        return 5

    col = min(int(mx / screen_w * 3), 2)  # 0=left, 1=centre, 2=right
    row = min(int(my / screen_h * 3), 2)  # 0=top, 1=centre, 2=bottom
    return row * 3 + col + 1              # 1–9


def run_mouse_mode(args) -> None:
    """
    Dwell-based mouse simulation.

    The user moves the mouse into a zone and holds it still.
    After `dwell_sec` seconds of continuous presence in that zone,
    the command fires. If they move away, the dwell resets.

    A cyan progress bar fills up at the bottom of the active cell.
    """
    predictor = Predictor(
        model_path=args.model_lm,
        word2idx_path=args.word2idx,
    )

    gui = KeyboardGUI(
        prediction_enabled=True,
        tts_enabled=args.tts,
    )
    gui.on_text_change = make_text_change_handler(gui, predictor)
    gui.update_predictions(predictor.get_completions("", top_k=3))

    # Get screen dimensions once
    try:
        import tkinter as tk
        _r = tk.Tk(); _r.withdraw()
        screen_w = _r.winfo_screenwidth()
        screen_h = _r.winfo_screenheight()
        _r.destroy()
    except Exception:
        screen_w, screen_h = 1920, 1080

    dwell_sec    = args.dwell_sec   # seconds to hold in zone before firing
    poll_hz      = 30               # polling rate
    poll_interval = 1.0 / poll_hz

    current_zone   = 5              # zone mouse is currently in
    dwell_start    = time.time()    # when we entered this zone

    def gaze_loop():
        nonlocal current_zone, dwell_start

        while True:
            zone = _mouse_zone(screen_w, screen_h)
            now  = time.time()

            if zone != current_zone:
                # Mouse moved to a new zone — reset dwell
                gui.set_dwell_progress(current_zone, 0.0)
                current_zone = zone
                dwell_start  = now

            # Update highlight
            gui.set_gaze_direction(zone)

            # Update progress bar
            elapsed  = now - dwell_start
            fraction = min(elapsed / dwell_sec, 1.0)
            gui.set_dwell_progress(zone, fraction)

            # Fire when dwell completes
            if fraction >= 1.0:
                gui.fire_command(zone)
                # Reset dwell so it doesn't keep firing
                dwell_start = now + dwell_sec  # force a gap before next fire

            time.sleep(poll_interval)

    t = threading.Thread(target=gaze_loop, daemon=True)
    t.start()
    gui.run()


# ── Webcam modes ───────────────────────────────────────────────────────────────

def run_webcam_mode(args, algorithm: str = "async") -> None:
    # pyrefly: ignore [missing-import]
    import cv2
    from src.eye_detector import EyeDetector
    from src.gaze_predictor import GazePredictor, DIR_NAMES as GAZE_DIR_NAMES

    user_model = Path("models") / f"user_gaze_{args.user_id}.pt"
    base_model  = Path(args.model_gaze)
    model_path  = str(user_model) if user_model.exists() else str(base_model)

    gaze_pred = GazePredictor(model_path=model_path)
    eye_det   = EyeDetector()
    predictor = Predictor(model_path=args.model_lm, word2idx_path=args.word2idx)

    gui = KeyboardGUI(prediction_enabled=True, tts_enabled=args.tts)
    gui.on_text_change = make_text_change_handler(gui, predictor)
    gui.update_predictions(predictor.get_completions("", top_k=3))
    # Pass session metadata so survey can record them
    gui._survey_user_id = args.user_id
    gui._survey_mode    = algorithm + ("_scan" if getattr(args, 'scan_mode', False) else "")

    import collections

    # Use a background thread to read frames, eliminating OpenCV buffer lag
    class CameraStream:
        def __init__(self, camera_id):
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.ret, self.frame = self.cap.read()
            self.running  = True
            self._spf     = 1.0 / 30   # seconds per frame target
            if self.cap.isOpened():
                self.thread = threading.Thread(target=self.update, daemon=True)
                self.thread.start()

        def update(self):
            while self.running:
                t0 = time.time()
                ret, frame = self.cap.read()
                if ret:
                    self.frame = frame
                    self.ret   = ret
                else:
                    # Back off on failure — prevents hammering a stalled driver
                    time.sleep(0.05)
                    continue
                # Throttle to ~30 fps so MSMF/DSHOW driver isn't overwhelmed
                gap = self._spf - (time.time() - t0)
                if gap > 0:
                    time.sleep(gap)

        def read(self):
            return self.ret, self.frame

        def release(self):
            self.running = False
            self.cap.release()

    cap = CameraStream(args.camera)
    if not cap.cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        return

    current_zone     = 5
    dwell_elapsed    = 0.0
    last_frame_time  = time.time()
    blink_start      = None

    # Tunable thresholds
    DWELL_SEC             = args.dwell_sec
    # Natural blink is ~100-400ms (~3-12 frames at 30fps).
    # We require the eye to be CLOSED for at least MIN_BLINK_FRAMES
    # before we count it as intentional — this filters involuntary blinks.
    MIN_BLINK_FRAMES      = 8    # ~0.27s minimum — ignores normal involuntary blinks
    MAX_BLINK_FRAMES      = 60   # ~2s  — longer = lost face, not a blink
    BLINK_CONFIRM_FRAC    = 0.55
    ZONE_LOCK_FRAMES      = 8
    MIN_CONF              = 0.30

    scan_mode             = getattr(args, 'scan_mode', False)
    SCAN_INTERVAL_SEC     = 1.5
    SCAN_ZONES            = list(range(1, 14))  # 1-9 keyboard + 10=pred1 11=pred2 12=pred3 13=done
    # In scan mode use a wider blink window to distinguish intentional from natural
    SCAN_BLINK_MIN        = 8    # ~0.27s
    SCAN_BLINK_MAX        = 45   # ~1.5s  (any longer = looked away)
    SCAN_POST_FIRE_SEC    = 2.0  # freeze scanning after a selection
    last_scan_time        = time.time()
    scan_cooldown_until   = 0.0  # absolute time before scanning resumes
    scan_zone_idx         = 0    # index into SCAN_ZONES

    # Longer smoothing window = smoother but slightly more latency
    history_len   = 25
    probs_history = collections.deque(maxlen=history_len)
    blink_frames  = 0

    candidate_zone        = current_zone
    candidate_zone_frames = 0

    def gaze_loop():
        nonlocal current_zone, dwell_elapsed, last_frame_time
        nonlocal blink_frames, blink_start
        nonlocal candidate_zone, candidate_zone_frames
        nonlocal last_scan_time, scan_cooldown_until, scan_zone_idx

        _preview_counter = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now       = time.time()
            dt        = now - last_frame_time   # real seconds since last frame
            last_frame_time = now

            left, right, annotated = eye_det.detect(frame)
            eyes_ok = (left is not None and right is not None)

            # ── Push annotated frame to embedded Tkinter camera panel ──────────
            # Every 3rd frame only — avoids flooding the GUI event queue.
            _preview_counter += 1
            if _preview_counter % 3 == 0:
                dbg = annotated.copy()
                status_txt = "Eyes: OK" if eyes_ok else "NOT DETECTED - move closer"
                color = (0, 255, 0) if eyes_ok else (0, 0, 255)
                cv2.putText(dbg, status_txt, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                cv2.putText(dbg,
                            f"Zone:{GAZE_DIR_NAMES.get(current_zone,'?')} "
                            f"cand:{GAZE_DIR_NAMES.get(candidate_zone,'?')}x{candidate_zone_frames}",
                            (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 0), 1)
                gui.update_camera_preview(dbg, eyes_ok)

            if not eyes_ok:
                # ── Eyes not detected: possible blink ──────────────────────────
                if blink_frames == 0:
                    blink_start = now
                blink_frames += 1

                if blink_frames > MAX_BLINK_FRAMES:
                    # User just looked away / occluded — reset dwell
                    dwell_elapsed         = 0.0
                    probs_history.clear()
                    candidate_zone        = current_zone
                    candidate_zone_frames = 0
                    gui.set_dwell_progress(current_zone, 0.0)
                    gui._set_status(
                        "NO FACE/EYES DETECTED — move closer, face the light, "
                        f"no glasses glare.  (frames={blink_frames})"
                    )

                time.sleep(0.01)
                continue

            # ── Eyes visible again ────────────────────────────────────────────────
            elapsed_blink = blink_frames
            blink_frames  = 0
            blink_start   = None

            # ── Deliberate blink detection ────────────────────────────────────────
            if scan_mode:
                # Scan mode: blink window is SCAN_BLINK_MIN..SCAN_BLINK_MAX
                # Natural blinks (< MIN_BLINK_FRAMES) are silently ignored.
                if SCAN_BLINK_MIN <= elapsed_blink <= SCAN_BLINK_MAX:
                    if now >= scan_cooldown_until:  # not in post-fire freeze
                        gui.fire_command(current_zone)
                        # Freeze scanning for SCAN_POST_FIRE_SEC after selection
                        scan_cooldown_until = now + SCAN_POST_FIRE_SEC
                        last_scan_time      = now + SCAN_POST_FIRE_SEC
                # Don't continue — fall through to update display below
            else:
                # Normal gaze mode: blink can confirm current dwell
                if MIN_BLINK_FRAMES <= elapsed_blink <= MAX_BLINK_FRAMES:
                    dwell_frac = min(dwell_elapsed / DWELL_SEC, 1.0)
                    if dwell_frac >= BLINK_CONFIRM_FRAC:
                        gui.fire_command(current_zone)
                        dwell_elapsed = -DWELL_SEC  # Cooldown before next fire
                        gui.set_dwell_progress(current_zone, 0.0)
                        continue

            # ── Normal gaze / scan display ────────────────────────────────────────
            if scan_mode:
                if now >= scan_cooldown_until:
                    elapsed_scan = now - last_scan_time
                    if elapsed_scan >= SCAN_INTERVAL_SEC:
                        # Advance to next zone in the full 1-13 cycle
                        scan_zone_idx  = (scan_zone_idx + 1) % len(SCAN_ZONES)
                        current_zone   = SCAN_ZONES[scan_zone_idx]
                        last_scan_time = now
                        elapsed_scan   = 0.0
                        # Clear dwell on all zones so only current zone shows progress
                        for z in SCAN_ZONES:
                            gui.set_dwell_progress(z, 0.0)
                    fraction = min(elapsed_scan / SCAN_INTERVAL_SEC, 1.0)
                else:
                    # Post-selection freeze: clear all bars and show countdown
                    fraction = 0.0
                    for z in SCAN_ZONES:
                        gui.set_dwell_progress(z, 0.0)

                gui.set_gaze_direction(current_zone)
                gui.set_dwell_progress(current_zone, fraction)

                zone_name = {
                    10: "Prediction 1", 11: "Prediction 2",
                    12: "Prediction 3", 13: "Done Typing"
                }.get(current_zone, f"zone {current_zone}")

                if now < scan_cooldown_until:
                    remaining = scan_cooldown_until - now
                    gui._set_status(f"SCAN MODE — Selected! Next scan in {remaining:.1f}s...")
                else:
                    gui._set_status(
                        f"SCAN MODE → [{zone_name}] — Blink 0.27-1.5s to select "
                        f"| Next in {(1.0 - fraction) * SCAN_INTERVAL_SEC:.1f}s"
                    )
            else:
                Lp, Rp, left_probs, right_probs = gaze_pred.predict(left, right)
    
                # 1. Soft voting between left and right eye
                avg_probs = (left_probs + right_probs) / 2.0
    
                # 2. Moving average smoothing across time
                probs_history.append(avg_probs)
                smoothed_probs = sum(probs_history) / len(probs_history)
    
                # 3. Raw predicted zone and its confidence
                raw_zone = int(smoothed_probs.argmax()) + 1
                conf     = float(smoothed_probs[raw_zone - 1])
    
                # 4. Zone-stability filter: only switch zones after ZONE_LOCK_FRAMES
                #    consecutive frames agree on the new zone AND confidence is high enough
                if raw_zone == candidate_zone:
                    candidate_zone_frames += 1
                else:
                    candidate_zone        = raw_zone
                    candidate_zone_frames = 1
    
                zone_switched = False
                # Accept zone switch when stable and confident enough
                if (candidate_zone != current_zone
                        and candidate_zone_frames >= ZONE_LOCK_FRAMES
                        and conf >= MIN_CONF):
                    gui.set_dwell_progress(current_zone, 0.0)
                    current_zone          = candidate_zone
                    dwell_elapsed         = 0.0   # reset accumulated time on zone change
                    candidate_zone_frames = 0
                    zone_switched         = True
    
                gui.set_gaze_direction(current_zone)
    
                # 5. Update dwell bar (advance by real elapsed time)
                if not zone_switched:
                    dwell_elapsed = min(dwell_elapsed + dt, DWELL_SEC)
                
                # fraction is 0 during the cooldown period
                fraction = max(0.0, min(dwell_elapsed / DWELL_SEC, 1.0))
                gui.set_dwell_progress(current_zone, fraction)
    
                # 6. Show live debug info in status bar
                gui._set_status(
                    f"Gaze: {GAZE_DIR_NAMES.get(current_zone,'?')} (zone {current_zone})  "
                    f"conf={conf:.0%}  candidate={GAZE_DIR_NAMES.get(candidate_zone,'?')}x{candidate_zone_frames}  "
                    f"dwell={fraction:.0%}"
                )
    
                # 7. Auto-fire on full dwell
                if fraction >= 1.0:
                    gui.fire_command(current_zone)
                    dwell_elapsed         = -DWELL_SEC  # Apply cooldown to prevent double-firing
                    candidate_zone_frames = 0

        cap.release()

    start_event = threading.Event()
    def start_cb(event=None):
        if not start_event.is_set():
            start_event.set()
            gui._set_status(f"Webcam tracking active — hold gaze for {DWELL_SEC:.1f}s to select.")

    gui.root.bind("<Return>", start_cb)
    gui.root.bind("<space>", start_cb)
    gui._set_status("POSITION YOURSELF IN FRONT OF WEBCAM. PRESS [ENTER] or [SPACE] TO START.")

    def gaze_loop_wrapped():
        start_event.wait()
        gaze_loop()

    t = threading.Thread(target=gaze_loop_wrapped, daemon=True)
    t.start()
    gui.run()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gaze-Controlled Virtual Keyboard + LSTM Prediction"
    )
    parser.add_argument("--mode", default=None,
                        choices=["mouse", "webcam_async", "webcam_sync", "calibrate"])
    parser.add_argument("--user_id",    default="default")
    parser.add_argument("--camera",     type=int, default=0)
    parser.add_argument("--model_gaze", default="models/base_gaze_model.pt")
    parser.add_argument("--model_lm",   default="models/lstm_lm.pt")
    parser.add_argument("--word2idx",   default="models/word2idx.json")
    parser.add_argument("--delta_t",    type=int, default=75,
                        help="Webcam: frames for agreement window (default=75 @ 30fps=2.5s)")
    parser.add_argument("--alpha",      type=float, default=6.0)
    parser.add_argument("--dwell_sec",  type=float, default=2.5,
                        help="Seconds to hold in a zone before firing (default=2.5)")
    parser.add_argument("--scan_mode",  action="store_true",
                        help="Enable auto-scan mode for switch scanning")
    parser.add_argument("--tts",        action="store_true")
    parser.add_argument("--admin_view_survey", action="store_true",
                        help="View collected survey data (requires admin password)")
    args = parser.parse_args()

    # Admin survey viewer — runs without starting the GUI
    if args.admin_view_survey:
        import getpass
        from src.survey import load_all, print_summary, verify_admin
        pwd = getpass.getpass("Enter admin password: ")
        try:
            records = load_all(pwd)
            print_summary(records)
        except PermissionError as e:
            print(f"\n  Access denied: {e}\n")
        return

    if args.mode == "mouse":
        run_mouse_mode(args)
    elif args.mode == "webcam_async":
        run_webcam_mode(args, algorithm="async")
    elif args.mode == "webcam_sync":
        run_webcam_mode(args, algorithm="sync")
    elif args.mode == "calibrate":
        from calibrate import run_calibration
        run_calibration(args.user_id, args.camera, args.model_gaze)


if __name__ == "__main__":
    main()
