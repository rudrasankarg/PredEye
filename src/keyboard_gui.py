"""
keyboard_gui.py — Gaze-controlled virtual keyboard with LSTM prediction slots

Layout (Tkinter):
    ┌─────────────────────────────────┐
    │     [PRED1]  [PRED2]  [PRED3]   │  ← prediction row (our novel addition)
    ├──────────────────────────────────┤
    │  NW    │   N    │   NE          │
    │  W     │   C    │   E           │  ← 3×3 command grid with dwell bars
    │  SW    │   S    │   SE          │
    ├──────────────────────────────────┤
    │        Typed text display        │
    └─────────────────────────────────┘

Dwell time: each cell shows a filling progress bar at the bottom.
The command fires only after the gaze stays in the zone for the full dwell period.
"""

import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from typing import Callable, Optional

try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ── Keyboard layout ────────────────────────────────────────────────────────────

LETTER_GROUPS: dict[int, str] = {
    1: "ABCD",
    2: "EFGH",
    3: "IJKL",
    4: "MNO",
    5: "SPACE",     # Centre = SPACE at top level; GO BACK at level 2
    6: "PQR",
    7: "STUV",
    8: "WXYZ",
    9: "DELETE",
}

DIR_POSITIONS: dict[int, tuple] = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
    7: (2, 0), 8: (2, 1), 9: (2, 2),
}

# ── Colour palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "bg":           "#1a1a2e",
    "cell_idle":    "#16213e",
    "cell_hover":   "#0f3460",
    "cell_active":  "#e94560",
    "pred_bg":      "#0d2137",
    "pred_hover":   "#533483",
    "pred_active":  "#e94560",
    "text_primary": "#eaeaea",
    "text_dim":     "#7f8c8d",
    "accent":       "#00d2ff",
    "typed_bg":     "#0a0a1a",
    "dwell_bar":    "#00d2ff",
    "dwell_track":  "#0a0a1a",
}


class KeyboardGUI:
    """
    Main gaze keyboard window.

    External interface:
        gui.set_gaze_direction(d)         -> highlight cell d
        gui.set_dwell_progress(d, frac)   -> update dwell bar 0.0–1.0 for cell d
        gui.fire_command(d)               -> execute command at direction d
        gui.update_predictions(words)     -> update 3 prediction buttons
        gui.set_text(text)                -> set typed text directly
    """

    def __init__(
        self,
        on_text_change: Optional[Callable] = None,
        prediction_enabled: bool = True,
        tts_enabled: bool = True,
    ):
        self.on_text_change    = on_text_change
        self.prediction_enabled = prediction_enabled
        self.tts_enabled       = tts_enabled

        self._typed_text  = ""
        self._level       = 1
        self._selected_group = 0
        self._current_cells: dict[int, str] = {}
        self._gaze_dir    = 5
        self._pred_labels = ["", "", ""]
        self._last_fire_time = 0.0   # cooldown: prevents double-fires
        self._fire_cooldown  = 0.6   # seconds between allowed fires
        self._is_done        = False # locks input when done typing

        # TTS
        self._tts = None
        if tts_enabled:
            try:
                # pyrefly: ignore [missing-import]
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 160)
            except Exception:
                pass

        self._build_window()
        self._set_level1()

    # ── Window construction ────────────────────────────────────────────────────

    def _build_window(self) -> None:
        self.root = tk.Tk()
        self.root.title("Gaze Keyboard — Mouse Mode (move mouse to zone, hold to select)")
        self.root.configure(bg=PALETTE["bg"])
        self.root.resizable(True, True)

        # Start maximized on Windows
        try:
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1100x700")

        # Fonts
        try:
            mono      = tkfont.Font(family="Consolas",  size=13, weight="bold")
            hdr       = tkfont.Font(family="Segoe UI",  size=10, weight="bold")
            big       = tkfont.Font(family="Consolas",  size=18, weight="bold")
            pred_font = tkfont.Font(family="Segoe UI",  size=12)
            small     = tkfont.Font(family="Segoe UI",  size=9)
        except Exception:
            mono = hdr = big = pred_font = small = tkfont.Font(size=11)

        self._font_mono = mono
        self._font_hdr  = hdr
        self._font_big  = big
        self._font_pred = pred_font
        self._font_small = small

        # ── Prediction row ─────────────────────────────────────────────────────
        pred_outer = tk.Frame(self.root, bg=PALETTE["pred_bg"])
        pred_outer.pack(fill="x", padx=12, pady=(10, 0))

        tk.Label(pred_outer, text="PREDICTIONS",
                 fg=PALETTE["text_dim"], bg=PALETTE["pred_bg"], font=hdr
                 ).pack(side="left", padx=(10, 0))

        self._pred_buttons: list[tk.Button] = []
        for i in range(3):
            btn = tk.Button(
                pred_outer,
                text="[PRED]",
                fg=PALETTE["text_primary"],
                bg=PALETTE["pred_bg"],
                activebackground=PALETTE["pred_active"],
                activeforeground="#fff",
                relief="flat",
                font=pred_font,
                width=16,
                pady=6,
                cursor="hand2",
                command=lambda idx=i: self._on_prediction_click(idx),
            )
            btn.pack(side="left", padx=8, pady=4)
            self._pred_buttons.append(btn)

        self._toggle_pred_btn = tk.Button(
            pred_outer,
            text="PRED: ON" if self.prediction_enabled else "PRED: OFF",
            fg=PALETTE["accent"],
            bg=PALETTE["pred_bg"],
            activebackground=PALETTE["cell_hover"],
            relief="flat",
            font=hdr,
            command=self._toggle_predictions,
            cursor="hand2",
        )
        self._toggle_pred_btn.pack(side="right", padx=10)

        self._done_btn = tk.Button(
            pred_outer,
            text="DONE TYPING",
            fg="#ffffff",
            bg="#27ae60",
            activebackground="#2ecc71",
            relief="flat",
            font=hdr,
            command=self._finish_typing,
            cursor="hand2",
            padx=10
        )
        self._done_btn.pack(side="right", padx=(0, 10))

        # ── 3×3 grid ───────────────────────────────────────────────────────────
        grid_frame = tk.Frame(self.root, bg=PALETTE["bg"])
        grid_frame.pack(fill="both", expand=True, padx=12, pady=10)

        for col in range(3):
            grid_frame.columnconfigure(col, weight=1)
        for row in range(3):
            grid_frame.rowconfigure(row, weight=1)

        self._cell_frames:    dict[int, tk.Frame]  = {}
        self._cell_labels:    dict[int, tk.Label]  = {}
        self._cell_sublabels: dict[int, tk.Label]  = {}
        self._dwell_canvases: dict[int, tk.Canvas] = {}
        self._dwell_bars:     dict[int, int]       = {}  # canvas item ids

        for direction, (row, col) in DIR_POSITIONS.items():
            outer = tk.Frame(grid_frame, bg=PALETTE["cell_idle"], bd=2, relief="flat")
            outer.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Direction number
            dir_lbl = tk.Label(outer, text=str(direction),
                                fg=PALETTE["text_dim"], bg=PALETTE["cell_idle"],
                                font=hdr, anchor="nw")
            dir_lbl.place(x=5, y=3)

            # Main content label
            content_lbl = tk.Label(outer, text="",
                                   fg=PALETTE["text_primary"], bg=PALETTE["cell_idle"],
                                   font=mono, wraplength=200, justify="center")
            content_lbl.place(relx=0.5, rely=0.42, anchor="center")

            # Sub-label
            sub_lbl = tk.Label(outer, text="",
                                fg=PALETTE["text_dim"], bg=PALETTE["cell_idle"],
                                font=small)
            sub_lbl.place(relx=0.5, rely=0.72, anchor="center")

            # Dwell progress bar (Canvas at bottom of each cell)
            dwell_canvas = tk.Canvas(outer, height=8, bg=PALETTE["dwell_track"],
                                      highlightthickness=0)
            dwell_canvas.place(relx=0, rely=1.0, anchor="sw", relwidth=1.0)
            bar_item = dwell_canvas.create_rectangle(0, 0, 0, 8,
                                                      fill=PALETTE["dwell_bar"],
                                                      outline="")

            self._cell_frames[direction]    = outer
            self._cell_labels[direction]    = content_lbl
            self._cell_sublabels[direction] = sub_lbl
            self._dwell_canvases[direction] = dwell_canvas
            self._dwell_bars[direction]     = bar_item

        # ── Typed text area ────────────────────────────────────────────────────
        text_frame = tk.Frame(self.root, bg=PALETTE["typed_bg"])
        text_frame.pack(fill="x", padx=12, pady=(0, 10))

        tk.Label(text_frame, text="Typed:", fg=PALETTE["text_dim"],
                 bg=PALETTE["typed_bg"], font=hdr).pack(side="left", padx=(8, 0))

        self._typed_display = tk.Label(
            text_frame, text="▌",
            fg=PALETTE["accent"], bg=PALETTE["typed_bg"],
            font=big, anchor="w",
        )
        self._typed_display.pack(side="left", fill="x", expand=True, padx=(8, 0))

        # Status bar
        self._status_bar = tk.Label(
            self.root,
            text="Move mouse to a zone and hold still to select",
            fg=PALETTE["text_dim"], bg=PALETTE["bg"],
            font=small, anchor="w",
        )
        self._status_bar.pack(fill="x", padx=12, pady=(0, 4))

        # ── Camera preview panel ────────────────────────────────────────────────
        cam_outer = tk.Frame(self.root, bg="#000", bd=1, relief="solid")
        cam_outer.place(relx=1.0, rely=1.0, anchor="se", x=-12, y=-30,
                        width=220, height=165)
        tk.Label(cam_outer, text="CAMERA PREVIEW",
                 fg=PALETTE["text_dim"], bg="#000",
                 font=small).pack()
        self._cam_canvas = tk.Canvas(cam_outer, width=220, height=150,
                                     bg="#111", highlightthickness=0)
        self._cam_canvas.pack()
        self._cam_photo = None   # keep reference to prevent GC
        self._cam_status_text = self._cam_canvas.create_text(
            110, 75, text="Press Enter/Space\nto start camera",
            fill=PALETTE["text_dim"], font=small, justify="center"
        )

    # ── Level management ───────────────────────────────────────────────────────

    def _set_level1(self) -> None:
        self._level = 1
        self._current_cells = {d: LETTER_GROUPS[d] for d in range(1, 10)}
        self._refresh_cells()

    def _set_level2(self, group_dir: int) -> None:
        self._level = 2
        self._selected_group = group_dir
        letters = list(LETTER_GROUPS[group_dir])
        positions = [d for d in range(1, 10) if d != 5]
        self._current_cells = {}
        for i, d in enumerate(positions):
            self._current_cells[d] = letters[i] if i < len(letters) else ""
        self._current_cells[5] = "<< BACK"   # GO BACK only at level 2
        self._refresh_cells()

    def _refresh_cells(self) -> None:
        for direction in range(1, 10):
            label = self._current_cells.get(direction, "")
            self._cell_labels[direction].config(text=label)
            if self._level == 1 and direction not in (5, 9):
                self._cell_sublabels[direction].config(text=f"({len(label)} letters)")
            else:
                self._cell_sublabels[direction].config(text="")
        # Reset all dwell bars when level changes
        self.root.after(0, lambda: [self.set_dwell_progress(d, 0.0) for d in range(1, 10)])

    # ── Gaze / dwell interface ─────────────────────────────────────────────────

    def set_gaze_direction(self, direction: int) -> None:
        """Highlight the focused cell. Thread-safe."""
        if direction == self._gaze_dir:
            return
        self._gaze_dir = direction
        self.root.after(0, self._update_cell_highlight)

    def _update_cell_highlight(self) -> None:
        for d, frame in self._cell_frames.items():
            is_focused = (d == self._gaze_dir)
            color = PALETTE["cell_hover"] if is_focused else PALETTE["cell_idle"]
            frame.config(bg=color)
            self._cell_labels[d].config(bg=color)
            self._cell_sublabels[d].config(bg=color)

    def set_dwell_progress(self, direction: int, fraction: float) -> None:
        """
        Update the dwell progress bar for a cell.
        fraction: 0.0 = empty, 1.0 = full (fires command).
        Thread-safe.
        """
        self.root.after(0, lambda: self._draw_dwell(direction, fraction))

    def _draw_dwell(self, direction: int, fraction: float) -> None:
        canvas = self._dwell_canvases.get(direction)
        bar    = self._dwell_bars.get(direction)
        if canvas is None or bar is None:
            return
        width = canvas.winfo_width()
        if width <= 1:
            width = 200  # fallback before layout is done
        fill_w = max(0, min(width, int(width * fraction)))
        canvas.coords(bar, 0, 0, fill_w, 8)
        # Colour: cyan -> red as it fills
        if fraction < 0.6:
            color = PALETTE["dwell_bar"]
        elif fraction < 0.9:
            color = "#f39c12"  # orange
        else:
            color = PALETTE["cell_active"]  # red — about to fire
        canvas.itemconfig(bar, fill=color)

    # ── Command execution ──────────────────────────────────────────────────────

    def fire_command(self, direction: int) -> None:
        """Execute the command at position `direction`. Thread-safe.
        Ignores rapid duplicate calls within the cooldown window.
        """
        import time as _time
        if getattr(self, "_is_done", False):
            return
        now = _time.time()
        if now - self._last_fire_time < self._fire_cooldown:
            return   # too soon — debounce
        self._last_fire_time = now
        self.root.after(0, lambda: self._execute_command(direction))

    def _execute_command(self, direction: int) -> None:
        # Flash cell
        frame = self._cell_frames.get(direction)
        if frame:
            frame.config(bg=PALETTE["cell_active"])
            self._cell_labels[direction].config(bg=PALETTE["cell_active"])
            self.root.after(400, self._update_cell_highlight)

        # Reset dwell bar
        self.set_dwell_progress(direction, 0.0)

        label = self._current_cells.get(direction, "")

        if self._level == 1:
            if direction == 9:             # DELETE
                self._typed_text = self._typed_text[:-1]
                self._flush_display()
                self._set_status("Deleted last character")
                if self.on_text_change:
                    self.on_text_change(self._typed_text)
            elif direction == 5:           # SPACE
                self._typed_text += " "
                self._flush_display()
                self._speak("Space")
                self._set_status("Space added")
                if self.on_text_change:
                    self.on_text_change(self._typed_text)
            else:                          # letter group
                self._set_level2(direction)
                self._speak(f"Group {label}")
                self._set_status(f"Selected group: {label}  — now pick a letter")
        else:  # level 2
            if direction == 5 or label in ("GO BACK", "<< BACK", "BACK"):
                self._set_level1()
                self._speak("Back")
                self._set_status("Back to main menu")
            elif label:   # valid letter
                self._type_character(label)
                self._set_level1()
            # else: empty cell — do nothing, don't change level

    def _type_character(self, char: str) -> None:
        self._typed_text += char
        self._flush_display()
        self._speak(char)
        self._set_status(f"Typed: {char}")
        if self.on_text_change:
            self.on_text_change(self._typed_text)

    def _flush_display(self) -> None:
        self._typed_display.config(text=self._typed_text + "|")

    def _set_status(self, msg: str) -> None:
        self._status_bar.config(text=msg)

    # ── Prediction slots ───────────────────────────────────────────────────────

    def update_predictions(self, completions: list) -> None:
        self._pred_labels = (completions + ["", "", ""])[:3]
        self.root.after(0, self._refresh_predictions)

    def _refresh_predictions(self) -> None:
        for i, btn in enumerate(self._pred_buttons):
            word = self._pred_labels[i]
            btn.config(
                text=word,
                state="normal" if (word and self.prediction_enabled) else "disabled",
                bg=PALETTE["pred_bg"],
            )

    def _on_prediction_click(self, idx: int) -> None:
        word = self._pred_labels[idx]
        if not word or not self.prediction_enabled:
            return
        parts = self._typed_text.split()
        if self._typed_text and not self._typed_text.endswith(" "):
            parts = parts[:-1]
        parts.append(word)
        self._typed_text = " ".join(parts) + " "
        self._flush_display()
        self._speak(word)
        self._set_status(f"Prediction selected: {word}")
        if self.on_text_change:
            self.on_text_change(self._typed_text)

    def set_prediction_hover(self, slot: int) -> None:
        self.root.after(0, lambda: self._highlight_pred(slot))

    def _highlight_pred(self, slot: int) -> None:
        for i, btn in enumerate(self._pred_buttons):
            btn.config(bg=PALETTE["pred_hover"] if i == slot else PALETTE["pred_bg"])

    def fire_prediction(self, slot: int) -> None:
        self.root.after(0, lambda: self._on_prediction_click(slot))

    def _toggle_predictions(self) -> None:
        self.prediction_enabled = not self.prediction_enabled
        self._toggle_pred_btn.config(
            text="PRED: ON" if self.prediction_enabled else "PRED: OFF"
        )
        self._refresh_predictions()

    # ── TTS ───────────────────────────────────────────────────────────────────

    def _speak(self, text: str) -> None:
        if self._tts and self.tts_enabled:
            try:
                self._tts.say(text)
                threading.Thread(target=self._tts.runAndWait, daemon=True).start()
            except Exception:
                pass

    def _finish_typing(self) -> None:
        if self._is_done: return
        self._is_done = True
        self._speak("Typing finished")
        
        # Hide all main UI elements
        for widget in self.root.winfo_children():
            widget.pack_forget()
            
        final_frame = tk.Frame(self.root, bg=PALETTE["bg"])
        final_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        tk.Label(final_frame, text="✅ FINAL TEXT", fg="#27ae60", bg=PALETTE["bg"], font=self._font_big).pack(pady=(50, 20))
        
        # Large prominent text display
        text_lbl = tk.Label(final_frame, text=self._typed_text, fg="#ffffff", bg=PALETTE["bg"], font=self._font_big, wraplength=900, justify="center")
        text_lbl.pack(pady=40, expand=True)
        
        tk.Button(final_frame, text="EXIT APP", font=self._font_hdr, bg="#e94560", fg="#fff", command=self.root.quit, padx=20, pady=10).pack(pady=40)

    # ── Public helpers ────────────────────────────────────────────────────────

    def set_text(self, text: str) -> None:
        self._typed_text = text
        self.root.after(0, self._flush_display)

    def get_text(self) -> str:
        return self._typed_text

    # ── Camera preview (called from gaze thread, renders on main thread) ──────

    def update_camera_preview(self, bgr_frame, eyes_detected: bool) -> None:
        """Thread-safe. Pass the annotated BGR numpy frame every N frames."""
        self.root.after(0, lambda: self._draw_camera_frame(bgr_frame, eyes_detected))

    def _draw_camera_frame(self, bgr_frame, eyes_detected: bool) -> None:
        import numpy as _np
        try:
            h, w = bgr_frame.shape[:2]
            scale = min(220 / w, 150 / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb = bgr_frame[:, :, ::-1]   # BGR -> RGB

            if _PIL_AVAILABLE:
                from PIL import Image as _Img, ImageTk as _ITk
                img   = _Img.fromarray(rgb).resize((nw, nh), _Img.BILINEAR)
                photo = _ITk.PhotoImage(img)
            else:
                # pyrefly: ignore [missing-import]
                import cv2 as _cv2
                resized = _cv2.resize(rgb, (nw, nh))
                header  = f"P6\n{nw} {nh}\n255\n".encode()
                photo   = tk.PhotoImage(data=header + resized.tobytes())

            self._cam_photo = photo   # prevent GC
            self._cam_canvas.delete("all")
            self._cam_canvas.create_image(110, 75, image=photo, anchor="center")

            border = "#27ae60" if eyes_detected else "#e74c3c"
            self._cam_canvas.create_rectangle(0, 0, 219, 149,
                                              outline=border, width=3)
            label  = "Eyes OK" if eyes_detected else "NO EYES - move closer"
            lcolor = "#2ecc71" if eyes_detected else "#e74c3c"
            self._cam_canvas.create_text(110, 142, text=label,
                                         fill=lcolor, font=self._font_small)
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()



# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Keyboard GUI demo — click prediction buttons to test")

    def on_change(text: str):
        print(f"Text: {text!r}")

    gui = KeyboardGUI(on_text_change=on_change, tts_enabled=False)
    gui.update_predictions(["hello", "world", "python"])
    gui.run()
