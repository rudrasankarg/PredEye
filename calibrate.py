"""
calibrate.py — 9-point calibration window + CNN fine-tuning

Shows 9 gaze targets one at a time (grid positions NW…SE).
For each target, captures 300 webcam frames of the user's eyes
and saves them to data/calibration/user_<id>/<direction>/<frame>.jpg.

After collection, immediately fine-tunes the base_gaze_model.pt on the
calibration data and saves user_gaze_<id>.pt.

Usage:
    python calibrate.py --user_id alice --camera 0 --model models/base_gaze_model.pt
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from src.eye_detector import EyeDetector

# ── Calibration config ─────────────────────────────────────────────────────────

CAPTURES_PER_DIR = 300
COUNTDOWN_SECS   = 2     # pause before capturing at each target
IMAGE_SIZE       = 100

DIR_NAMES = {
    1: "NW", 2: "N",  3: "NE",
    4: "W",  5: "C",  6: "E",
    7: "SW", 8: "S",  9: "SE",
}

# Target dot positions on screen (normalised 0–1, col/row)
TARGET_POSITIONS = {
    1: (0.15, 0.15), 2: (0.50, 0.15), 3: (0.85, 0.15),
    4: (0.15, 0.50), 5: (0.50, 0.50), 6: (0.85, 0.50),
    7: (0.15, 0.80), 8: (0.50, 0.80), 9: (0.85, 0.80),
}


def run_calibration(user_id: str, camera_id: int, model_path: str) -> None:
    detector = EyeDetector(image_size=IMAGE_SIZE)
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Cannot open camera {camera_id}")
        return

    # Get screen size via OpenCV window trick
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    ret, probe = cap.read()
    if not ret:
        print("Cannot read from camera.")
        return
    screen_h, screen_w = 900, 1600  # fallback; override below if detectable

    # Try to get actual screen resolution
    try:
        import tkinter as tk
        _root = tk.Tk(); _root.withdraw()
        screen_w = _root.winfo_screenwidth()
        screen_h = _root.winfo_screenheight()
        _root.destroy()
    except Exception:
        pass

    for direction in range(1, 10):
        out_dir = Path(f"data/calibration/user_{user_id}/{direction}")
        out_dir.mkdir(parents=True, exist_ok=True)

        nx, ny = TARGET_POSITIONS[direction]
        tx = int(nx * screen_w)
        ty = int(ny * screen_h)

        # Wait for user to be ready
        while True:
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.putText(canvas,
                        f"Direction {direction}: {DIR_NAMES[direction]}",
                        (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
            cv2.putText(canvas,
                        "Look at the green dot, then press SPACE to capture.",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.circle(canvas, (tx, ty), 30, (0, 255, 0), -1)
            cv2.imshow("Calibration", canvas)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Capture frames
        n_captured = 0
        while n_captured < CAPTURES_PER_DIR:
            ret, frame = cap.read()
            if not ret:
                continue

            left, right, _ = detector.detect(frame)

            # Draw calibration canvas
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.putText(canvas,
                        f"Capturing {n_captured}/{CAPTURES_PER_DIR}  "
                        f"[{DIR_NAMES[direction]}]  — KEEP LOOKING AT THE DOT",
                        (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.circle(canvas, (tx, ty), 25, (0, 200, 255), -1)

            if left is not None and right is not None:
                cv2.imwrite(str(out_dir / f"L_{n_captured:04d}.jpg"), left)
                cv2.imwrite(str(out_dir / f"R_{n_captured:04d}.jpg"), right)
                n_captured += 1
            else:
                cv2.putText(canvas, "No eyes detected — adjust position",
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        print(f"  Direction {direction} ({DIR_NAMES[direction]}): {n_captured} images")

    cap.release()
    cv2.destroyAllWindows()
    print("\nCalibration images collected. Starting fine-tuning…")

    _fine_tune(user_id, model_path)


def _fine_tune(user_id: str, base_model_path: str) -> None:
    """Fine-tune the base CNN on this user's calibration data."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from src.train_gaze import GazeCNN

    calib_dir = f"data/calibration/user_{user_id}"
    out_path  = Path("models") / f"user_gaze_{user_id}.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on {device}…")

    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(calib_dir, transform=tf)
    n_val   = max(1, int(len(dataset) * 0.25))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=128, shuffle=False)

    model = GazeCNN().to(device)
    ckpt  = Path(base_model_path)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"  Loaded base model from {ckpt}")

    # Fine-tune the entire network to adapt to the real webcam domain
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    # Optimize all layers with a lower learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    for epoch in range(1, 71):
        model.train()
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total   += len(labels)
        acc = correct / total
        print(f"  Fine-tune epoch {epoch:2d}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_acc": acc, "user_id": user_id,
            }, out_path)

    print(f"\nFine-tuning complete. Best val_acc={best_acc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="9-point calibration + fine-tune")
    parser.add_argument("--user_id", default="default")
    parser.add_argument("--camera",  type=int, default=0)
    parser.add_argument("--model",   default="models/base_gaze_model.pt")
    args = parser.parse_args()

    run_calibration(args.user_id, args.camera, args.model)
