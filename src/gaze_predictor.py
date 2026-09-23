"""
gaze_predictor.py — Load trained CNN and predict gaze direction

Wraps the GazeCNN PyTorch model for real-time inference.
Handles both webcam mode (eye crops from EyeDetector) and
mouse-simulation mode (returns a fixed direction for testing).

Usage:
    from gaze_predictor import GazePredictor
    predictor = GazePredictor("models/base_gaze_model.pt")
    direction = predictor.predict(left_eye_crop, right_eye_crop)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow imports from src/ when called from repo root
sys.path.insert(0, str(Path(__file__).parent))
from train_gaze import GazeCNN

DIR_NAMES = {
    1: "NW", 2: "N",  3: "NE",
    4: "W",  5: "C",  6: "E",
    7: "SW", 8: "S",  9: "SE",
}


class GazePredictor:
    """
    Loads a trained GazeCNN checkpoint and predicts gaze direction.

    Args:
        model_path: path to .pt checkpoint saved by train_gaze.py
        device:     "auto" | "cuda" | "cpu"
        num_classes: 9 (default)
    """

    def __init__(
        self,
        model_path: str = "models/base_gaze_model.pt",
        device: str = "auto",
        num_classes: int = 9,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GazeCNN(num_classes=num_classes).to(self.device)
        self.model.eval()

        ckpt_path = Path(model_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"[GazePredictor] Loaded checkpoint: {ckpt_path} "
                  f"(epoch={ckpt.get('epoch','?')}, "
                  f"val_acc={ckpt.get('val_acc',0):.3f})")
        else:
            print(f"[GazePredictor] WARNING: checkpoint not found at {ckpt_path}. "
                  "Using random weights (mock mode).")

    def predict(
        self,
        left_eye: np.ndarray | None,
        right_eye: np.ndarray | None,
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """
        Predict gaze direction from left and right eye crops.

        Args:
            left_eye:  100×100 uint8 or float32 grayscale crop (or None)
            right_eye: 100×100 uint8 or float32 grayscale crop (or None)

        Returns:
            (Lp, Rp, left_probs, right_probs)
            Lp, Rp: predicted class index 1–9 for each eye
            left_probs, right_probs: float32 probability vectors (length 9)
        """
        def _preprocess(eye: np.ndarray | None) -> torch.Tensor:
            if eye is None:
                # Return uniform distribution input (centre=C=5 will win)
                return torch.zeros(1, 1, 100, 100, device=self.device)
            arr = eye.astype(np.float32) / 255.0 if eye.max() > 1.0 else eye.astype(np.float32)
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            return tensor.to(self.device)

        with torch.no_grad():
            left_tensor  = _preprocess(left_eye)
            right_tensor = _preprocess(right_eye)

            left_logits  = self.model(left_tensor)
            right_logits = self.model(right_tensor)

            left_probs  = F.softmax(left_logits, dim=1).squeeze().cpu().numpy()
            right_probs = F.softmax(right_logits, dim=1).squeeze().cpu().numpy()

        Lp = int(left_probs.argmax()) + 1   # 1-indexed
        Rp = int(right_probs.argmax()) + 1

        return Lp, Rp, left_probs, right_probs


class MockGazePredictor:
    """
    Mouse-controlled mock predictor for GUI testing without a trained model.
    Reads the current mouse position and maps it to a 3×3 gaze direction.
    """

    def __init__(self):
        print("[MockGazePredictor] Running in mouse-simulation mode.")
        self._forced_dir: int | None = None

    def force_direction(self, direction: int) -> None:
        """Override with a fixed direction (1–9) for automated testing."""
        self._forced_dir = direction

    def predict(self, *args, **kwargs) -> tuple[int, int, np.ndarray, np.ndarray]:
        import tkinter as tk

        if self._forced_dir is not None:
            d = self._forced_dir
            probs = np.zeros(9, dtype=np.float32)
            probs[d - 1] = 1.0
            return d, d, probs, probs

        try:
            root = tk.Tk()
            root.withdraw()
            mx = root.winfo_pointerx()
            my = root.winfo_pointery()
            sw = root.winfo_screenwidth()
            sh = root.winfo_screenheight()
            root.destroy()

            col = min(int(mx / sw * 3), 2)
            row = min(int(my / sh * 3), 2)
            d   = row * 3 + col + 1  # 1–9

        except Exception:
            d = 5  # default: centre

        probs = np.zeros(9, dtype=np.float32)
        probs[d - 1] = 0.9
        return d, d, probs, probs
