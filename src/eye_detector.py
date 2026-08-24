"""
eye_detector.py — Haar cascade face + eye crop using OpenCV

Takes a BGR frame from the webcam, detects the face, extracts both eyes,
returns them as 100×100 grayscale crops ready for the gaze CNN.

Usage (standalone test):
    python src/eye_detector.py --camera 0
"""

import argparse
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# ── Cascade paths ──────────────────────────────────────────────────────────────
# Pointing to locally downloaded cascades in models/
_MODELS_DIR = Path(__file__).parent.parent / "models"

FACE_CASCADE_PATH = _MODELS_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH  = _MODELS_DIR / "haarcascade_eye.xml"


class EyeDetector:
    """
    Detects left and right eye crops from a webcam frame.

    Face -> eye region -> 100×100 grayscale crop.
    Returns None for missing eyes (e.g., detection failure).
    """

    def __init__(self, image_size: int = 100, scale_factor: float = 1.3,
                 min_neighbours: int = 5):
        if not FACE_CASCADE_PATH.exists():
            raise FileNotFoundError(f"Face cascade not found: {FACE_CASCADE_PATH}")
        if not EYE_CASCADE_PATH.exists():
            raise FileNotFoundError(f"Eye cascade not found: {EYE_CASCADE_PATH}")

        self.face_cascade = cv2.CascadeClassifier(str(FACE_CASCADE_PATH))
        self.eye_cascade  = cv2.CascadeClassifier(str(EYE_CASCADE_PATH))
        self.image_size   = image_size
        self.scale_factor = scale_factor
        self.min_neighbours = min_neighbours

    def detect(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        """
        Args:
            frame: BGR webcam frame (H×W×3 uint8)

        Returns:
            left_eye  : grayscale crop (100×100) or None
            right_eye : grayscale crop (100×100) or None
            annotated : copy of frame with bounding boxes drawn
        """
        annotated = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbours,
            minSize=(60, 60),   # was 80 — works at greater distances now
        )

        if len(faces) == 0:
            return None, None, annotated

        # Use largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        cv2.rectangle(annotated, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 2)

        face_gray = gray[fy:fy+fh, fx:fx+fw]

        # Detect eyes within face region
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.08,    # was 1.1 — finer scale steps
            minNeighbors=4,      # was 10 — much more lenient, catches more real eyes
            minSize=(15, 15),    # was 20 — catches smaller eye crops too
        )

        if len(eyes) < 2:
            return None, None, annotated

        # Sort eyes left-to-right (by x coordinate within face)
        eyes = sorted(eyes, key=lambda e: e[0])

        left_crop  = self._extract_crop(face_gray, eyes[0])
        right_crop = self._extract_crop(face_gray, eyes[1])

        # Draw eye boxes
        for ex, ey, ew, eh in eyes[:2]:
            cv2.rectangle(annotated,
                           (fx+ex, fy+ey),
                           (fx+ex+ew, fy+ey+eh),
                           (255, 0, 0), 1)

        return left_crop, right_crop, annotated

    def _extract_crop(self, face_gray: np.ndarray, eye_rect: tuple) -> np.ndarray:
        """Crop eye region and resize to model input size."""
        ex, ey, ew, eh = eye_rect
        crop = face_gray[ey:ey+eh, ex:ex+ew]
        return cv2.resize(crop, (self.image_size, self.image_size))


# ── Standalone test ────────────────────────────────────────────────────────────

def _run_preview(camera_id: int = 0) -> None:
    detector = EyeDetector()
    cap = cv2.VideoCapture(camera_id)
    print("Eye detector preview — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        left, right, annotated = detector.detect(frame)
        cv2.imshow("Eye Detector", annotated)

        if left is not None:
            cv2.imshow("Left Eye",  left)
        if right is not None:
            cv2.imshow("Right Eye", right)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    _run_preview(args.camera)
