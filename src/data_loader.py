"""
data_loader.py — UnityEyes dataset parser + NLTK corpus loader

UnityEyes output format: flat folder of <N>.jpg + <N>.json pairs.
Each JSON contains a "look_vec" field (3D unit gaze vector [x, y, z]).
We map that vector to one of 9 gaze directions (1-indexed, row-major):
    1=NW  2=N  3=NE
    4=W   5=C  6=E
    7=SW  8=S  9=SE

Usage:
    python src/data_loader.py --input_dir data/synthetic_raw \
                               --output_dir data/synthetic \
                               --n_images 20000
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# ── Direction mapping ──────────────────────────────────────────────────────────

# 9 named directions (label 1–9, stored in subfolders named 1–9)
DIR_NAMES = {
    1: "NW", 2: "N",  3: "NE",
    4: "W",  5: "C",  6: "E",
    7: "SW", 8: "S",  9: "SE",
}

# Gaze-vector thresholds (horizontal x, vertical y component)
# x: left = negative, right = positive
# y: up = positive, down = negative
# z: forward (into screen) = positive — we mostly use x and y
X_THRESHOLD = 0.15   # horizontal dead-zone around centre
Y_THRESHOLD = 0.15   # vertical dead-zone around centre


def gaze_vec_to_label(look_vec: list[float]) -> int:
    """Map a 3-element gaze unit vector to a 1–9 direction label."""
    x, y, _ = look_vec  # z is depth, not used for direction
    # Horizontal bucket: left / centre / right
    if x < -X_THRESHOLD:
        h = 0   # left column
    elif x > X_THRESHOLD:
        h = 2   # right column
    else:
        h = 1   # centre column

    # Vertical bucket: up / centre / down
    if y > Y_THRESHOLD:
        v = 0   # top row
    elif y < -Y_THRESHOLD:
        v = 2   # bottom row
    else:
        v = 1   # centre row

    # Label = row * 3 + col + 1  (1-indexed)
    return v * 3 + h + 1


def parse_unityeyes_json(json_path: Path) -> Optional[list[float]]:
    """
    Return [x, y, z] gaze vector from a UnityEyes JSON file, or None on error.

    UnityEyes stores look_vec inside eye_details as a formatted string:
        "(-0.3232, -0.1136, -0.9395, 0.0000)"
    We parse the first 3 components (x=horizontal, y=vertical, z=depth).
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Primary location: eye_details.look_vec (string format)
        raw = None
        eye_details = data.get("eye_details")
        if eye_details:
            raw = eye_details.get("look_vec")

        # Fallback: top-level look_vec (list format)
        if raw is None:
            raw = data.get("look_vec")

        if raw is None:
            return None

        # Parse string format "(x, y, z, w)" -> [x, y, z]
        if isinstance(raw, str):
            cleaned = raw.strip().strip("()")
            parts = [float(v.strip()) for v in cleaned.split(",")]
            return parts[:3]  # x, y, z only

        # Already a list
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw[:3]]

        return None

    except (json.JSONDecodeError, FileNotFoundError, ValueError):
        return None


def build_dataset(
    input_dir: str,
    output_dir: str,
    n_images: int = 20_000,
    image_size: int = 100,
) -> dict[int, int]:
    """
    Parse UnityEyes output and organize into class folders.

    Args:
        input_dir:  folder containing *.jpg + *.json pairs from UnityEyes
        output_dir: destination — class folders 1–9 will be created here
        n_images:   maximum images to process (0 = all)
        image_size: output image size in pixels (square, grayscale)

    Returns:
        dict mapping label -> count of images processed
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create class folders
    for label in range(1, 10):
        (output_path / str(label)).mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(input_path.glob("*.json"))
    if n_images > 0:
        json_files = json_files[:n_images]

    counts = {i: 0 for i in range(1, 10)}
    skipped = 0

    print(f"Processing {len(json_files)} UnityEyes samples…")
    for json_file in tqdm(json_files):
        img_file = json_file.with_suffix(".jpg")
        if not img_file.exists():
            img_file = json_file.with_suffix(".png")
        if not img_file.exists():
            skipped += 1
            continue

        look_vec = parse_unityeyes_json(json_file)
        if look_vec is None:
            skipped += 1
            continue

        label = gaze_vec_to_label(look_vec)

        # Load, convert to grayscale, resize
        try:
            img = Image.open(img_file).convert("L").resize(
                (image_size, image_size), Image.LANCZOS
            )
        except Exception:
            skipped += 1
            continue

        dest = output_path / str(label) / img_file.name
        img.save(dest)
        counts[label] += 1

    print(f"\nDataset built -> {output_path}")
    print(f"Skipped: {skipped}")
    for label, count in counts.items():
        print(f"  Class {label} ({DIR_NAMES[label]}): {count} images")

    return counts


# ── Calibration data loader ────────────────────────────────────────────────────

def load_calibration_data(
    calib_dir: str,
    image_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load per-user calibration images from a directory structured as:
        calib_dir/1/  calib_dir/2/  …  calib_dir/9/
    Returns (X, y) numpy arrays ready for fine-tuning.
    """
    calib_path = Path(calib_dir)
    images, labels = [], []

    for label in range(1, 10):
        class_dir = calib_path / str(label)
        if not class_dir.exists():
            continue
        for img_file in class_dir.glob("*.jpg"):
            try:
                img = Image.open(img_file).convert("L").resize(
                    (image_size, image_size), Image.LANCZOS
                )
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(label - 1)  # 0-indexed for CrossEntropy
            except Exception:
                continue

    if not images:
        raise ValueError(f"No calibration images found in {calib_dir}")

    X = np.stack(images)[:, np.newaxis, :, :]  # (N, 1, H, W) — PyTorch format
    y = np.array(labels, dtype=np.int64)
    return X, y


# ── Text corpus loader ─────────────────────────────────────────────────────────

def load_text_corpus(max_words: int = 0) -> list[str]:
    """
    Load the NLTK Brown corpus and return a flat list of lowercase words.
    Downloads corpus automatically on first run.
    """
    # pyrefly: ignore [missing-import]
    import nltk
    nltk.download("brown", quiet=True)
    # pyrefly: ignore [missing-import]
    from nltk.corpus import brown

    words = [w.lower() for w in brown.words() if w.isalpha()]
    if max_words > 0:
        words = words[:max_words]
    print(f"Corpus loaded: {len(words):,} words")
    return words


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build UnityEyes dataset")
    parser.add_argument("--input_dir",  required=True, help="UnityEyes output folder")
    parser.add_argument("--output_dir", default="data/synthetic", help="Destination class folders")
    parser.add_argument("--n_images",   type=int, default=20_000, help="Max images (0=all)")
    parser.add_argument("--image_size", type=int, default=100, help="Output size in px")
    args = parser.parse_args()

    build_dataset(args.input_dir, args.output_dir, args.n_images, args.image_size)
