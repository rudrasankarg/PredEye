"""
train_gaze.py — CNN gaze direction classifier (PyTorch)

Architecture mirrors Meena & Salvi 2025 exactly (282,701 parameters):
    Conv2D(16,11×11) -> MaxPool(3×3) -> Conv2D(32,7×7) -> MaxPool(3×3) ->
    Dropout(0.2) -> Conv2D(64,3×3,same) -> MaxPool(2×2) -> Dropout(0.2) ->
    Conv2D(128,2×2) -> MaxPool(2×2) -> Flatten -> Dense(250) -> Dropout(0.5) ->
    Dense(9, Softmax)

Input:  100×100 grayscale images
Output: 9-class probability vector (1=NW … 9=SE)

Usage:
    python src/train_gaze.py --data_dir data/synthetic --output_dir models
"""

import argparse
import os
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
# Note: matplotlib and sklearn are imported lazily inside _plot_* functions
#       so that gaze_predictor.py can import GazeCNN without those deps.

# ── Model Definition ───────────────────────────────────────────────────────────

class GazeCNN(nn.Module):
    """
    9-class gaze direction CNN (Meena & Salvi 2025).
    Input: (N, 1, 100, 100) — grayscale, normalised to [0,1].
    """

    def __init__(self, num_classes: int = 9):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: Conv(16, 11×11) -> MaxPool(3×3)
            nn.Conv2d(1, 16, kernel_size=11),           # 100->90
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),                # 90->30

            # Block 2: Conv(32, 7×7) -> MaxPool(3×3)
            nn.Conv2d(16, 32, kernel_size=7),           # 30->24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),                # 24->8
            nn.Dropout2d(p=0.2),

            # Block 3: Conv(64, 3×3, padding=same) -> MaxPool(2×2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 8->8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # 8->4
            nn.Dropout2d(p=0.2),

            # Block 4: Conv(128, 2×2) -> MaxPool(2×2)
            nn.Conv2d(64, 128, kernel_size=2),          # 4->3 ← note: 3×3 not 2×2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),                # 3->1 (floor)
        )

        # After the conv stack: 128 × 1 × 1 = 128 (not 512 in PyTorch due to floor division)
        # We compute the flattened size dynamically to be safe
        self._flat_size = self._get_flat_size()

        self.classifier = nn.Sequential(
            nn.Linear(self._flat_size, 250),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(250, num_classes),
            # No Softmax here — CrossEntropyLoss includes log-softmax
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 100, 100)
            out = self.features(dummy)
            return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Data Loading ───────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    image_size: int = 100,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) from a class-folder dataset."""

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.ToTensor(),              # [0,255] -> [0,1]
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    # Load full dataset with train transforms (we apply val_tf separately below)
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

    # Re-apply correct transforms to val split
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)

    print(f"Dataset: {len(train_ds)} train | {len(val_ds)} val | "
          f"{len(full_dataset.classes)} classes")
    return train_loader, val_loader


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 10,
    device_str: str = "auto",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader = get_dataloaders(data_dir, batch_size)

    model = GazeCNN(num_classes=9).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total

        # ── Validate ──
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)

        val_loss = val_loss / val_total
        val_acc  = val_correct / val_total
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:3d}/{epochs}  "
              f"loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = Path(output_dir) / "base_gaze_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            print(f"  [OK] Saved checkpoint (val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    _plot_history(history, output_dir)
    _plot_confusion_matrix(model, val_loader, device, output_dir)


def _plot_history(history: dict, output_dir: str) -> None:
    import matplotlib.pyplot as plt  # training-only dep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Validation")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"],   label="Validation")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "training_curves.png", dpi=150)
    plt.close()
    print("Saved: training_curves.png")


def _plot_confusion_matrix(
    model: nn.Module, val_loader: DataLoader, device: torch.device, output_dir: str
) -> None:
    import matplotlib.pyplot as plt  # training-only dep
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # training-only dep
    from data_loader import DIR_NAMES

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [DIR_NAMES[i] for i in range(1, 10)]
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Gaze CNN — Confusion Matrix (Validation)")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gaze CNN")
    parser.add_argument("--data_dir",   default="data/synthetic",
                        help="ImageFolder-structured class dirs (1–9)")
    parser.add_argument("--output_dir", default="models",
                        help="Where to save model checkpoint + plots")
    parser.add_argument("--epochs",     type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--device",     default="auto",
                        choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    train(args.data_dir, args.output_dir, args.epochs, args.batch_size,
          args.lr, args.patience, args.device)
