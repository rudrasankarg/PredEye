"""
train_lm.py — Word2Vec embeddings + LSTM language model  ← NOVEL CONTRIBUTION

Pipeline:
  1. Load NLTK Brown corpus (~1M words)
  2. Train Word2Vec (gensim) -> 100-dim embeddings
  3. Build (context window -> next word) training pairs
  4. Train stacked LSTM initialized with Word2Vec weights
  5. Save word2vec.bin + lstm_lm.pt

Usage:
    python src/train_lm.py --output_dir models
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Hyperparameters ────────────────────────────────────────────────────────────

VOCAB_SIZE    = 15_000     # top-K words kept
EMBED_DIM     = 100        # Word2Vec output dims
CONTEXT_LEN   = 5          # sliding context window (# words -> predict next)
LSTM1_UNITS   = 256
LSTM2_UNITS   = 128
DROPOUT_RATE  = 0.3
BATCH_SIZE    = 128
EPOCHS        = 30
LR            = 1e-3
PATIENCE      = 5

SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}


# ── Step 1 — Word2Vec ──────────────────────────────────────────────────────────

def train_word2vec(words: list[str], output_dir: str, vocab_size: int = VOCAB_SIZE,
                   embed_dim: int = EMBED_DIM) -> tuple:
    """
    Train gensim Word2Vec on the word list.
    Returns (w2v_model, word2idx, idx2word, embedding_matrix).
    """
    # pyrefly: ignore [missing-import]
    from gensim.models import Word2Vec
    from collections import Counter

    print(f"\n[1/3] Training Word2Vec on {len(words):,} words…")

    # Build vocabulary: top vocab_size words
    counter = Counter(words)
    common_words = [w for w, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]

    # Train Word2Vec on sentences (Brown corpus is already sentence-split; 
    # here we use a simple chunked approach)
    chunk_size = 500
    sentences = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size)]

    w2v = Word2Vec(
        sentences=sentences,
        vector_size=embed_dim,
        window=5,
        min_count=2,
        workers=4,
        sg=1,          # skip-gram
        epochs=10,
    )

    # Save
    w2v_path = Path(output_dir) / "word2vec.bin"
    w2v.save(str(w2v_path))
    print(f"  Saved Word2Vec -> {w2v_path}")

    # Build word ↔ index maps
    word2idx = {**SPECIAL_TOKENS}
    idx2word = {v: k for k, v in SPECIAL_TOKENS.items()}
    for i, word in enumerate(common_words, start=len(SPECIAL_TOKENS)):
        word2idx[word] = i
        idx2word[i] = word

    # Build embedding matrix (vocab_size × embed_dim)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    n_found = 0
    for word, idx in word2idx.items():
        if word in w2v.wv:
            embedding_matrix[idx] = w2v.wv[word]
            n_found += 1

    print(f"  Vocab size: {len(word2idx):,} | "
          f"Words with embeddings: {n_found:,}/{len(word2idx):,}")

    return w2v, word2idx, idx2word, embedding_matrix


# ── Step 2 — Build training pairs ─────────────────────────────────────────────

def build_sequences(
    words: list[str],
    word2idx: dict[str, int],
    context_len: int = CONTEXT_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert word list -> (X, y) integer arrays.
    X shape: (N, context_len), y shape: (N,)
    """
    print(f"\n[2/3] Building {context_len}-token context windows…")
    UNK = SPECIAL_TOKENS["<UNK>"]

    indices = [word2idx.get(w, UNK) for w in words]

    X, y = [], []
    for i in range(context_len, len(indices)):
        X.append(indices[i - context_len: i])
        y.append(indices[i])

    X = np.array(X, dtype=np.int64)
    y = np.array(y, dtype=np.int64)
    print(f"  Sequences: {len(X):,}  shape={X.shape}")
    return X, y


# ── Step 3 — LSTM Model ────────────────────────────────────────────────────────

class LSTMLanguageModel(nn.Module):
    """
    Stacked LSTM language model with pre-trained Word2Vec embeddings.

    Input:  (batch, seq_len) integer token IDs
    Output: (batch, vocab_size) logits over next word
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        lstm1_units: int,
        lstm2_units: int,
        dropout_rate: float,
        embedding_matrix: np.ndarray,
    ):
        super().__init__()

        # Embedding layer — initialised from Word2Vec weights
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        self.lstm1   = nn.LSTM(embed_dim, lstm1_units, batch_first=True)
        self.drop1   = nn.Dropout(dropout_rate)
        self.lstm2   = nn.LSTM(lstm1_units, lstm2_units, batch_first=True)
        self.drop2   = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(lstm2_units, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x)                    # (batch, seq_len, embed_dim)
        out, _ = self.lstm1(emb)                   # (batch, seq_len, lstm1)
        out = self.drop1(out)
        out, _ = self.lstm2(out)                   # (batch, seq_len, lstm2)
        out = self.drop2(out)
        out = out[:, -1, :]                        # last timestep: (batch, lstm2)
        logits = self.fc(out)                      # (batch, vocab_size)
        return logits


# ── Step 4 — Training loop ────────────────────────────────────────────────────

def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    embedding_matrix: np.ndarray,
    vocab_size: int,
    output_dir: str,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    patience: int = PATIENCE,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3/3] Training LSTM on {device}…")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Split 90/10
    n_val = max(1, int(len(X) * 0.1))
    idx = np.random.permutation(len(X))
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    X_t = torch.from_numpy(X[train_idx])
    y_t = torch.from_numpy(y[train_idx])
    X_v = torch.from_numpy(X[val_idx])
    y_v = torch.from_numpy(y[val_idx])

    train_dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
    val_dl   = DataLoader(TensorDataset(X_v, y_v), batch_size=batch_size,
                          shuffle=False, num_workers=2, pin_memory=True)

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        lstm1_units=LSTM1_UNITS,
        lstm2_units=LSTM2_UNITS,
        dropout_rate=DROPOUT_RATE,
        embedding_matrix=embedding_matrix,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LSTM parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore PAD
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in tqdm(train_dl, desc=f"Epoch {epoch:2d} train", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss = total_loss / n

        # Validate
        model.eval()
        val_loss, n_val_total = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * len(yb)
                n_val_total += len(yb)
        val_loss /= n_val_total

        history.append((train_loss, val_loss))
        ppl_train = np.exp(train_loss)
        ppl_val   = np.exp(val_loss)
        print(f"  Epoch {epoch:2d}  loss={train_loss:.4f} ppl={ppl_train:.1f}  "
              f"val_loss={val_loss:.4f} val_ppl={ppl_val:.1f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = Path(output_dir) / "lstm_lm.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "vocab_size": vocab_size,
                "embed_dim": EMBED_DIM,
                "lstm1_units": LSTM1_UNITS,
                "lstm2_units": LSTM2_UNITS,
                "dropout_rate": DROPOUT_RATE,
                "context_len": CONTEXT_LEN,
            }, ckpt)
            print(f"  [OK] Checkpoint saved (val_ppl={ppl_val:.1f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation perplexity: {np.exp(best_val_loss):.2f}")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    # Allow running from repo root
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_text_corpus

    parser = argparse.ArgumentParser(description="Train Word2Vec + LSTM language model")
    parser.add_argument("--output_dir",  default="models")
    parser.add_argument("--vocab_size",  type=int, default=VOCAB_SIZE)
    parser.add_argument("--embed_dim",   type=int, default=EMBED_DIM)
    parser.add_argument("--context_len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--epochs",      type=int, default=EPOCHS)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--max_words",   type=int, default=0, help="0=all")
    args = parser.parse_args()

    words = load_text_corpus(max_words=args.max_words)

    _, word2idx, _, embedding_matrix = train_word2vec(
        words, args.output_dir, args.vocab_size, args.embed_dim
    )

    # Save word2idx for inference
    import json
    with open(Path(args.output_dir) / "word2idx.json", "w") as f:
        json.dump(word2idx, f)
    print("Saved word2idx.json")

    X, y = build_sequences(words, word2idx, args.context_len)

    train_lstm(X, y, embedding_matrix, args.vocab_size, args.output_dir,
               args.epochs, args.batch_size, args.lr)
