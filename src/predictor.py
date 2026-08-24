"""
predictor.py — LSTM-based word completion API  ← NOVEL CONTRIBUTION

Given partially-typed text, returns the top-K most likely next-word completions
using the trained LSTM language model + Word2Vec embeddings.

Prefix filtering: if the user has typed a partial word (e.g., "hel"),
only completions starting with "hel" are returned.

Fallback: if the model is not loaded, returns frequency-based completions
from a hardcoded top-English-words list.

Usage:
    predictor = Predictor("models/lstm_lm.pt", "models/word2idx.json")
    completions = predictor.get_completions("the quick brown", top_k=3)
    # -> ["fox", "bear", "dog"]
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

# ── Fallback word list (used when model not loaded) ───────────────────────────

_COMMON_WORDS = [
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
    "but", "his", "by", "from", "they", "we", "say", "her", "she", "or",
    "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could",
    "them", "see", "other", "than", "then", "now", "look", "only", "come",
    "its", "over", "think", "also", "back", "after", "use", "two", "how",
    "our", "work", "first", "well", "way", "even", "new", "want", "because",
    "any", "these", "give", "day", "most", "us",
]


class Predictor:
    """
    LSTM word-completion predictor.

    If the .pt model and word2idx.json exist, uses the trained LSTM.
    Otherwise falls back to frequency-list completions (useful during development).
    """

    def __init__(
        self,
        model_path: str = "models/lstm_lm.pt",
        word2idx_path: str = "models/word2idx.json",
        context_len: int = 5,
    ):
        self.context_len = context_len
        self.model = None
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try loading model
        ckpt_path = Path(model_path)
        w2i_path  = Path(word2idx_path)

        if ckpt_path.exists() and w2i_path.exists():
            self._load_model(ckpt_path, w2i_path)
        else:
            print(f"[Predictor] Model not found — using fallback word list.")
            print(f"  (Train with: python src/train_lm.py)")

    def _load_model(self, ckpt_path: Path, w2i_path: Path) -> None:
        from train_lm import LSTMLanguageModel, EMBED_DIM, LSTM1_UNITS, LSTM2_UNITS, DROPOUT_RATE

        with open(w2i_path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        vocab_size   = ckpt.get("vocab_size", len(self.word2idx))
        embed_dim    = ckpt.get("embed_dim", EMBED_DIM)
        lstm1_units  = ckpt.get("lstm1_units", LSTM1_UNITS)
        lstm2_units  = ckpt.get("lstm2_units", LSTM2_UNITS)
        dropout_rate = ckpt.get("dropout_rate", DROPOUT_RATE)
        self.context_len = ckpt.get("context_len", self.context_len)

        # Build dummy embedding matrix for constructor (weights loaded from state_dict)
        dummy_emb = np.zeros((vocab_size, embed_dim), dtype=np.float32)
        self.model = LSTMLanguageModel(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm1_units=lstm1_units,
            lstm2_units=lstm2_units,
            dropout_rate=dropout_rate,
            embedding_matrix=dummy_emb,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(f"[Predictor] Loaded LSTM from {ckpt_path}")

    def get_completions(
        self,
        typed_text: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        Return top_k word completions for the given typed text.

        - If typed_text ends with a complete word (trailing space): predict next word.
        - If typed_text ends mid-word (no trailing space): predict completions of that prefix.

        Args:
            typed_text: everything the user has typed so far
            top_k:      number of suggestions to return

        Returns:
            list of word strings (may be shorter than top_k if prefix is unusual)
        """
        text = typed_text.strip()
        words = text.lower().split() if text else []

        # Determine prefix filter
        has_trailing_space = typed_text.endswith(" ")
        if has_trailing_space or not words:
            prefix = ""
        else:
            prefix = words[-1]
            words = words[:-1]  # context excludes partial word

        if self.model is not None:
            return self._lstm_completions(words, prefix, top_k)
        else:
            return self._fallback_completions(prefix, top_k)

    def _lstm_completions(
        self,
        context_words: list[str],
        prefix: str,
        top_k: int,
    ) -> list[str]:
        """Run LSTM inference and filter by prefix."""
        UNK = self.word2idx.get("<UNK>", 1)
        PAD = self.word2idx.get("<PAD>", 0)

        # Build context tensor of length context_len (left-pad with PAD)
        context_ids = [self.word2idx.get(w, UNK) for w in context_words]
        if len(context_ids) < self.context_len:
            context_ids = [PAD] * (self.context_len - len(context_ids)) + context_ids
        else:
            context_ids = context_ids[-self.context_len:]

        x = torch.tensor([context_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x)                        # (1, vocab_size)
            probs  = F.softmax(logits, dim=-1).squeeze()  # (vocab_size,)

        # Filter by prefix
        candidates = []
        for idx, prob in enumerate(probs.cpu().numpy()):
            word = self.idx2word.get(idx, "")
            if not word or word.startswith("<"):
                continue
            if prefix and not word.startswith(prefix):
                continue
            candidates.append((word, float(prob)))

        # Sort by probability descending
        candidates.sort(key=lambda t: t[1], reverse=True)
        return [w for w, _ in candidates[:top_k]]

    def _fallback_completions(self, prefix: str, top_k: int) -> list[str]:
        """Simple frequency-based fallback (no model required)."""
        if not prefix:
            return _COMMON_WORDS[:top_k]
        filtered = [w for w in _COMMON_WORDS if w.startswith(prefix)]
        return filtered[:top_k] if filtered else _COMMON_WORDS[:top_k]


# ── Standalone smoke test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = Predictor()
    test_cases = [
        "the quick brown ",
        "I want to ",
        "hel",
        "wh",
        "",
    ]
    for text in test_cases:
        comps = predictor.get_completions(text, top_k=3)
        print(f"  '{text}' -> {comps}")
