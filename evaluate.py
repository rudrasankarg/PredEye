"""
evaluate.py - Measure ITR with vs. without LSTM prediction

Reproduces the evaluation table from Meena & Salvi 2025 (baseline)
and adds the prediction-enabled comparison (our contribution).

Methodology:
  1. Choose a test sentence (e.g., "painting which landform")
  2. Count gaze-selections required to type it WITHOUT prediction
  3. Count gaze-selections required to type it WITH prediction active
  4. Compute ITR = B × (log2(N) + P×log2(P) + (1-P)×log2((1-P)/(N-1))) bits/selection
     where N=9 (positions), P=accuracy

Usage:
    python evaluate.py --sentence "painting which landform" --mode simulate
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.predictor import Predictor
from src.keyboard_gui import LETTER_GROUPS

# ── ITR calculation ────────────────────────────────────────────────────────────

def itr_bits_per_selection(N: int, P: float) -> float:
    """
    Information Transfer Rate per selection (Wolpaw et al. 2000).
    N: number of choices (9 for this keyboard)
    P: selection accuracy (0–1)
    """
    if P <= 0 or P > 1 or N <= 1:
        return 0.0
    if P == 1.0:
        return math.log2(N)
    return (math.log2(N)
            + P * math.log2(P)
            + (1 - P) * math.log2((1 - P) / (N - 1)))


def itr_bits_per_minute(bits_per_sel: float, selections_per_minute: float) -> float:
    return bits_per_sel * selections_per_minute


# ── Selection count (simulated) ───────────────────────────────────────────────

def count_selections_no_prediction(sentence: str) -> tuple[int, list[str]]:
    """
    Count how many gaze-selections are needed to type `sentence`
    WITHOUT prediction (letter-by-letter).

    Each character requires:
      - 1 selection to pick the letter group (Level 1)
      - 1 selection to pick the letter     (Level 2)
      Total: 2 selections per letter.
    SPACE and BACK require 1 extra selection each.
    """
    steps = []
    total = 0
    for char in sentence.upper():
        if char == " ":
            # Find which group contains SPACE (handled via GO_BACK + space logic)
            # In the baseline keyboard, space is typed after finishing a word
            # by returning to Level 1 - already counted as part of the last letter
            continue
        if not char.isalpha():
            continue
        # Find which group this letter belongs to
        for d, group in LETTER_GROUPS.items():
            if char in group and d not in (5, 9):  # skip GO_BACK and DELETE
                steps.append(f"[sel group {d} -> {group}] -> [sel letter {char}]")
                total += 2
                break

    return total, steps


def count_selections_with_prediction(
    sentence: str, predictor: Predictor
) -> tuple[int, list[str]]:
    """
    Count selections with LSTM prediction enabled.

    Strategy: at the start of each word, check if the predictor
    offers the word as a completion. If yes, use the prediction
    (1 selection instead of 2×len(word) selections).
    If no, fall back to letter-by-letter.
    """
    words = sentence.lower().split()
    total = 0
    steps = []
    typed_so_far = ""

    for word in words:
        completions = predictor.get_completions(typed_so_far, top_k=3)
        if word in completions:
            slot = completions.index(word)
            steps.append(f"[PRED slot {slot+1}: '{word}'] - 1 selection")
            total += 1
        else:
            # Type letter by letter
            for char in word.upper():
                for d, group in LETTER_GROUPS.items():
                    if char in group and d not in (5, 9):
                        steps.append(f"[group {d}:{group}] -> [{char}]")
                        total += 2
                        break
            # After each letter, predictions update - check if word becomes predictable
            # (conservative simulation: check only at word boundaries)

        typed_so_far += word + " "

    return total, steps


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    sentence: str,
    predictor: Predictor,
    accuracy: float = 0.95,
    selections_per_minute: float = 30.0,
) -> dict:
    """
    Run both baseline and prediction evaluations and return a results dict.

    NOTE: selections_per_minute and accuracy are experimentally measured values.
    Replace with your real measured numbers before finalising the report.
    """
    n_base, steps_base = count_selections_no_prediction(sentence)
    n_pred, steps_pred = count_selections_with_prediction(sentence, predictor)

    bits_per_sel = itr_bits_per_selection(9, accuracy)
    itr_base     = itr_bits_per_minute(bits_per_sel, selections_per_minute)
    itr_pred_spm = selections_per_minute  # assume same speed; real test would measure
    itr_pred     = itr_bits_per_minute(bits_per_sel, itr_pred_spm)

    reduction_pct = (1 - n_pred / n_base) * 100 if n_base > 0 else 0

    results = {
        "sentence": sentence,
        "accuracy_assumed": accuracy,
        "spm_assumed": selections_per_minute,
        "selections_without_prediction": n_base,
        "selections_with_prediction": n_pred,
        "reduction_percent": round(reduction_pct, 1),
        "itr_baseline_bpm": round(itr_base, 2),
        "itr_predicted_bpm": round(itr_pred, 2),
        "steps_baseline": steps_base,
        "steps_with_prediction": steps_pred,
    }
    return results


def print_results(r: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  Sentence: \"{r['sentence']}\"")
    print("=" * 60)
    print(f"  {'Metric':<38} {'Baseline':>10}  {'+ LSTM':>10}")
    print("-" * 60)
    print(f"  {'Gaze selections':<38} {r['selections_without_prediction']:>10}  "
          f"{r['selections_with_prediction']:>10}")
    print(f"  {'Selection reduction':<38} {'-':>10}  "
          f"{r['reduction_percent']:>9.1f}%")
    itr_label = f"ITR (bits/min) [assumed acc={r['accuracy_assumed']:.0%}]"
    print(f"  {itr_label:<38} "
          f"{r['itr_baseline_bpm']:>10.2f}  {r['itr_predicted_bpm']:>10.2f}")
    print("=" * 60)
    print()
    print("[!]  REMINDER: Replace 'assumed' values with real measurements from a live session.")
    print("   - Measure actual selections/min by timing a typed test sentence.")
    print("   - Measure accuracy by logging correct vs. total commands during typing.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ITR with/without prediction")
    parser.add_argument("--sentence",  default="painting which landform",
                        help="Test sentence to evaluate")
    parser.add_argument("--model_lm",  default="models/lstm_lm.pt")
    parser.add_argument("--word2idx",  default="models/word2idx.json")
    parser.add_argument("--accuracy",  type=float, default=0.95,
                        help="Assumed selection accuracy (replace with real value)")
    parser.add_argument("--spm",       type=float, default=30.0,
                        help="Assumed selections/min (replace with real value)")
    parser.add_argument("--save_json", default="",
                        help="If set, save results to this JSON file")
    args = parser.parse_args()

    pred = Predictor(model_path=args.model_lm, word2idx_path=args.word2idx)
    results = run_evaluation(args.sentence, pred, args.accuracy, args.spm)
    print_results(results)

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.save_json}")
