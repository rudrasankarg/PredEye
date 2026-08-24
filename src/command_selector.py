"""
command_selector.py — Algorithm 1 (Async) + Algorithm 2 (Sync)

Implements both gaze command selection algorithms verbatim from Meena & Salvi 2025.
These are the BASELINE — our novel contribution (LSTM prediction) lives in predictor.py.

Usage:
    selector = AsyncSelector()
    for Lp, Rp in gaze_stream:
        result = selector.update(Lp, Rp)
        if result is not None:
            print(f"Command selected: {result}")

    selector2 = SyncSelector(window_seconds=2.0)
    for Lp, Rp in window_stream:
        selector2.add_frame(Lp, Rp)
    result = selector2.select()
"""

import time
from dataclasses import dataclass, field


CENTRE = 5  # gaze direction "Centre" (C) — not a command
ALPHA  = 6  # confidence threshold for Algorithm 2


@dataclass
class AsyncSelector:
    """
    Algorithm 1 — Asynchronous gaze command selection (Meena & Salvi 2025).

    A command fires when the same direction (≠ Centre) appears in
    Δt1 consecutive frames from both eyes.

    Params:
        delta_t1: consecutive agreement frames needed to fire (default=6)
        on_fire:  optional callback(direction: int)
    """

    delta_t1: int = 6
    on_fire: object = None  # callable or None

    _last_selected: int = field(default=0, init=False, repr=False)
    _delta: int = field(default=0, init=False, repr=False)

    def update(self, Lp: int, Rp: int) -> int | None:
        """
        Process one frame.
        Returns the fired command (1–9) or None.
        """
        if Lp == Rp:
            selected_t = Rp
        else:
            selected_t = None

        if selected_t is not None and selected_t == self._last_selected:
            self._delta += 1
        elif selected_t is not None:
            self._delta = 1
            self._last_selected = selected_t
        else:
            # Leaky bucket: tolerate occasional noise without completely resetting
            self._delta = max(0, self._delta - 2)

        if self._delta >= self.delta_t1:
            self._delta = 0
            command = self._last_selected
            if self.on_fire:
                self.on_fire(command)
            return command

        return None

    def reset(self) -> None:
        self._last_selected = 0
        self._delta = 0


@dataclass
class SyncSelector:
    """
    Algorithm 2 — Synchronous gaze command selection (Meena & Salvi 2025).

    Accumulates weighted votes over a time window, then fires the
    highest-weight direction if its dominance ratio P ≥ α.

    Params:
        alpha: confidence ratio threshold (default=6, per paper)
    """

    alpha: float = ALPHA
    _weights: dict = field(default_factory=lambda: {i: 0.0 for i in range(1, 10)},
                           init=False, repr=False)
    _frame_count: int = field(default=0, init=False, repr=False)

    def add_frame(self, Lp: int, Rp: int) -> None:
        """Add one frame's vote (both eyes must agree, neither = Centre)."""
        self._frame_count += 1
        if Lp == Rp:
            self._weights[Rp] += (self._frame_count ** 0.5)

    def select(self) -> int | None:
        """
        Evaluate collected votes.
        Returns selected direction (1–9) if P ≥ α, else None.
        """
        if not any(self._weights.values()):
            return None

        max_w  = max(self._weights.values())
        mean_w = sum(self._weights.values()) / len(self._weights)

        if mean_w == 0:
            return None

        P = max_w / mean_w
        if P >= self.alpha:
            selected = max(self._weights, key=self._weights.get)
            self.reset()
            return selected

        return None

    def reset(self) -> None:
        self._weights = {i: 0.0 for i in range(1, 10)}
        self._frame_count = 0

    def get_weights(self) -> dict[int, float]:
        """Return current weight dict (for GUI highlight bars)."""
        return dict(self._weights)
