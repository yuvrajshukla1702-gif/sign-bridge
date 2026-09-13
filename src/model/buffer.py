"""
Module B — Circular frame buffer + prediction debounce.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np


class SequenceBuffer:
    """Sliding window of the last `window_size` feature frames."""

    def __init__(self, window_size: int = 30, feature_dim: int = 147) -> None:
        self.window_size = window_size
        self.feature_dim = feature_dim
        self._buf: Deque[np.ndarray] = deque(maxlen=window_size)

    def push(self, features: np.ndarray) -> None:
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {vec.shape[0]}")
        self._buf.append(vec)

    def ready(self) -> bool:
        return len(self._buf) == self.window_size

    def get_window(self) -> Optional[np.ndarray]:
        """Return (window_size, feature_dim) or None if not full."""
        if not self.ready():
            return None
        return np.stack(list(self._buf), axis=0)

    def clear(self) -> None:
        self._buf.clear()


class PredictionDebouncer:
    """
    Emit a label only when the same class stays above `threshold`
    for at least `min_consecutive` frames (reduces flicker).
    """

    def __init__(
        self,
        threshold: float = 0.85,
        min_consecutive: int = 10,
        cooldown_frames: int = 15,
    ) -> None:
        self.threshold = threshold
        self.min_consecutive = min_consecutive
        self.cooldown_frames = cooldown_frames
        self._current_label: Optional[str] = None
        self._streak: int = 0
        self._cooldown: int = 0
        self._last_emitted: Optional[str] = None

    def update(self, label: str, confidence: float) -> Optional[str]:
        """
        Feed one prediction. Returns label when a stable detection fires, else None.
        """
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        if confidence < self.threshold:
            self._current_label = None
            self._streak = 0
            return None

        if label == self._current_label:
            self._streak += 1
        else:
            self._current_label = label
            self._streak = 1

        if self._streak >= self.min_consecutive:
            # Avoid immediately re-emitting the same word
            if label == self._last_emitted:
                self._streak = 0
                return None
            self._last_emitted = label
            self._streak = 0
            self._cooldown = self.cooldown_frames
            return label
        return None

    def reset(self) -> None:
        self._current_label = None
        self._streak = 0
        self._cooldown = 0
        self._last_emitted = None
