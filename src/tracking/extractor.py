"""
Landmark extraction helpers for dual-mode Sign Bridge.

Hand-only helpers avoid importing MediaPipe until a camera Extractor is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from src.tracking.constants import FEATURE_DIM, HAND_FEATURE_DIM, HAND_LANDMARKS, COORDS

if TYPE_CHECKING:
    from src.tracking.landmarks import LandmarkExtractor


def normalize_hand_landmarks(pts: np.ndarray) -> np.ndarray:
    """Wrist-relative + scale-normalized flat (63,) vector."""
    if pts is None or getattr(pts, "size", 0) == 0:
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    wrist = pts[0].copy()
    centered = pts - wrist
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    centered /= scale
    return centered.flatten().astype(np.float32)


def hand_from_holistic_array(hand_flat_63: np.ndarray) -> np.ndarray:
    """Ensure a (63,) float32 vector."""
    v = np.asarray(hand_flat_63, dtype=np.float32).reshape(-1)
    if v.shape[0] < HAND_FEATURE_DIM:
        out = np.zeros(HAND_FEATURE_DIM, dtype=np.float32)
        out[: v.shape[0]] = v
        return out
    return v[:HAND_FEATURE_DIM]


def Extractor(*args, **kwargs):
    """Lazy factory — MediaPipe loads only when camera extraction is needed."""
    from src.tracking.landmarks import LandmarkExtractor

    return _Extractor(*args, **kwargs)


class _Extractor:
    def __init__(self, *args, **kwargs) -> None:
        from src.tracking.landmarks import LandmarkExtractor

        self._inner = LandmarkExtractor(*args, **kwargs)

    def process_hands(self, frame_bgr):
        features, results = self._inner.process(frame_bgr)
        left = None
        right = None
        if results.left_hand_landmarks:
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
                dtype=np.float32,
            )
            left = normalize_hand_landmarks(pts)
        if results.right_hand_landmarks:
            pts = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
                dtype=np.float32,
            )
            right = normalize_hand_landmarks(pts)
        return features, left, right, results

    def close(self) -> None:
        self._inner.close()


__all__ = [
    "Extractor",
    "FEATURE_DIM",
    "HAND_FEATURE_DIM",
    "normalize_hand_landmarks",
    "hand_from_holistic_array",
]
