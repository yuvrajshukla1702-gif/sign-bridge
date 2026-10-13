"""
Hand feature normalization shared by training, the backend, and (mirrored in)
web/js/landmarks.js. Keep all three in sync.

Raw MediaPipe x/y are fractions of image width/height, so on a 4:3 webcam the
hand is horizontally squashed relative to a square training image. Multiplying
x by the aspect ratio (w/h) first makes the shape resolution-independent.
"""

from __future__ import annotations

import numpy as np

FEATURE_VERSION = "hand63_aspect_v1"
MODEL_FEATURE_VERSION = "hand63_aspect_v1+shape25"

# Finger chains wrist→tip; 3 bend angles per finger. Tips for pairwise spread.
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
TIPS = (4, 8, 12, 16, 20)


def normalize_raw_hand(pts: np.ndarray, aspect: float = 1.0) -> np.ndarray:
    """pts (21,3) raw MediaPipe coords -> (63,) wrist-centred, max-2D-radius scaled."""
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3).copy()
    p[:, 0] *= aspect
    p[:, 2] *= aspect
    p -= p[0]
    scale = float(np.max(np.linalg.norm(p[:, :2], axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return (p / scale).reshape(-1).astype(np.float32)


def model_features(hand63: np.ndarray) -> np.ndarray:
    """(N,63) or (63,) normalized hands -> (N,88): coords + 15 bend cosines + 10 tip distances.

    Bend angles and fingertip spreads barely change with camera viewpoint, which
    is what the raw coordinates alone generalize worst on.
    """
    X = np.asarray(hand63, dtype=np.float32)
    single = X.ndim == 1
    P = X.reshape(-1, 21, 3)
    feats = []
    for ch in CHAINS:
        for j in range(1, 4):
            u = P[:, ch[j]] - P[:, ch[j - 1]]
            v = P[:, ch[j + 1]] - P[:, ch[j]]
            feats.append((u * v).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-6))
    for i in range(5):
        for j in range(i + 1, 5):
            feats.append(np.linalg.norm(P[:, TIPS[i]] - P[:, TIPS[j]], axis=1))
    out = np.concatenate([P.reshape(len(P), -1), np.stack(feats, 1)], 1).astype(np.float32)
    return out[0] if single else out
