"""Shared hand-pose similarity for Learn Mode + static classification."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

TIP_IDX = [4, 8, 12, 16, 20]
JOINT_WEIGHTS = np.ones(21, dtype=np.float32)
for _i in TIP_IDX:
    JOINT_WEIGHTS[_i] = 2.4
for _i in (3, 6, 7, 10, 11, 14, 15, 18, 19):
    JOINT_WEIGHTS[_i] = 1.35
JOINT_WEIGHTS /= JOINT_WEIGHTS.mean()


def as_pts(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size < 63:
        pad = np.zeros(63, dtype=np.float32)
        pad[: v.size] = v
        v = pad
    return v[:63].reshape(21, 3)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)[:63]
    bb = np.asarray(b, dtype=np.float32).reshape(-1)[:63]
    na = float(np.linalg.norm(aa) + 1e-8)
    nb = float(np.linalg.norm(bb) + 1e-8)
    return float(np.dot(aa, bb) / (na * nb))


def weighted_mean_distance(live: np.ndarray, ref: np.ndarray) -> Tuple[float, np.ndarray]:
    live_pts = as_pts(live)
    ref_pts = as_pts(ref)
    dists = np.linalg.norm(live_pts - ref_pts, axis=1).astype(np.float32)
    mean_d = float(np.average(dists, weights=JOINT_WEIGHTS))
    return mean_d, dists


def pair_score(live: np.ndarray, ref: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """
    Absolute similarity in [0, 100] between two hand vectors.
    Tuned so identical poses ~95–100 and confused poses drop hard.
    """
    mean_d, dists = weighted_mean_distance(live, ref)
    # Distance scale: room for mild camera jitter while keeping confusion low
    geom = float(max(0.0, min(1.0, 1.0 - mean_d / 0.48)))
    cos = cosine(live, ref)
    cos_n = (cos + 1.0) / 2.0
    raw = 100.0 * (0.55 * geom + 0.45 * cos_n)
    if raw >= 88:
        raw = min(100.0, raw * 1.04)
    return raw, cos, dists


def _glyph_kind(label: str) -> str:
    if isinstance(label, str) and len(label) == 1:
        if label.isalpha():
            return "letter"
        if label.isdigit():
            return "digit"
    return "phrase"


def competitor_labels(target: str, refs: Dict[str, np.ndarray]) -> Iterable[str]:
    """Letters compete with letters, digits with digits, phrases with phrases."""
    want = _glyph_kind(target)
    for lab in refs:
        if lab == target:
            continue
        if _glyph_kind(lab) == want:
            yield lab


def score_against_library(
    live: np.ndarray,
    target: str,
    refs: Dict[str, np.ndarray],
) -> Tuple[float, float, np.ndarray, Optional[str], float]:
    """
    Returns (match_pct, cos_to_target, joint_dists, best_other_label, margin).
    match_pct uses absolute quality AND margin vs best competing label.
    """
    if target not in refs:
        return 0.0, -1.0, np.zeros(21, dtype=np.float32), None, 0.0

    abs_score, cos_t, dists = pair_score(live, refs[target])
    best_other = None
    best_other_cos = -2.0
    for lab in competitor_labels(target, refs):
        c = cosine(live, refs[lab])
        if c > best_other_cos:
            best_other_cos = c
            best_other = lab

    margin = cos_t - best_other_cos if best_other is not None else 1.0
    # Perfect / near-perfect absolute match: trust geometry over thin margins
    if abs_score >= 96 and cos_t >= 0.98:
        match = float(min(100.0, abs_score))
        return match, cos_t, dists, best_other, float(margin)

    if margin < -0.02:
        abs_score *= 0.42
    elif margin < 0.03:
        abs_score *= 0.62
    elif margin < 0.07:
        abs_score *= 0.82
    elif margin < 0.12:
        abs_score *= 0.92

    match = float(max(0.0, min(100.0, abs_score)))
    return match, cos_t, dists, best_other, float(margin)


def predict_nn(
    live: np.ndarray,
    refs: Dict[str, np.ndarray],
    *,
    glyphs_only: bool = True,
) -> Tuple[str, float]:
    """Nearest-neighbor label with confidence from softmax over pair scores."""
    if glyphs_only:
        pool = {
            k: v
            for k, v in refs.items()
            if _glyph_kind(k) in ("letter", "digit")
        }
    else:
        pool = refs
    if not pool:
        pool = refs
    if not pool:
        return "—", 0.0
    labels: List[str] = []
    scores: List[float] = []
    for lab, ref in pool.items():
        s, _, _ = pair_score(live, ref)
        labels.append(lab)
        scores.append(s)
    arr = np.asarray(scores, dtype=np.float32)
    t = 8.0
    ex = np.exp((arr - arr.max()) / t)
    proba = ex / (ex.sum() + 1e-8)
    idx = int(np.argmax(proba))
    return labels[idx], float(proba[idx])
