"""
Generate approximate ASL finger-spelling / number reference landmarks.

These are educational templates (normalized wrist-relative) so Learn Mode
works before the team records real captures. Replace by recording real
poses into data/references/static_landmarks.json when ready.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "references" / "static_landmarks.json"


def base_open_hand() -> np.ndarray:
    """21 landmarks roughly like an open facing palm (normalized)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # wrist 0
    pts[0] = [0, 0, 0]
    # thumb CMC..tip 1-4
    pts[1] = [-0.15, -0.05, 0.02]
    pts[2] = [-0.28, -0.18, 0.03]
    pts[3] = [-0.35, -0.32, 0.02]
    pts[4] = [-0.40, -0.45, 0.01]
    # index 5-8
    pts[5] = [-0.08, -0.18, 0]
    pts[6] = [-0.10, -0.40, 0]
    pts[7] = [-0.11, -0.58, 0]
    pts[8] = [-0.12, -0.75, 0]
    # middle 9-12
    pts[9] = [0.00, -0.20, 0]
    pts[10] = [0.00, -0.45, 0]
    pts[11] = [0.00, -0.65, 0]
    pts[12] = [0.00, -0.82, 0]
    # ring 13-16
    pts[13] = [0.08, -0.18, 0]
    pts[14] = [0.10, -0.40, 0]
    pts[15] = [0.11, -0.58, 0]
    pts[16] = [0.12, -0.72, 0]
    # pinky 17-20
    pts[17] = [0.15, -0.12, 0]
    pts[18] = [0.18, -0.30, 0]
    pts[19] = [0.20, -0.45, 0]
    pts[20] = [0.22, -0.58, 0]
    return pts


def curl_finger(pts: np.ndarray, idxs: list[int], amount: float = 0.55) -> None:
    """Pull fingertips toward palm (positive y toward wrist in our coords is down = +y after flip? we use -y up)."""
    for i in idxs[1:]:
        pts[i, 1] = pts[idxs[0], 1] - amount * abs(pts[i, 1] - pts[idxs[0], 1]) * 0.2
        pts[i, 1] *= 1.0 - amount * 0.65
        pts[i, 0] *= 1.0 - amount * 0.15


def make_letter(kind: str) -> np.ndarray:
    p = base_open_hand()
    k = kind.upper()

    # Fist-like helpers
    fist_fingers = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if k == "A":  # fist + thumb aside
        curl_finger(p, [5, 6, 7, 8], 0.9)
        curl_finger(p, [9, 10, 11, 12], 0.9)
        curl_finger(p, [13, 14, 15, 16], 0.9)
        curl_finger(p, [17, 18, 19, 20], 0.9)
        p[4] = [-0.42, -0.15, 0.02]
    elif k == "B":  # flat hand, thumb tucked
        p[4] = [0.05, -0.10, 0.05]
        p[1:5, 0] *= 0.3
    elif k == "C":  # curved C
        for i in range(1, 21):
            p[i, 0] -= 0.12
            p[i, 1] *= 0.75
        p[8, 0] -= 0.15
        p[12, 0] -= 0.18
        p[4, 0] = -0.45
        p[4, 1] = -0.35
    elif k == "D":  # index up, others curled
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.02, -0.25, 0.02]
    elif k == "E":  # curled fist tips on palm
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 1.0)
        p[4] = [-0.05, -0.12, 0.02]
    elif k == "F":  # OK shape: thumb+index touch, others up
        p[4] = [-0.12, -0.35, 0]
        p[8] = [-0.12, -0.35, 0]
        p[6:8] *= 0.5
    elif k == "G":  # index + thumb horizontal
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.55, -0.15, 0]
        p[4] = [-0.45, -0.05, 0]
        p[6] = [-0.30, -0.12, 0]
        p[7] = [-0.42, -0.14, 0]
    elif k == "H":  # index+middle out
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.05, -0.08, 0]
        p[8, 1] = -0.70
        p[12, 1] = -0.72
    elif k == "I":  # pinky up
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.2, -0.1, 0]
        p[20, 1] = -0.70
    elif k == "J":  # like I (motion ignored in static)
        return make_letter("I")
    elif k == "K":  # index+middle V with thumb
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.15, -0.75, 0]
        p[12] = [0.12, -0.75, 0]
        p[4] = [0.0, -0.45, 0]
    elif k == "L":  # L shape
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.05, -0.78, 0]
        p[4] = [-0.55, -0.05, 0]
    elif k == "M":  # thumb under 3 fingers
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.85)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.05, -0.05, 0.05]
    elif k == "N":  # thumb under 2 fingers
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12]):
            curl_finger(p, idxs, 0.85)
        for idxs in ([13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[4] = [0.02, -0.05, 0.05]
    elif k == "O":  # O circle
        for i in [4, 8, 12, 16, 20]:
            p[i] = [-0.15, -0.35, 0.02]
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.55)
        p[4] = [-0.22, -0.32, 0]
    elif k == "P":  # like K pointing down-ish
        q = make_letter("K")
        q[:, 1] *= -0.4
        return q
    elif k == "Q":  # G pointing down
        q = make_letter("G")
        q[:, 1] *= -0.5
        return q
    elif k == "R":  # crossed index/middle
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [0.05, -0.75, 0]
        p[12] = [-0.05, -0.75, 0]
        p[4] = [0.08, -0.12, 0]
    elif k == "S":  # fist thumb over fingers
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.05, -0.22, 0.03]
    elif k == "T":  # thumb between index/middle
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.9)
        p[4] = [-0.02, -0.28, 0.04]
    elif k == "U":  # index+middle together up
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.04, -0.78, 0]
        p[12] = [0.04, -0.78, 0]
        p[4] = [0.1, -0.1, 0]
    elif k == "V":  # victory
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.18, -0.78, 0]
        p[12] = [0.18, -0.78, 0]
        p[4] = [0.1, -0.08, 0]
    elif k == "W":  # 3 fingers
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8, 1] = -0.75
        p[12, 1] = -0.78
        p[16, 1] = -0.72
        p[4] = [0.12, -0.08, 0]
    elif k == "X":  # hooked index
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.08, -0.45, 0]
        p[7] = [-0.08, -0.55, 0]
        p[4] = [0.05, -0.15, 0]
    elif k == "Y":  # Y hand
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.50, -0.25, 0]
        p[20] = [0.35, -0.55, 0]
    elif k == "Z":  # like index (motion ignored)
        return make_letter("D")
    elif k.isdigit():
        n = int(k)
        if n == 0:
            return make_letter("O")
        if n == 1:
            return make_letter("D")
        if n == 2:
            return make_letter("V")
        if n == 3:
            return make_letter("W")
        if n == 4:
            # four fingers, thumb tucked
            p[4] = [0.05, -0.08, 0.05]
            return p
        if n == 5:
            return base_open_hand()
        if n == 6:
            # thumb+pinky touch-ish, others up — simplified
            p[4] = [0.18, -0.35, 0]
            p[20] = [0.18, -0.35, 0]
            return p
        if n == 7:
            p[4] = [0.10, -0.40, 0]
            p[16] = [0.10, -0.40, 0]
            return p
        if n == 8:
            p[4] = [0.05, -0.42, 0]
            p[12] = [0.05, -0.42, 0]
            return p
        if n == 9:
            p[4] = [-0.05, -0.40, 0]
            p[8] = [-0.05, -0.40, 0]
            return p
    return p


def normalize(pts: np.ndarray) -> np.ndarray:
    wrist = pts[0].copy()
    c = pts - wrist
    scale = float(np.max(np.linalg.norm(c[:, :2], axis=1)) or 1.0)
    c /= scale
    return c.flatten().astype(np.float32)


def main() -> None:
    landmarks = {}
    meta = {}
    for ch in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)]:
        landmarks[ch] = normalize(make_letter(ch)).tolist()
        meta[ch] = {
            "label": ch,
            "category": "alphabet" if ch.isalpha() else "number",
            "system": "ASL-approx",
            "note": "Template pose — replace with recorded landmarks for production accuracy.",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "feature_dim": 63,
        "description": "Canonical static hand references for Learn Mode",
        "landmarks": landmarks,
        "meta": meta,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(landmarks)} signs)")


if __name__ == "__main__":
    main()
