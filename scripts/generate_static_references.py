"""
Generate approximate ASL finger-spelling / number / phrase reference landmarks.

Educational templates (normalized wrist-relative) so Learn Mode works before
real captures. Replace by recording into data/references/static_landmarks.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "references" / "static_landmarks.json"

PHRASE_KEYS = [
    "hello",
    "thank_you",
    "yes",
    "no",
    "water",
    "family",
    "iloveyou",
    "need_help",
    "emergency",
    "hospital",
    "medical_help",
    "pain",
]


def base_open_hand() -> np.ndarray:
    """21 landmarks roughly like an open facing palm (normalized)."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0, 0, 0]
    pts[1] = [-0.15, -0.05, 0.02]
    pts[2] = [-0.28, -0.18, 0.03]
    pts[3] = [-0.35, -0.32, 0.02]
    pts[4] = [-0.40, -0.45, 0.01]
    pts[5] = [-0.08, -0.18, 0]
    pts[6] = [-0.10, -0.40, 0]
    pts[7] = [-0.11, -0.58, 0]
    pts[8] = [-0.12, -0.75, 0]
    pts[9] = [0.00, -0.20, 0]
    pts[10] = [0.00, -0.45, 0]
    pts[11] = [0.00, -0.65, 0]
    pts[12] = [0.00, -0.82, 0]
    pts[13] = [0.08, -0.18, 0]
    pts[14] = [0.10, -0.40, 0]
    pts[15] = [0.11, -0.58, 0]
    pts[16] = [0.12, -0.72, 0]
    pts[17] = [0.15, -0.12, 0]
    pts[18] = [0.18, -0.30, 0]
    pts[19] = [0.20, -0.45, 0]
    pts[20] = [0.22, -0.58, 0]
    return pts


def curl_finger(pts: np.ndarray, idxs: list[int], amount: float = 0.55) -> None:
    for i in idxs[1:]:
        pts[i, 1] = pts[idxs[0], 1] - amount * abs(pts[i, 1] - pts[idxs[0], 1]) * 0.2
        pts[i, 1] *= 1.0 - amount * 0.65
        pts[i, 0] *= 1.0 - amount * 0.15


def make_letter(kind: str) -> np.ndarray:
    p = base_open_hand()
    raw = kind.strip()
    k = raw.upper() if len(raw) == 1 else raw.lower()

    if k == "A":  # fist + thumb aside (clearly lateral)
        curl_finger(p, [5, 6, 7, 8], 0.95)
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [-0.55, -0.05, 0.02]
        p[3] = [-0.42, -0.08, 0.02]
    elif k == "B":  # flat hand, thumb tucked across palm
        p[4] = [0.08, -0.18, 0.06]
        p[3] = [0.05, -0.12, 0.05]
        p[1:5, 0] *= 0.25
        p[8, 1] = -0.78
        p[12, 1] = -0.82
        p[16, 1] = -0.74
        p[20, 1] = -0.62
    elif k == "C":  # curved C
        for i in range(1, 21):
            p[i, 0] -= 0.18
            p[i, 1] *= 0.72
        p[8, 0] -= 0.22
        p[12, 0] -= 0.25
        p[4, 0] = -0.52
        p[4, 1] = -0.28
        p[20, 0] = -0.08
    elif k == "D":  # index up, others curled, thumb on middle
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.02, -0.28, 0.02]
        p[8] = [-0.10, -0.82, 0]
        p[7] = [-0.10, -0.62, 0]
    elif k == "E":  # fingertips on palm, thumb across (no lateral thumb)
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 1.0)
        p[4] = [0.05, -0.18, 0.04]
        p[8] = [-0.05, -0.18, 0.02]
        p[12] = [0.0, -0.16, 0.02]
        p[16] = [0.05, -0.14, 0.02]
        p[20] = [0.10, -0.12, 0.02]
    elif k == "F":  # OK: thumb+index circle, others up
        p[4] = [-0.14, -0.38, 0]
        p[8] = [-0.14, -0.38, 0]
        p[6] = [-0.10, -0.28, 0]
        p[7] = [-0.12, -0.33, 0]
        p[12, 1] = -0.80
        p[16, 1] = -0.74
        p[20, 1] = -0.62
    elif k == "G":  # index + thumb horizontal gun
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.62, -0.08, 0]
        p[4] = [-0.50, 0.05, 0]
        p[6] = [-0.32, -0.08, 0]
        p[7] = [-0.48, -0.08, 0]
    elif k == "H":  # index+middle horizontal-ish out
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.08, -0.05, 0]
        p[8] = [-0.55, -0.35, 0]
        p[12] = [-0.52, -0.22, 0]
    elif k == "I":  # pinky up only
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.18, -0.08, 0]
        p[20] = [0.28, -0.72, 0]
        p[19] = [0.24, -0.55, 0]
    elif k == "J":  # pinky tip shifted (hook start)
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.22, -0.06, 0]
        p[20] = [0.42, -0.48, 0.04]
        p[19] = [0.32, -0.40, 0.02]
        p[18] = [0.24, -0.30, 0.01]
    elif k == "K":  # index+middle V, thumb between
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.22, -0.78, 0]
        p[12] = [0.18, -0.72, 0]
        p[4] = [0.02, -0.42, 0]
    elif k == "L":  # L shape
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.02, -0.82, 0]
        p[4] = [-0.62, 0.02, 0]
        p[3] = [-0.48, 0.0, 0]
    elif k == "M":  # thumb under 3 fingers (thumb more central)
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.88)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[4] = [0.12, -0.12, 0.06]
        p[8, 1] = -0.28
        p[12, 1] = -0.30
        p[16, 1] = -0.28
    elif k == "N":  # thumb under 2 fingers only
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12]):
            curl_finger(p, idxs, 0.88)
        for idxs in ([13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.05, -0.10, 0.05]
        p[8, 1] = -0.30
        p[12, 1] = -0.32
        p[16, 1] = -0.12
    elif k == "O":  # round O
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.45)
        p[4] = [-0.28, -0.30, 0]
        p[8] = [-0.22, -0.38, 0.02]
        p[12] = [-0.10, -0.40, 0.02]
        p[16] = [0.02, -0.36, 0.02]
        p[20] = [0.10, -0.28, 0.02]
    elif k == "P":  # K pointing down
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.15, 0.42, 0]
        p[12] = [0.12, 0.38, 0]
        p[4] = [0.0, 0.08, 0]
        p[7] = [-0.12, 0.22, 0]
        p[11] = [0.10, 0.20, 0]
    elif k == "Q":  # G pointing down
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.28, 0.55, 0]
        p[4] = [-0.38, 0.35, 0]
        p[7] = [-0.22, 0.35, 0]
    elif k == "R":  # crossed index/middle
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [0.08, -0.78, 0]
        p[12] = [-0.08, -0.76, 0]
        p[7] = [0.02, -0.58, 0]
        p[11] = [-0.02, -0.58, 0]
        p[4] = [0.12, -0.10, 0]
    elif k == "S":  # fist thumb OVER fingers (front)
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.02, -0.32, 0.08]
        p[3] = [-0.05, -0.22, 0.06]
        p[8, 1] = -0.22
        p[12, 1] = -0.24
    elif k == "T":  # thumb BETWEEN index and middle knuckle
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.92)
        p[4] = [-0.08, -0.35, 0.05]
        p[8, 1] = -0.26
        p[12, 1] = -0.28
        p[5] = [-0.12, -0.22, 0]
        p[9] = [0.02, -0.24, 0]
    elif k == "U":  # index+middle together
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.02, -0.80, 0]
        p[12] = [0.02, -0.80, 0]
        p[4] = [0.14, -0.08, 0]
    elif k == "V":  # victory spread
        curl_finger(p, [13, 14, 15, 16], 0.95)
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.28, -0.80, 0]
        p[12] = [0.28, -0.80, 0]
        p[4] = [0.12, -0.06, 0]
    elif k == "W":  # 3 fingers spread
        curl_finger(p, [17, 18, 19, 20], 0.95)
        p[8] = [-0.28, -0.76, 0]
        p[12] = [0.0, -0.82, 0]
        p[16] = [0.28, -0.74, 0]
        p[4] = [0.15, -0.05, 0]
    elif k == "X":  # hooked index
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.05, -0.38, 0]
        p[7] = [-0.08, -0.52, 0]
        p[6] = [-0.10, -0.42, 0]
        p[4] = [0.08, -0.12, 0]
    elif k == "Y":  # Y hand
        for idxs in ([5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]):
            curl_finger(p, idxs, 0.95)
        p[4] = [-0.55, -0.22, 0]
        p[20] = [0.42, -0.58, 0]
    elif k == "Z":  # index sideways Z start
        for idxs in ([9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]):
            curl_finger(p, idxs, 0.95)
        p[8] = [-0.58, -0.18, 0]
        p[7] = [-0.40, -0.28, 0]
        p[6] = [-0.22, -0.25, 0]
        p[4] = [0.10, -0.10, 0]
    # Phrase handshapes
    elif k in ("hello", "hi"):
        return make_letter("B")
    elif k in ("thank_you", "thanks"):
        p[4] = [-0.38, -0.12, 0]
        p[8, 1] = -0.70
        p[12, 1] = -0.72
        p[16, 1] = -0.68
        p[20, 1] = -0.55
        p[8, 0] = -0.05
    elif k == "yes":
        return make_letter("S")
    elif k == "no":
        return make_letter("U")
    elif k == "water":
        return make_letter("W")
    elif k == "family":
        return make_letter("F")
    elif k in ("iloveyou", "i_love_you", "love"):
        curl_finger(p, [9, 10, 11, 12], 0.95)
        curl_finger(p, [13, 14, 15, 16], 0.95)
        p[4] = [-0.50, -0.30, 0]
        p[8] = [-0.18, -0.80, 0]
        p[20] = [0.35, -0.60, 0]
    elif k in ("need_help", "help"):
        p[4] = [-0.48, -0.38, 0]
        p[8, 1] = -0.74
        p[12, 1] = -0.80
        p[16, 1] = -0.72
        p[20, 1] = -0.60
        p[8, 0] = -0.18
        p[16, 0] = 0.18
    elif k == "emergency":
        return make_letter("E")
    elif k == "hospital":
        return make_letter("H")
    elif k in ("medical_help", "medical"):
        return make_letter("M")
    elif k == "pain":
        return make_letter("D")
    elif k.isdigit():
        n = int(k)
        # Distinct digit poses (not letter aliases) for cue cards + digit scoring
        if n == 0:
            return make_letter("O")
        if n == 1:
            curl_finger(p, [9, 10, 11, 12], 0.95)
            curl_finger(p, [13, 14, 15, 16], 0.95)
            curl_finger(p, [17, 18, 19, 20], 0.95)
            p[4] = [0.15, -0.15, 0]  # thumb out vs D
            p[8] = [-0.08, -0.85, 0]
        elif n == 2:
            curl_finger(p, [13, 14, 15, 16], 0.95)
            curl_finger(p, [17, 18, 19, 20], 0.95)
            p[8] = [-0.22, -0.78, 0]
            p[12] = [0.22, -0.78, 0]
            p[4] = [0.05, -0.20, 0.05]  # thumb tucked vs V
        elif n == 3:
            curl_finger(p, [17, 18, 19, 20], 0.95)
            p[8] = [-0.25, -0.75, 0]
            p[12] = [0.0, -0.80, 0]
            p[16] = [0.25, -0.72, 0]
            p[4] = [0.08, -0.22, 0.05]
        elif n == 4:
            p[4] = [0.02, -0.15, 0.08]
            p[8, 1] = -0.76
            p[12, 1] = -0.80
            p[16, 1] = -0.74
            p[20, 1] = -0.64
        elif n == 5:
            return base_open_hand()
        elif n == 6:
            p[4] = [0.22, -0.40, 0]
            p[20] = [0.22, -0.40, 0]
            p[8, 1] = -0.75
            p[12, 1] = -0.78
            p[16, 1] = -0.72
        elif n == 7:
            p[4] = [0.12, -0.42, 0]
            p[16] = [0.12, -0.42, 0]
            p[8, 1] = -0.75
            p[12, 1] = -0.78
            p[20, 1] = -0.55
        elif n == 8:
            p[4] = [0.06, -0.44, 0]
            p[12] = [0.06, -0.44, 0]
            p[8, 1] = -0.75
            p[16, 1] = -0.70
            p[20, 1] = -0.55
        elif n == 9:
            p[4] = [-0.08, -0.42, 0]
            p[8] = [-0.08, -0.42, 0]
            p[12, 1] = -0.78
            p[16, 1] = -0.72
            p[20, 1] = -0.58
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
    keys = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + [str(i) for i in range(10)] + PHRASE_KEYS
    for ch in keys:
        pts = make_letter(ch)
        if pts.ndim == 1:
            pts = pts.reshape(21, 3)
        landmarks[ch] = normalize(pts).tolist()
        if len(ch) == 1 and ch.isalpha():
            cat = "alphabet"
        elif len(ch) == 1 and ch.isdigit():
            cat = "number"
        else:
            cat = "phrase"
        meta[ch] = {
            "label": ch,
            "category": cat,
            "system": "ASL-approx",
            "note": "Template pose — replace with recorded landmarks for production accuracy.",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "feature_dim": 63,
        "description": "Canonical static hand references for Learn Mode (alphabet, numbers, phrases)",
        "landmarks": landmarks,
        "meta": meta,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(landmarks)} signs)")


if __name__ == "__main__":
    main()
