"""
Learn Mode feedback: compare live hand landmarks to a canonical reference.

Returns match %, tip text, and per-landmark error highlights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.tracking.constants import HAND_FEATURE_DIM
from src.tracking.extractor import hand_from_holistic_array

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "data" / "references" / "static_landmarks.json"

# MediaPipe hand landmark names (subset tips for coaching)
TIP_INDEX = {
    4: "thumb tip",
    8: "index tip",
    12: "middle tip",
    16: "ring tip",
    20: "pinky tip",
}


class FeedbackEvaluator:
    def __init__(self, ref_path: Path | str = REF_PATH) -> None:
        self.ref_path = Path(ref_path)
        self.refs: Dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        if not self.ref_path.exists():
            self.refs = {}
            return
        data = json.loads(self.ref_path.read_text(encoding="utf-8"))
        self.refs = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)[:HAND_FEATURE_DIM]
            for k, v in data.get("landmarks", {}).items()
        }

    def evaluate(self, target: str, hand_63: np.ndarray) -> Dict[str, Any]:
        key = str(target).strip().upper() if len(str(target).strip()) == 1 else str(target).strip()
        # numbers stay as "0".."9"
        if key.lower() in self.refs and key not in self.refs:
            key = key.lower()
        if key not in self.refs and key.upper() in self.refs:
            key = key.upper()
        if key not in self.refs:
            return {
                "target": target,
                "match_pct": 0.0,
                "tip": f"No reference pose for '{target}'. Add it to the dictionary.",
                "ok": False,
                "worst": [],
            }

        live = hand_from_holistic_array(hand_63)
        ref = self.refs[key]
        live_pts = live.reshape(21, 3)
        ref_pts = ref.reshape(21, 3)

        # Euclidean distances in normalized space
        dists = np.linalg.norm(live_pts - ref_pts, axis=1)
        mean_d = float(np.mean(dists))
        # Soften scoring for beginners — templates are approximate
        match = float(max(0.0, min(100.0, (1.0 - mean_d / 0.72) * 100.0)))
        a = live / (np.linalg.norm(live) + 1e-8)
        b = ref / (np.linalg.norm(ref) + 1e-8)
        cos = float(np.dot(a, b))
        match = float(max(0.0, min(100.0, 0.35 * match + 0.65 * ((cos + 1) / 2) * 100)))
        # Gentle boost so clear attempts don't feel stuck low
        match = float(min(100.0, match * 1.12))

        worst_idx = list(np.argsort(-dists)[:3])
        worst = []
        tips: List[str] = []
        for i in worst_idx:
            name = TIP_INDEX.get(int(i), f"joint {i}")
            delta = live_pts[i] - ref_pts[i]
            worst.append({"index": int(i), "name": name, "error": float(dists[i])})
            # Coaching hint from dominant axis
            ax = int(np.argmax(np.abs(delta[:2])))
            axis_name = ["sideways", "up/down"][ax]
            direction = "higher" if (ax == 1 and delta[1] < 0) else "lower" if ax == 1 else "outward" if delta[0] > 0 else "inward"
            if dists[i] > 0.08:
                tips.append(f"Adjust your {name} slightly {direction} ({axis_name}).")

        if match >= 90:
            tip = f"Excellent! Match {match:.0f}% — hold steady."
        elif match >= 75:
            tip = tips[0] if tips else f"Good — fine-tune fingers ({match:.0f}%)."
        elif match >= 50:
            tip = tips[0] if tips else f"Getting closer ({match:.0f}%). Check finger curl."
        else:
            tip = tips[0] if tips else f"Reshape the hand to match '{key}' ({match:.0f}%)."

        return {
            "target": key,
            "match_pct": round(match, 1),
            "tip": tip,
            "ok": match >= 75.0,
            "worst": worst,
            "mean_distance": round(mean_d, 4),
        }
