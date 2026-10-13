"""
Learn Mode feedback.

Letters (and phrases built on a letter handshape) are scored by the trained MLP:
match_pct = P(target | hand) * 100. Template poses are only used for the
"nudge your X tip" hint, and as the scorer for phrases the model cannot cover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.engine.static_model import StaticLetterModel
from src.tracking.constants import HAND_FEATURE_DIM
from src.tracking.extractor import hand_from_holistic_array

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "data" / "references" / "static_landmarks.json"

_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_PARTS = ("base", "knuckle", "middle joint", "tip")
TIP_INDEX = {0: "wrist", **{i: f"{_FINGERS[(i - 1) // 4]} {_PARTS[(i - 1) % 4]}" for i in range(1, 21)}}

PASS_PCT = 70.0


class FeedbackEvaluator:
    def __init__(self, ref_path: Path | str = REF_PATH) -> None:
        self.ref_path = Path(ref_path)
        self.refs: Dict[str, np.ndarray] = {}
        self.model = StaticLetterModel()
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

    def _resolve_key(self, target: str) -> str:
        key = str(target).strip()
        if len(key) == 1 and key.isalpha():
            key = key.upper()
        if key in self.refs:
            return key
        if key.lower() in self.refs:
            return key.lower()
        if key.upper() in self.refs:
            return key.upper()
        return key

    def _joint_hints(
        self, ref: Optional[np.ndarray], live: np.ndarray
    ) -> tuple[List[Dict[str, Any]], List[str], Optional[float]]:
        if ref is None:
            return [], [], None
        live_pts = live.reshape(21, 3)
        ref_pts = ref.reshape(21, 3)
        dists = np.linalg.norm(live_pts - ref_pts, axis=1).astype(np.float32)
        worst, tips = [], []
        for i in list(np.argsort(-dists)[:3]):
            name = TIP_INDEX.get(int(i), f"joint {i}")
            delta = live_pts[i] - ref_pts[i]
            worst.append({"index": int(i), "name": name, "error": float(dists[i])})
            ax = int(np.argmax(np.abs(delta[:2])))
            if ax == 1:
                direction = "higher" if delta[1] < 0 else "lower"
            else:
                direction = "outward" if delta[0] > 0 else "inward"
            if dists[i] > 0.08:
                tips.append(f"Nudge your {name} {direction}.")
        return worst, tips, float(np.mean(dists))

    def _template_score(self, key: str, live: np.ndarray) -> float:
        ref = self.refs[key]
        mean_d = float(np.mean(np.linalg.norm(live.reshape(21, 3) - ref.reshape(21, 3), axis=1)))
        geom = max(0.0, min(100.0, (1.0 - mean_d / 0.72) * 100.0))
        cos = float(np.dot(live, ref) / ((np.linalg.norm(live) + 1e-8) * (np.linalg.norm(ref) + 1e-8)))
        return float(min(100.0, (0.35 * geom + 0.65 * ((cos + 1.0) / 2.0) * 100.0) * 1.12))

    def evaluate(self, target: str, hand_63: np.ndarray) -> Dict[str, Any]:
        live = hand_from_holistic_array(hand_63)
        key = self._resolve_key(target)
        label = self.model.target_label(target) if self.model.ready else None
        ref = self.model.nearest_exemplar(label, live) if label is not None else None
        if ref is None:
            ref = self.refs.get(key)
        worst, tips, mean_d = self._joint_hints(ref, live)

        if label is not None:
            if label in self.model.digit_labels:
                p = self.model.digit_proba(live)
                names = self.model.digit_labels
            else:
                p = self.model.proba(live)
                names = self.model.labels
            match = float(p[names.index(label)] * 100.0)
            top_i = int(np.argmax(p))
            top, top_p = names[top_i], float(p[top_i])
            scorer = "trained_mlp"
            if match >= 90:
                tip = f"Locked in — hold still ({match:.0f}%)."
            elif match >= PASS_PCT:
                tip = f"Good {label} ({match:.0f}%). Hold…"
            elif top != label and top_p >= 0.4:
                tip = f"That reads as “{top}”. " + (tips[0] if tips else f"Reshape toward “{label}”.")
            else:
                tip = tips[0] if tips else f"Reshape toward “{label}” ({match:.0f}%)."
            extra = {"predicted": top, "predicted_conf": round(top_p, 3)}
        elif key in self.refs:
            match = self._template_score(key, live)
            scorer = "template"
            if match >= 90:
                tip = f"Locked in — hold still ({match:.0f}%)."
            elif match >= PASS_PCT:
                tip = tips[0] if tips else f"Almost ({match:.0f}%). Hold…"
            else:
                tip = tips[0] if tips else f"Reshape toward “{key}” ({match:.0f}%)."
            extra = {}
        else:
            return {"target": target, "match_pct": 0.0, "tip": f"No reference for '{target}'.", "ok": False, "worst": []}

        out = {
            "target": key,
            "match_pct": round(match, 1),
            "tip": tip,
            "ok": match >= PASS_PCT,
            "worst": worst,
            "scorer": scorer,
            **extra,
        }
        if mean_d is not None:
            out["mean_distance"] = round(mean_d, 4)
        return out
