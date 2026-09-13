"""
Static sign classifier for A–Z and 0–9.

Uses nearest-neighbor against canonical reference landmarks by default.
Optionally loads a trained sklearn joblib model from models/static_clf.joblib.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tracking.constants import HAND_FEATURE_DIM
from src.tracking.extractor import hand_from_holistic_array

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "data" / "references" / "static_landmarks.json"
MODEL_PATH = ROOT / "models" / "static_clf.joblib"


class StaticClassifier:
    def __init__(self, ref_path: Path | str = REF_PATH) -> None:
        self.ref_path = Path(ref_path)
        self.labels: List[str] = []
        self.refs: Dict[str, np.ndarray] = {}
        self._sklearn = None
        self._load_refs()
        self._try_load_sklearn()

    def _load_refs(self) -> None:
        if not self.ref_path.exists():
            self.labels = []
            self.refs = {}
            return
        data = json.loads(self.ref_path.read_text(encoding="utf-8"))
        self.refs = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)[:HAND_FEATURE_DIM]
            for k, v in data.get("landmarks", {}).items()
        }
        self.labels = sorted(self.refs.keys(), key=lambda x: (len(x), x))

    def _try_load_sklearn(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            import joblib

            self._sklearn = joblib.load(MODEL_PATH)
        except Exception:
            self._sklearn = None

    @property
    def ready(self) -> bool:
        return bool(self.refs) or self._sklearn is not None

    def predict(self, hand_63: np.ndarray) -> Tuple[str, float]:
        vec = hand_from_holistic_array(hand_63)
        if self._sklearn is not None:
            try:
                proba = self._sklearn.predict_proba([vec])[0]
                idx = int(np.argmax(proba))
                label = str(self._sklearn.classes_[idx])
                return label, float(proba[idx])
            except Exception:
                pass
        if not self.refs:
            return "—", 0.0
        best_label = "—"
        best_score = -1.0
        for label, ref in self.refs.items():
            # cosine similarity on normalized vectors
            a = vec / (np.linalg.norm(vec) + 1e-8)
            b = ref / (np.linalg.norm(ref) + 1e-8)
            sim = float(np.dot(a, b))
            if sim > best_score:
                best_score = sim
                best_label = label
        # map [-1,1] roughly to [0,1]
        conf = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
        return best_label, conf

    def list_labels(self) -> List[str]:
        if self._sklearn is not None:
            try:
                return [str(c) for c in self._sklearn.classes_]
            except Exception:
                pass
        return list(self.labels)
