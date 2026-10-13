"""
Static sign classifier for A–Z.

Uses the trained MLP (models/static_mlp.json). Falls back to nearest-neighbour
over the template poses only when the trained model is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.engine.hand_similarity import predict_nn
from src.engine.static_model import StaticLetterModel
from src.tracking.constants import HAND_FEATURE_DIM
from src.tracking.extractor import hand_from_holistic_array

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "data" / "references" / "static_landmarks.json"


class StaticClassifier:
    def __init__(self, ref_path: Path | str = REF_PATH) -> None:
        self.ref_path = Path(ref_path)
        self.model = StaticLetterModel()
        self.refs: Dict[str, np.ndarray] = {}
        if not self.model.ready:
            self._load_refs()

    def _load_refs(self) -> None:
        if not self.ref_path.exists():
            return
        data = json.loads(self.ref_path.read_text(encoding="utf-8"))
        self.refs = {
            str(k): np.asarray(v, dtype=np.float32).reshape(-1)[:HAND_FEATURE_DIM]
            for k, v in data.get("landmarks", {}).items()
        }

    @property
    def ready(self) -> bool:
        return self.model.ready or bool(self.refs)

    @property
    def source(self) -> str:
        return "trained_mlp" if self.model.ready else "template_nn"

    def predict(self, hand_63: np.ndarray) -> Tuple[str, float]:
        vec = hand_from_holistic_array(hand_63)
        if self.model.ready:
            return self.model.predict(vec)
        return predict_nn(vec, self.refs, glyphs_only=True)

    def list_labels(self) -> List[str]:
        if self.model.ready:
            return list(self.model.labels)
        return sorted(self.refs.keys(), key=lambda x: (len(x), x))
