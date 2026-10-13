"""
Trained A–Z static-letter MLP (real MediaPipe landmarks from Hugging Face ASL
datasets, see scripts/ml/train_static.py). Pure-numpy forward pass; the same
JSON is served to the browser (web/js/static_model.js).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.engine.hand_features import model_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "static_mlp.json"
DIGIT_PATH = ROOT / "models" / "digit_mlp.json"
EXEMPLAR_PATH = ROOT / "data" / "references" / "letter_exemplars.json"

# Lesson phrases whose ASL sign is built on a fingerspelled handshape. The model
# verifies the handshape only; the motion part of the sign is not scored.
PHRASE_HANDSHAPES: Dict[str, str] = {
    "hello": "B",
    "yes": "S",
    "water": "W",
    "family": "F",
    "hospital": "H",
    "emergency": "E",
    "medical_help": "M",
    "pain": "D",
}


class StaticLetterModel:
    def __init__(self, path: Path | str = MODEL_PATH) -> None:
        self.path = Path(path)
        self.labels: List[str] = []
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        self.meta: Dict = {}
        self.exemplars: Dict[str, np.ndarray] = {}
        self.input_dim = 63
        self.digit_labels: List[str] = []
        self.digit_layers: List[Tuple[np.ndarray, np.ndarray]] = []
        self.digit_input_dim = 63
        self.load_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.load_error = f"{self.path.name} missing — run scripts/ml/train_static.py"
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.labels = [str(x) for x in data["labels"]]
        self.input_dim = int(data.get("input_dim", 63))
        # JSON stores W as (out, in); keep (in, out) for x @ W
        self.layers = [
            (np.asarray(L["W"], dtype=np.float32).T, np.asarray(L["b"], dtype=np.float32))
            for L in data["layers"]
        ]
        self.meta = {k: data.get(k) for k in ("feature", "metrics", "trained_at", "sources")}
        self.exemplars: Dict[str, np.ndarray] = {}
        if EXEMPLAR_PATH.exists():
            ex = json.loads(EXEMPLAR_PATH.read_text(encoding="utf-8")).get("exemplars", {})
            self.exemplars = {k: np.asarray(v, dtype=np.float32) for k, v in ex.items()}
        if DIGIT_PATH.exists():
            dig = json.loads(DIGIT_PATH.read_text(encoding="utf-8"))
            self.digit_labels = [str(x) for x in dig["labels"]]
            self.digit_input_dim = int(dig.get("input_dim", 63))
            self.digit_layers = [
                (np.asarray(L["W"], dtype=np.float32).T, np.asarray(L["b"], dtype=np.float32))
                for L in dig["layers"]
            ]

    def nearest_exemplar(self, label: str, hand63: np.ndarray) -> Optional[np.ndarray]:
        bank = self.exemplars.get(label)
        if bank is None or not len(bank):
            return None
        live = np.asarray(hand63, dtype=np.float32).reshape(-1)[:63]
        return bank[int(np.argmin(np.linalg.norm(bank - live, axis=1)))]

    @property
    def ready(self) -> bool:
        return bool(self.layers)

    def proba(self, hand63: np.ndarray) -> np.ndarray:
        return self._forward(hand63, self.layers, self.input_dim)

    def _forward(self, hand63: np.ndarray, layers, input_dim: int) -> np.ndarray:
        x = np.asarray(hand63, dtype=np.float32).reshape(1, -1)[:, :63]
        if input_dim > 63:
            x = model_features(x)
        for i, (W, b) in enumerate(layers):
            x = x @ W + b
            if i < len(layers) - 1:
                x = np.maximum(x, 0.0)
        x = x[0] - x[0].max()
        e = np.exp(x)
        return e / e.sum()

    def digit_proba(self, hand63: np.ndarray) -> np.ndarray:
        return self._forward(hand63, self.digit_layers, self.digit_input_dim)

    def predict(self, hand63: np.ndarray) -> Tuple[str, float]:
        p = self.proba(hand63)
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])

    def target_label(self, target: str) -> Optional[str]:
        """Map a lesson target (letter or phrase key) to a model label, if scorable."""
        t = str(target).strip()
        if len(t) == 1 and t.isdigit() and t in self.digit_labels:
            return t
        if len(t) == 1 and t.upper() in self.labels:
            return t.upper()
        return PHRASE_HANDSHAPES.get(t.lower())
