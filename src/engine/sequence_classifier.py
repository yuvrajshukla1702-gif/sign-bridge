"""
Dynamic sequence classifier wrapper around the LSTM SignPredictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.tracking.constants import FEATURE_DIM

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "sign_lstm.pt"
LABELS_PATH = ROOT / "models" / "labels.json"


class SequenceClassifier:
    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        labels_path: Path | str = LABELS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.labels: List[str] = []
        self._predictor = None
        self.load_error: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        if self.labels_path.exists():
            self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))
        else:
            self.labels = []

        if not self.model_path.exists():
            self._predictor = None
            self.load_error = "No LSTM model — train with train_model.py"
            return
        try:
            from src.model.lstm_model import SignPredictor

            if not self.labels:
                import torch

                ckpt = torch.load(self.model_path, map_location="cpu")
                self.labels = ckpt.get("labels", [])
            self._predictor = SignPredictor(self.model_path, self.labels)
            self.load_error = None
        except Exception as exc:  # noqa: BLE001
            self._predictor = None
            self.load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self._predictor is not None

    def predict(self, window: np.ndarray) -> Tuple[str, float]:
        if self._predictor is None:
            return "—", 0.0
        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("window must be (T, F)")
        if arr.shape[1] < FEATURE_DIM:
            pad = np.zeros((arr.shape[0], FEATURE_DIM - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > FEATURE_DIM:
            arr = arr[:, :FEATURE_DIM]
        return self._predictor.predict(arr)
