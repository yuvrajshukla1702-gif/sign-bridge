"""
LSTM sequence classifier for dynamic sign gestures (Module B).
Input shape: (batch, seq_len=30, feature_dim≈147)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class SignLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 147,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # last timestep
        return self.head(last)


class SignPredictor:
    """Load a trained checkpoint and run softmax inference."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: List[str],
        device: Optional[str] = None,
    ) -> None:
        self.class_names = class_names
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=self.device)
        input_size = ckpt.get("input_size", 147)
        hidden_size = ckpt.get("hidden_size", 128)
        num_layers = ckpt.get("num_layers", 2)
        self.model = SignLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=len(class_names),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, window: np.ndarray) -> Tuple[str, float]:
        """
        window: (T, F) numpy array
        Returns (label, confidence)
        """
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return self.class_names[idx], float(probs[idx].item())
