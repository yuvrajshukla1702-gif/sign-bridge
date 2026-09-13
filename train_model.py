"""
train_model.py — Train LSTM sequence classifier on collected .npy clips.

Usage:
  python train_model.py
  python train_model.py --epochs 40 --batch-size 16 --window 30

Expects:
  data/raw/<label>/sample_XXXX.npy   each (T, feature_dim)

Writes:
  models/sign_lstm.pt
  models/labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.model.lstm_model import SignLSTM  # noqa: E402
from src.tracking.landmarks import FEATURE_DIM  # noqa: E402


def pad_or_sample(seq: np.ndarray, window: int) -> np.ndarray:
    """Resize variable-length clip to fixed `window` frames."""
    t = seq.shape[0]
    if t == window:
        return seq
    if t > window:
        # Uniform sample indices
        idx = np.linspace(0, t - 1, window).astype(int)
        return seq[idx]
    # Pad by repeating last frame
    pad = np.repeat(seq[-1:], window - t, axis=0)
    return np.concatenate([seq, pad], axis=0)


class SignDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def load_raw(raw_dir: Path, window: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])
    if not labels:
        raise FileNotFoundError(
            f"No label folders under {raw_dir}. Run data_collection.py first."
        )
    xs, ys = [], []
    for yi, label in enumerate(labels):
        for path in sorted((raw_dir / label).glob("*.npy")):
            arr = np.load(path).astype(np.float32)
            if arr.ndim != 2:
                continue
            # Pad / trim feature dim if older samples differ
            if arr.shape[1] < FEATURE_DIM:
                pad = np.zeros((arr.shape[0], FEATURE_DIM - arr.shape[1]), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=1)
            elif arr.shape[1] > FEATURE_DIM:
                arr = arr[:, :FEATURE_DIM]
            xs.append(pad_or_sample(arr, window))
            ys.append(yi)
    X = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)
    return X, y, labels


def train(args: argparse.Namespace) -> None:
    raw_dir = ROOT / "data" / "raw"
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    X, y, labels = load_raw(raw_dir, args.window)
    print(f"Loaded {len(y)} samples | classes={labels} | X={X.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    train_loader = DataLoader(
        SignDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(SignDataset(X_val, y_val), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignLSTM(
        input_size=FEATURE_DIM,
        hidden_size=args.hidden,
        num_layers=args.layers,
        num_classes=len(labels),
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = models_dir / "sign_lstm.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        avg_loss = total_loss / max(len(y_train), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_acc={acc:.3f}")

        if acc >= best_acc:
            best_acc = acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": FEATURE_DIM,
                    "hidden_size": args.hidden,
                    "num_layers": args.layers,
                    "labels": labels,
                    "window": args.window,
                },
                out_path,
            )

    labels_path = models_dir / "labels.json"
    labels_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(f"Best val_acc={best_acc:.3f}")
    print(f"Saved model → {out_path}")
    print(f"Saved labels → {labels_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train sign LSTM")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
