"""
Real-hand exemplars per letter for Learn Mode "nudge" hints.

K-means (per class, both hands via mirroring) over normalized real landmarks,
then keep the actual sample nearest each centroid. The coach compares the live
hand to the closest exemplar of the target letter, so hints point toward a pose
real signers actually make.

  python scripts/ml/build_exemplars.py [--data data/landmarks/asl_letters.npz] [--k 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.hand_features import FEATURE_VERSION, normalize_raw_hand  # noqa: E402

OUT = ROOT / "data" / "references" / "letter_exemplars.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "landmarks" / "asl_letters.npz"))
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X = np.stack([normalize_raw_hand(p, w / h) for p, (w, h) in zip(d["pts"], d["wh"])])
    y = np.array([str(v) for v in d["y"]])
    Xm = X.reshape(-1, 21, 3).copy()
    Xm[:, :, 0] *= -1
    X_all = np.concatenate([X, Xm.reshape(len(X), -1)])
    y_all = np.concatenate([y, y])

    out = {}
    for label in sorted(set(y_all)):
        Xi = X_all[y_all == label]
        k = min(args.k, len(Xi))
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Xi)
        picks = [int(np.argmin(np.linalg.norm(Xi - c, axis=1))) for c in km.cluster_centers_]
        out[label] = [np.round(Xi[i], 4).tolist() for i in picks]
        print(label, len(Xi), "->", k)

    OUT.write_text(
        json.dumps({"feature": FEATURE_VERSION, "k": args.k, "exemplars": out}),
        encoding="utf-8",
    )
    print("->", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
