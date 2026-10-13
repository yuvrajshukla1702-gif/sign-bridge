"""
Train the A–Z static-letter classifier on real MediaPipe landmarks.

Input : data/landmarks/asl_letters.npz  (from extract_landmarks.py)
Output: models/static_mlp.json          (weights for backend + browser)
        docs/STATIC_MODEL_REPORT.json   (metrics)

Evaluation (all on images the model never saw during training):
  1. group split   — 80/20 split by original image group (no near-duplicate leakage)
  2. cross-dataset — train on one dataset, test on the other (hardest, most honest)

Usage:
  python scripts/ml/train_static.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.engine.hand_features import (  # noqa: E402
    FEATURE_VERSION,
    MODEL_FEATURE_VERSION,
    model_features,
    normalize_raw_hand,
)

DATA = ROOT / "data" / "landmarks" / "asl_letters.npz"
OUT_MODEL = ROOT / "models" / "static_mlp.json"
OUT_REPORT = ROOT / "docs" / "STATIC_MODEL_REPORT.json"
SEED = 7


def load():
    d = np.load(DATA, allow_pickle=False)
    pts, wh, y, src, grp = d["pts"], d["wh"], d["y"], d["source"], d["group"]
    aspect = wh[:, 0] / wh[:, 1]
    return pts, aspect, y, src, grp


def featurize(pts, aspect):
    return np.stack([normalize_raw_hand(p, a) for p, a in zip(pts, aspect)])


def rotation_matrices(n: int, rng: np.random.Generator, yaw=35.0, pitch=25.0, roll=18.0) -> np.ndarray:
    a, b, c = (np.deg2rad(rng.uniform(-v, v, n)) for v in (yaw, pitch, roll))
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    one, zero = np.ones(n), np.zeros(n)
    Ry = np.stack([ca, zero, sa, zero, one, zero, -sa, zero, ca], 1).reshape(n, 3, 3)
    Rx = np.stack([one, zero, zero, zero, cb, -sb, zero, sb, cb], 1).reshape(n, 3, 3)
    Rz = np.stack([cc, -sc, zero, sc, cc, zero, zero, zero, one], 1).reshape(n, 3, 3)
    return Rz @ Rx @ Ry


def augment(pts, aspect, y, copies: int, rng: np.random.Generator):
    """Mirror (either hand), 3D rotation (camera viewpoint), aspect/depth jitter, landmark noise."""
    xs, ys = [featurize(pts, aspect)], [y]
    for _ in range(copies):
        p = pts.copy()
        a = aspect * rng.uniform(0.8, 1.25, size=len(p))
        mirror = rng.random(len(p)) < 0.5
        p[mirror, :, 0] = 1.0 - p[mirror, :, 0]
        feats = featurize(p, a).reshape(-1, 21, 3)
        feats[:, :, 2] *= rng.uniform(0.7, 1.3, size=(len(p), 1))
        feats = np.einsum("nij,nkj->nki", rotation_matrices(len(p), rng), feats)
        feats += rng.normal(0, 0.012, size=feats.shape)
        # re-scale to unit max radius so train matches inference normalization
        r = np.max(np.linalg.norm(feats[:, :, :2], axis=2), axis=1)[:, None, None]
        feats /= np.maximum(r, 1e-6)
        xs.append(feats.reshape(len(p), -1).astype(np.float32))
        ys.append(y)
    return model_features(np.concatenate(xs)), np.concatenate(ys)


def test_features(pts, aspect):
    return model_features(featurize(pts, aspect))


def mirrored_test(pts, aspect):
    p = pts.copy()
    p[:, :, 0] = 1.0 - p[:, :, 0]
    return test_features(p, aspect)


def metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": round(100 * acc, 2), "macro_precision": round(100 * p, 2),
            "macro_recall": round(100 * r, 2), "macro_f1": round(100 * f, 2), "n": int(len(y_true))}


def per_class(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {lab: {"precision": round(100 * p[i], 1), "recall": round(100 * r[i], 1),
                  "f1": round(100 * f[i], 1), "support": int(s[i])} for i, lab in enumerate(labels)}


def top_confusions(y_true, y_pred, labels, k=8):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.fill_diagonal(cm, 0)
    flat = [(int(cm[i, j]), labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if cm[i, j]]
    return [{"true": t, "pred": p, "count": c} for c, t, p in sorted(flat, reverse=True)[:k]]


def make_mlp(hidden=(256, 128)):
    return MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-4,
                         batch_size=256, learning_rate_init=1e-3, max_iter=200,
                         early_stopping=True, n_iter_no_change=12, random_state=SEED)


MLP_SIZES = {"mlp": (256, 128), "mlp_large": (512, 256)}


def candidates():
    return {
        "mlp": make_mlp(MLP_SIZES["mlp"]),
        "mlp_large": make_mlp(MLP_SIZES["mlp_large"]),
        "svm_rbf": SVC(C=10, gamma="scale", probability=False, random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED),
        "logreg": LogisticRegression(max_iter=3000, C=5.0),
    }


def export_mlp(clf: MLPClassifier, labels, report):
    layers = [{"W": W.T.astype(np.float32).round(6).tolist(), "b": b.astype(np.float32).round(6).tolist()}
              for W, b in zip(clf.coefs_, clf.intercepts_)]
    payload = {
        "type": "mlp",
        "feature": MODEL_FEATURE_VERSION,
        "input_dim": int(clf.coefs_[0].shape[0]),
        "activation": "relu",
        "output": "softmax",
        "labels": [str(c) for c in clf.classes_],
        "layers": layers,
        "metrics": report,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": ["Marxulia/asl_sign_languages_alphabets_v03", "atalaydenknalbant/asl-dataset"],
    }
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {OUT_MODEL} ({OUT_MODEL.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    import argparse

    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()
    DATA = Path(args.data)

    rng = np.random.default_rng(SEED)
    pts, aspect, y, src, grp = load()
    labels = sorted(set(y.tolist()))
    print(f"{len(y)} samples, {len(labels)} classes, sources: "
          + ", ".join(f"{s}={int((src == s).sum())}" for s in sorted(set(src.tolist()))))

    report = {"feature": FEATURE_VERSION, "n_samples": int(len(y)), "labels": labels,
              "class_counts": {lab: int((y == lab).sum()) for lab in labels}}

    # ---- 1. group split, model comparison ----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr, te = next(gss.split(pts, y, grp))
    Xtr, ytr = augment(pts[tr], aspect[tr], y[tr], copies=3, rng=rng)
    Xte, yte = test_features(pts[te], aspect[te]), y[te]
    Xte_m = mirrored_test(pts[te], aspect[te])

    comparison = {}
    fitted = {}
    for name, clf in candidates().items():
        t0 = time.time()
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        pred_m = clf.predict(Xte_m)
        comparison[name] = {**metrics(yte, pred, labels),
                            "mirrored_hand_accuracy": round(100 * accuracy_score(yte, pred_m), 2),
                            "train_seconds": round(time.time() - t0, 1)}
        fitted[name] = clf
        print(f"  {name:14s} {comparison[name]}", flush=True)
    report["group_split_comparison"] = comparison

    # ---- 2. cross-dataset generalisation, picks the MLP size ----
    cross = {}
    for size_name, hidden in MLP_SIZES.items():
        cross[size_name] = {}
        for train_src in sorted(set(src.tolist())):
            test_mask = src != train_src
            train_mask = src == train_src
            if test_mask.sum() == 0:
                continue
            Xa, ya = augment(pts[train_mask], aspect[train_mask], y[train_mask], copies=3, rng=rng)
            clf = make_mlp(hidden).fit(Xa, ya)
            te_labels = sorted(set(y[test_mask].tolist()) & set(ya.tolist()))
            m = np.isin(y, te_labels) & test_mask
            pr = clf.predict(test_features(pts[m], aspect[m]))
            cross[size_name][f"train_{train_src}_test_others"] = metrics(y[m], pr, te_labels)
            print(f"  cross {size_name} {train_src}: {cross[size_name][f'train_{train_src}_test_others']}", flush=True)
    report["cross_dataset"] = cross
    best = max(cross, key=lambda k: np.mean([v["accuracy"] for v in cross[k].values()]) if cross[k] else 0)
    if not cross[best]:
        best = max(MLP_SIZES, key=lambda k: comparison[k]["macro_f1"])
    report["chosen_mlp"] = best
    print(f"  chosen: {best} {MLP_SIZES[best]}")

    pred = fitted[best].predict(Xte)
    report["group_split_mlp_per_class"] = per_class(yte, pred, labels)
    report["group_split_mlp_top_confusions"] = top_confusions(yte, pred, labels)

    if args.no_export:
        return

    # ---- 3. final model on everything ----
    Xall, yall = augment(pts, aspect, y, copies=4, rng=rng)
    final = make_mlp(MLP_SIZES[best]).fit(Xall, yall)
    report["final_train_samples_augmented"] = int(len(yall))
    export_mlp(final, labels, {"chosen_mlp": best, "group_split": comparison[best],
                               "cross_dataset": cross[best]})

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
