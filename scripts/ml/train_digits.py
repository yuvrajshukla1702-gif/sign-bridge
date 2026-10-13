"""Train a 0–9 MLP on Vedant3907/Numbers-Sign-Language-Dataset and refresh number cue photos."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "ml"))

from extract_landmarks import build_detector, detect  # noqa: E402
from src.engine.hand_features import MODEL_FEATURE_VERSION, model_features, normalize_raw_hand  # noqa: E402
import train_static as T  # noqa: E402

PARQUET = ROOT / "data" / "external" / "asl_numbers" / "data" / "train-00000-of-00001.parquet"
OUT_NPZ = ROOT / "data" / "landmarks" / "asl_digits.npz"
OUT_MODEL = ROOT / "models" / "digit_mlp.json"
CUE_DIR = ROOT / "assets" / "signs_dict" / "numbers"


def save_cues(images, labels) -> None:
    CUE_DIR.mkdir(parents=True, exist_ok=True)
    best = {}
    for im, lab in zip(images, labels):
        a = np.asarray(im, dtype=np.float32)
        white = float(np.mean(np.all(a > 235, axis=2)))
        if white > 0.7:
            continue
        sat = float((a.max(2) - a.min(2)).mean())
        bright = float(a.mean())
        score = bright - abs(sat - 40) + min(im.size) * 0.02
        prev = best.get(lab)
        if prev is None or score > prev[0]:
            best[lab] = (score, im)
    for lab, (_, im) in best.items():
        canvas = Image.new("RGB", (480, 560), (244, 247, 251))
        pic = im.copy()
        scale = min(440 / pic.width, 500 / pic.height)
        pic = pic.resize((max(1, int(pic.width * scale)), max(1, int(pic.height * scale))), Image.Resampling.LANCZOS)
        canvas.paste(pic, ((480 - pic.width) // 2, (560 - pic.height) // 2))
        canvas.save(CUE_DIR / f"{lab}.png")
    print("cues", sorted(best))


def load_folder_digits():
    root = ROOT / "data" / "external" / "sld_digits"
    folders = [p for p in root.rglob("*") if p.is_dir() and p.name.isdigit()]
    opened, labels = [], []
    for folder in folders:
        for img_path in sorted(folder.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            opened.append(Image.open(img_path).convert("RGB"))
            labels.append(folder.name)
    return opened, labels


def main() -> None:
    opened, labels = load_folder_digits()
    if not opened:
        table = pq.read_table(PARQUET)
        images = table.column("image").to_pylist()
        labels = [str(int(x)) for x in table.column("label").to_pylist()]
        opened = [Image.open(io.BytesIO(im["bytes"])).convert("RGB") for im in images]
    print("images", len(opened), "classes", sorted(set(labels)))
    save_cues(opened, labels)

    mp, detector = build_detector()
    pts, wh, y = [], [], []
    for i, (img, lab) in enumerate(zip(opened, labels)):
        hit = detect(mp, detector, img)
        if hit is None:
            continue
        p, w, h = hit
        pts.append(p)
        wh.append((w, h))
        y.append(lab)
        if i and i % 1000 == 0:
            print(f"  {i} images, {len(y)} hands", flush=True)
    pts = np.stack(pts)
    wh = np.asarray(wh, dtype=np.float32)
    y = np.asarray(y)
    np.savez_compressed(OUT_NPZ, pts=pts, wh=wh, y=y)
    print(f"hands {len(y)} / {len(opened)}")

    aspect = wh[:, 0] / wh[:, 1]
    X = model_features(np.stack([normalize_raw_hand(p, a) for p, a in zip(pts, aspect)]))
    labels_sorted = sorted(set(y.tolist()), key=int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)
    rng = np.random.default_rng(7)
    # light augmentation via the letter trainer's raw-point augment is overkill; jitter features
    noise = rng.normal(0, 0.01, size=(3, *Xtr.shape)).astype(np.float32)
    Xaug = np.concatenate([Xtr, Xtr + noise[0], Xtr + noise[1], Xtr + noise[2]])
    yaug = np.concatenate([ytr, ytr, ytr, ytr])
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=12,
        random_state=7,
    )
    t0 = time.time()
    clf.fit(Xaug, yaug)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=labels_sorted, average="macro", zero_division=0)
    print(f"digits acc {100*acc:.1f} prec {100*p:.1f} rec {100*r:.1f} f1 {100*f:.1f} in {time.time()-t0:.0f}s n={len(yte)}")
    layers = [
        {"W": W.T.astype(np.float32).round(6).tolist(), "b": b.astype(np.float32).round(6).tolist()}
        for W, b in zip(clf.coefs_, clf.intercepts_)
    ]
    payload = {
        "type": "mlp",
        "feature": MODEL_FEATURE_VERSION,
        "input_dim": int(clf.coefs_[0].shape[0]),
        "labels": [str(c) for c in clf.classes_],
        "layers": layers,
        "metrics": {
            "accuracy": round(100 * acc, 2),
            "macro_precision": round(100 * p, 2),
            "macro_recall": round(100 * r, 2),
            "macro_f1": round(100 * f, 2),
            "n": int(len(yte)),
        },
        "source": "Vedant3907/Numbers-Sign-Language-Dataset",
    }
    OUT_MODEL.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print("wrote", OUT_MODEL, f"{OUT_MODEL.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
