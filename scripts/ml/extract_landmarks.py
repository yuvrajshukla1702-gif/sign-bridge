"""
Run MediaPipe HandLandmarker over the downloaded ASL image datasets and save
raw 21-point landmarks for training.

Output: data/landmarks/asl_letters.npz
  pts     (N, 21, 3) float32  raw MediaPipe coords (x, y in [0,1] of image, z relative)
  wh      (N, 2)     float32  image width/height (for aspect correction)
  y       (N,)       <U1      letter label A–Z
  source  (N,)       <U16     dataset name
  group   (N,)       <U64     group id (original file stem) for leak-free splits

Usage:
  python scripts/ml/extract_landmarks.py
"""

from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = ROOT / "data" / "landmarks" / "asl_letters.npz"
TASK = ROOT / "models" / "hand_landmarker.task"
LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


def build_detector(min_conf: float = 0.3):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
    )
    return mp, vision.HandLandmarker.create_from_options(opts)


def detect(mp, detector, img: Image.Image):
    """Return (pts(21,3), w, h) or None. Upscales small images for the detector."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) < 320:
        s = 320 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
        w, h = img.size
    arr = np.asarray(img, dtype=np.uint8)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.hand_landmarks:
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]], dtype=np.float32)
    return pts, w, h


def iter_marxulia():
    import pyarrow.parquet as pq

    files = sorted((EXT / "marxulia_v03").rglob("*.parquet"))
    for f in files:
        table = pq.read_table(f)
        images = table.column("image").to_pylist()
        labels = table.column("label").to_pylist()
        for i, (im, lab) in enumerate(zip(images, labels)):
            path = (im or {}).get("path") or f"{f.stem}_{i}"
            yield Image.open(io.BytesIO(im["bytes"])), LETTERS[int(lab)], "marxulia", Path(path).stem


def iter_atalay():
    # data.yaml lists 24 names (no J/Z) but label ids are A–Z positions
    # (verified against "hand1_y_..." filenames: y=24, k=10).
    names = [chr(ord("A") + i) for i in range(26)]
    root = EXT / "atalay_asl"
    for img_path in sorted(root.rglob("images/*.jpg")):
        lbl = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        cls = int(lines[0][0])
        letter = names[cls]
        if letter not in LETTERS:
            continue
        # Roboflow names look like "A22_jpg.rf.<hash>" / "A_mp4-9_jpg.rf.<hash>" — group by original stem
        group = re.split(r"_jpg\.rf\.|\.rf\.", img_path.stem)[0]
        img = Image.open(img_path)
        box = [float(v) for v in lines[0][1:5]]
        yield (img, box), letter, "atalay", group


def crop_box(img: Image.Image, box, pad: float = 1.6) -> Image.Image:
    w, h = img.size
    cx, cy, bw, bh = box[0] * w, box[1] * h, box[2] * w * pad, box[3] * h * pad
    side = max(bw, bh)
    l, t = max(0, cx - side / 2), max(0, cy - side / 2)
    r, b = min(w, cx + side / 2), min(h, cy + side / 2)
    return img.crop((int(l), int(t), int(r), int(b)))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["marxulia", "atalay"])
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)

    if not TASK.exists():
        sys.exit("Missing models/hand_landmarker.task — run scripts/ml/fetch_datasets.py")
    mp, det = build_detector()

    pts_l, wh_l, y_l, src_l, grp_l = [], [], [], [], []
    stats = {}

    def add(res, letter, src, group):
        pts, w, h = res
        pts_l.append(pts)
        wh_l.append((w, h))
        y_l.append(letter)
        src_l.append(src)
        grp_l.append(f"{src}:{group}")

    marx = iter_marxulia() if "marxulia" in args.sources else iter(())
    atal = iter_atalay() if "atalay" in args.sources else iter(())

    for n, (img, letter, src, group) in enumerate(marx):
        res = detect(mp, det, img)
        st = stats.setdefault(src, [0, 0])
        st[1] += 1
        if res is not None:
            st[0] += 1
            add(res, letter, src, group)
        if n % 1000 == 0:
            print(f"[{src}] {n} processed, {st[0]} hands", flush=True)

    for n, ((img, box), letter, src, group) in enumerate(atal):
        st = stats.setdefault(src, [0, 0])
        st[1] += 1
        # Try the full frame, then fall back to a padded crop around the labeled hand box
        res = detect(mp, det, img)
        if res is None:
            crop = crop_box(img, box)
            res = detect(mp, det, crop)
        if res is not None:
            st[0] += 1
            add(res, letter, src, group)
        if n % 1000 == 0:
            print(f"[{src}] {n} processed, {st[0]} hands", flush=True)

    det.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        pts=np.stack(pts_l).astype(np.float32),
        wh=np.asarray(wh_l, dtype=np.float32),
        y=np.asarray(y_l),
        source=np.asarray(src_l),
        group=np.asarray(grp_l),
    )
    for src, (ok, tot) in stats.items():
        print(f"{src}: {ok}/{tot} images with a detected hand ({100 * ok / max(1, tot):.1f}%)")
    print(f"Saved {len(y_l)} samples -> {out}")


if __name__ == "__main__":
    main()
