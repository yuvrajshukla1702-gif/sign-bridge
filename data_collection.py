"""
data_collection.py — Record custom keypoint sequence datasets via webcam.

Usage:
  python data_collection.py --label thank_you --samples 30
  python data_collection.py --label water --samples 30 --camera 0

Saves each clip as:
  data/raw/<label>/sample_XXXX.npy   shape (frames, feature_dim)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.tracking.landmarks import FEATURE_DIM, LandmarkExtractor  # noqa: E402


def next_sample_index(label_dir: Path) -> int:
    existing = sorted(label_dir.glob("sample_*.npy"))
    if not existing:
        return 0
    last = existing[-1].stem  # sample_0012
    try:
        return int(last.split("_")[1]) + 1
    except (IndexError, ValueError):
        return len(existing)


def collect(
    label: str,
    samples: int,
    camera: int,
    countdown: float,
    min_frames: int,
    max_frames: int,
) -> None:
    label_dir = ROOT / "data" / "raw" / label
    label_dir.mkdir(parents=True, exist_ok=True)
    idx = next_sample_index(label_dir)

    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera}")

    print(f"Collecting {samples} samples for label='{label}'")
    print("Keys: [SPACE]=start recording  [Q]=quit  [N]=skip current sample")

    collected = 0
    recording = False
    frames: list[np.ndarray] = []

    try:
        while collected < samples:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            features, results = extractor.process(frame)
            vis = extractor.draw_landmarks(frame, results)

            status = "RECORDING" if recording else "READY (press SPACE)"
            color = (0, 0, 255) if recording else (0, 200, 0)
            cv2.putText(
                vis,
                f"{label} | {status} | {collected}/{samples} | frames={len(frames)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
            cv2.imshow("Sign Data Collection", vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("n") and recording:
                recording = False
                frames = []
                print("Skipped sample.")
            if key == ord(" ") and not recording:
                # Countdown
                for t in range(int(countdown), 0, -1):
                    ok2, f2 = cap.read()
                    if not ok2:
                        break
                    f2 = cv2.flip(f2, 1)
                    cv2.putText(
                        f2,
                        f"Starting in {t}",
                        (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 255),
                        3,
                    )
                    cv2.imshow("Sign Data Collection", f2)
                    cv2.waitKey(1000)
                recording = True
                frames = []
                print("Recording… perform the sign.")

            if recording:
                frames.append(features.copy())
                # Auto-stop when max_frames reached OR user presses SPACE again
                if len(frames) >= max_frames or key == ord(" "):
                    if len(frames) < min_frames:
                        print(f"Too short ({len(frames)} < {min_frames}), discarded.")
                    else:
                        arr = np.stack(frames, axis=0).astype(np.float32)
                        out = label_dir / f"sample_{idx:04d}.npy"
                        np.save(out, arr)
                        print(f"Saved {out} shape={arr.shape}")
                        idx += 1
                        collected += 1
                    recording = False
                    frames = []
                    time.sleep(0.3)
    finally:
        extractor.close()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Collected {collected} samples → {label_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record sign keypoint sequences")
    p.add_argument("--label", required=True, help="Class name, e.g. thank_you")
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--countdown", type=float, default=2.0)
    p.add_argument("--min-frames", type=int, default=15)
    p.add_argument("--max-frames", type=int, default=45)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        label=args.label,
        samples=args.samples,
        camera=args.camera,
        countdown=args.countdown,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
