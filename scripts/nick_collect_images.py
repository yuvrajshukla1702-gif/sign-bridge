"""
Collect webcam images for LabelImg — Nick Renotte TFOD / RealTimeObjectDetection flow.

Saves under:
  Tensorflow/workspace/images/collected/<label>/

Then:
  1) Split into images/train and images/test
  2) Label with LabelImg (https://github.com/HumanSignal/labelImg)
  3) python scripts/nick_create_labelmap.py
  4) generate_tfrecord + train (Tutorial.ipynb)

Usage:
  python scripts/nick_collect_images.py
  python scripts/nick_collect_images.py --labels Hello Yes No --per-class 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ["Hello", "Yes", "No", "Thank You", "I Love You"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out_root = ROOT / "Tensorflow" / "workspace" / "images" / "collected"
    out_root.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Camera failed to open")

    print("SPACE = capture | N = next label | Q = quit")
    for label in args.labels:
        folder = out_root / label.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)
        count = len(list(folder.glob("*.jpg")))
        print(f"\n=== Collecting: {label} (have {count}, need {args.per_class}) ===")
        while count < args.per_class:
            ok, frame = cap.read()
            if not ok:
                break
            disp = frame.copy()
            cv2.putText(
                disp,
                f"{label}  {count}/{args.per_class}  [SPACE capture]",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Nick TFOD collect", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key == ord("n"):
                break
            if key == ord(" "):
                path = folder / f"{label.replace(' ', '_').lower()}_{count:03d}.jpg"
                cv2.imwrite(str(path), frame)
                count += 1
                print(f"  saved {path.name}")
                time.sleep(0.15)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Images in {out_root}")
    print("Next: label with LabelImg, put XML+JPG into images/train and images/test")


if __name__ == "__main__":
    main()
