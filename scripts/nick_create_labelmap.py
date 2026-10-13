"""
1. Create Label Map — Nick Tutorial.ipynb (ASL signs for the SSD video).
Writes Tensorflow/workspace/annotations/label_map.pbtxt
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nick_tfod_paths import ANNOTATION_PATH, ASL_LABELS  # noqa: E402


def main() -> None:
    ANNOTATION_PATH.mkdir(parents=True, exist_ok=True)
    out = ANNOTATION_PATH / "label_map.pbtxt"
    with out.open("w", encoding="utf-8") as f:
        for label in ASL_LABELS:
            f.write("item { \n")
            f.write("\tname:'{}'\n".format(label["name"]))
            f.write("\tid:{}\n".format(label["id"]))
            f.write("}\n")
    print(f"Wrote {out}")
    for lab in ASL_LABELS:
        print(f"  {lab['id']}: {lab['name']}")


if __name__ == "__main__":
    main()
