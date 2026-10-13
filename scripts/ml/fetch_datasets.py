"""
Download real ASL alphabet image datasets + the MediaPipe hand landmark model.

Sources (Hugging Face):
  - Marxulia/asl_sign_languages_alphabets_v03   (~10.9k images, A–Z, parquet)
  - atalaydenknalbant/asl-dataset               (~6.3k images, many signers, YOLO labels)

Usage:
  python scripts/ml/fetch_datasets.py
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
MODELS = ROOT / "models"

DATASETS = {
    "marxulia_v03": "Marxulia/asl_sign_languages_alphabets_v03",
    "atalay_asl": "atalaydenknalbant/asl-dataset",
}

HAND_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def main() -> None:
    EXT.mkdir(parents=True, exist_ok=True)
    for name, repo in DATASETS.items():
        dest = EXT / name
        print(f"Downloading {repo} -> {dest}")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest), max_workers=8)

    MODELS.mkdir(parents=True, exist_ok=True)
    task = MODELS / "hand_landmarker.task"
    if not task.exists():
        print(f"Downloading {HAND_TASK_URL}")
        urllib.request.urlretrieve(HAND_TASK_URL, task)
    print("Done.")


if __name__ == "__main__":
    main()
