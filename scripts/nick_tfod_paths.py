"""
Nick Renotte RealTimeObjectDetection — path setup (Tutorial.ipynb cell 0).
ASL sign labels for the YouTube SSD tutorial (Hello / Yes / No / Thank You / I Love You).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WORKSPACE_PATH = ROOT / "Tensorflow" / "workspace"
SCRIPTS_PATH = ROOT / "Tensorflow" / "scripts"
APIMODEL_PATH = ROOT / "Tensorflow" / "models"
ANNOTATION_PATH = WORKSPACE_PATH / "annotations"
IMAGE_PATH = WORKSPACE_PATH / "images"
MODEL_PATH = WORKSPACE_PATH / "models"
PRETRAINED_MODEL_PATH = WORKSPACE_PATH / "pre-trained-models"
CUSTOM_MODEL_NAME = "my_ssd_mobnet"
CONFIG_PATH = MODEL_PATH / CUSTOM_MODEL_NAME / "pipeline.config"
CHECKPOINT_PATH = MODEL_PATH / CUSTOM_MODEL_NAME

# Same signs Nick collects in https://youtu.be/pDXdlXlaCco
ASL_LABELS = [
    {"name": "Hello", "id": 1},
    {"name": "Yes", "id": 2},
    {"name": "No", "id": 3},
    {"name": "Thank You", "id": 4},
    {"name": "I Love You", "id": 5},
]
