"""
8. Detect in Real-Time — Nick Tutorial.ipynb (TFOD SSD).

Requires a trained checkpoint under:
  Tensorflow/workspace/models/my_ssd_mobnet/ckpt-*

Install TFOD API first:
  https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

Usage:
  python scripts/nick_detect_realtime.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nick_tfod_paths import (  # noqa: E402
    ANNOTATION_PATH,
    CHECKPOINT_PATH,
    CONFIG_PATH,
)


def main() -> None:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"Missing {CONFIG_PATH}\n"
            "Follow Tutorial.ipynb steps 3–6 (download pretrained, copy config, train)."
        )

    import tensorflow as tf
    from object_detection.builders import model_builder
    from object_detection.utils import config_util, label_map_util
    from object_detection.utils import visualization_utils as viz_utils

    label_map = ANNOTATION_PATH / "label_map.pbtxt"
    if not label_map.exists():
        raise SystemExit("Run: python scripts/nick_create_labelmap.py")

    configs = config_util.get_configs_from_pipeline_file(str(CONFIG_PATH))
    detection_model = model_builder.build(model_config=configs["model"], is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    # Prefer newest ckpt-* in folder (Nick uses ckpt-6 in the notebook)
    ckpts = sorted(CHECKPOINT_PATH.glob("ckpt-*.index"))
    if not ckpts:
        raise SystemExit(f"No checkpoint in {CHECKPOINT_PATH} — train first (Tutorial step 6).")
    ckpt_path = str(ckpts[-1]).replace(".index", "")
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Loaded {ckpt_path}")

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(str(label_map))
    cap = cv2.VideoCapture(0)
    print("Q = quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {k: v[0, :num_detections].numpy() for k, v in detections.items()}
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + 1,
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=0.5,
            agnostic_mode=False,
        )
        cv2.imshow("Nick TFOD detect", cv2.resize(image_np_with_detections, (800, 600)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
