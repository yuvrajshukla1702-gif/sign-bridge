"""
Module A — Landmark extraction & normalization.

Uses MediaPipe Holistic (hands + upper-body pose) when available,
falls back to Hands-only if Holistic is unavailable.
Coordinates are wrist-relative and scale-normalized for distance invariance.
"""

from __future__ import annotations

from src.tracking.constants import (
    COORDS,
    FEATURE_DIM,
    FEATURE_DIM_HANDS,
    FEATURE_DIM_POSE,
    HAND_LANDMARKS,
    POSE_SELECTED,
)

# Re-export for older imports
__all__ = [
    "COORDS",
    "FEATURE_DIM",
    "FEATURE_DIM_HANDS",
    "FEATURE_DIM_POSE",
    "HAND_LANDMARKS",
    "POSE_SELECTED",
    "LandmarkExtractor",
]

# Lazy-ish: keep LandmarkExtractor in this module but delay mediapipe until class init
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class LandmarkExtractor:
    """Extract and normalize hand + pose keypoints from a BGR frame."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_pose: bool = True,
    ) -> None:
        self.use_pose = use_pose
        self.mp_holistic = mp.solutions.holistic
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self._holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )
        self._hands = None  # lazy fallback if needed

    def close(self) -> None:
        self._holistic.close()
        if self._hands is not None:
            self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        Returns:
            features: float32 vector of length FEATURE_DIM (zeros if no hands)
            results: MediaPipe Holistic results (for overlay drawing)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._holistic.process(rgb)
        rgb.flags.writeable = True

        left = self._landmarks_to_array(results.left_hand_landmarks)
        right = self._landmarks_to_array(results.right_hand_landmarks)
        pose = (
            self._pose_to_array(results.pose_landmarks)
            if self.use_pose
            else np.zeros(FEATURE_DIM_POSE, dtype=np.float32)
        )

        # Prefer normalizing each hand relative to its own wrist (landmark 0)
        left_n = self._normalize_hand(left) if left is not None else np.zeros(HAND_LANDMARKS * COORDS)
        right_n = self._normalize_hand(right) if right is not None else np.zeros(HAND_LANDMARKS * COORDS)
        pose_n = self._normalize_pose(pose) if results.pose_landmarks else pose

        features = np.concatenate([left_n, right_n, pose_n]).astype(np.float32)
        return features, results

    def draw_landmarks(self, frame_bgr: np.ndarray, results) -> np.ndarray:
        """Draw hand + pose landmarks on a copy of the frame."""
        out = frame_bgr.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                out,
                results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style(),
            )
        for hand_lms in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand_lms:
                self.mp_drawing.draw_landmarks(
                    out,
                    hand_lms,
                    self.mp_holistic.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return out

    @staticmethod
    def _landmarks_to_array(hand_landmarks) -> Optional[np.ndarray]:
        if hand_landmarks is None:
            return None
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        return pts  # (21, 3)

    def _pose_to_array(self, pose_landmarks) -> np.ndarray:
        if pose_landmarks is None:
            return np.zeros(FEATURE_DIM_POSE, dtype=np.float32)
        pts = []
        for idx in POSE_SELECTED:
            lm = pose_landmarks.landmark[idx]
            pts.extend([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _normalize_hand(pts: np.ndarray) -> np.ndarray:
        """Wrist-relative + bbox scale normalization → flat (63,) vector."""
        wrist = pts[0].copy()
        centered = pts - wrist
        # Scale by max extent in x/y for distance invariance
        scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
        if scale < 1e-6:
            scale = 1.0
        centered /= scale
        return centered.flatten()

    @staticmethod
    def _normalize_pose(pose_flat: np.ndarray) -> np.ndarray:
        """Center pose on midpoint of shoulders (indices 1,2 in POSE_SELECTED)."""
        if pose_flat.shape[0] < FEATURE_DIM_POSE or np.allclose(pose_flat, 0):
            return pose_flat
        pts = pose_flat.reshape(-1, 3)
        # shoulders are at local indices 1 and 2
        mid = (pts[1] + pts[2]) / 2.0
        centered = pts - mid
        scale = np.max(np.linalg.norm(centered[:, :2], axis=1))
        if scale < 1e-6:
            scale = 1.0
        centered /= scale
        return centered.flatten().astype(np.float32)
