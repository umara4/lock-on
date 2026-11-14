"""Low-rate MediaPipe Pose observations used to verify crossed forearms."""

from __future__ import annotations

import cv2
import numpy as np

import config


class BodyPoseTracker:
    """Track upper-body landmarks on sampled frames, not on GUI refreshes."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe Pose is required for crossed-arm recognition."
            ) from error
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.GESTURE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.GESTURE_MIN_TRACKING_CONFIDENCE,
        )

    def process(
        self,
        frame: np.ndarray,
    ) -> tuple[tuple[float, float, float, float], ...]:
        height, width = frame.shape[:2]
        scale = min(
            1.0,
            config.GESTURE_MAX_PROCESSING_DIMENSION / max(height, width),
        )
        sampled = (
            cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else frame
        )
        result = self._pose.process(cv2.cvtColor(sampled, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            return ()
        return tuple(
            (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(landmark.visibility),
            )
            for landmark in result.pose_landmarks.landmark
        )

    def close(self) -> None:
        self._pose.close()
