"""MediaPipe hand landmark detection and geometric gesture classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

import config


class GestureType(str, Enum):
    NONE = "none"
    OTHER = "other"
    POINTING = "pointing"
    OPEN_PALM = "open palm"
    PINCH = "pinch"
    FIST = "closed fist"


@dataclass(frozen=True)
class DetectedHand:
    """One hand's landmark geometry and visual gesture classification."""

    landmarks: tuple[tuple[float, float, float], ...]
    handedness: str | None = None
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    tracking_confidence: float = 0.0


@dataclass(frozen=True)
class HandGesture:
    """Primary command-hand gesture plus all visible hands' landmark data."""

    gesture: GestureType = GestureType.NONE
    landmarks: tuple[tuple[float, float, float], ...] = ()
    handedness: str | None = None
    confidence: float = 0.0
    hands: tuple[DetectedHand, ...] = ()

    @property
    def detected(self) -> bool:
        return bool(self.landmarks)


class HandGestureRecognizer:
    """Classify the primary hand and expose both detected hands for rendering."""

    def __init__(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed. Install dependencies with "
                "'python -m pip install -r requirements.txt'."
            ) from error

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.GESTURE_MAX_HANDS,
            model_complexity=config.GESTURE_MODEL_COMPLEXITY,
            min_detection_confidence=config.GESTURE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.GESTURE_MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame: np.ndarray) -> HandGesture:
        """Return primary gesture and per-hand visual data for one BGR frame."""
        height, width = frame.shape[:2]
        scale = min(1.0, config.GESTURE_MAX_PROCESSING_DIMENSION / max(height, width))
        if scale < 1.0:
            resized = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = frame

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return HandGesture(GestureType.NONE)

        candidates: list[
            tuple[float, int, DetectedHand]
        ] = []
        for index, hand in enumerate(results.multi_hand_landmarks):
            landmarks = tuple(
                (float(point.x), float(point.y), float(point.z))
                for point in hand.landmark
            )
            xs = [point[0] for point in landmarks]
            ys = [point[1] for point in landmarks]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            handedness = None
            tracking_confidence = 0.0
            if results.multi_handedness and index < len(results.multi_handedness):
                classification = results.multi_handedness[index].classification[0]
                handedness = classification.label
                tracking_confidence = float(getattr(classification, "score", 0.0))
                if not config.GESTURE_INPUT_MIRRORED:
                    handedness = "Right" if handedness == "Left" else "Left"
            gesture, confidence = classify_landmarks(
                landmarks,
                aspect_ratio=width / height,
            )
            candidates.append(
                (
                    area,
                    index,
                    DetectedHand(
                        landmarks,
                        handedness,
                        gesture,
                        confidence,
                        tracking_confidence,
                    ),
                )
            )

        _, _, primary = max(candidates, key=lambda candidate: candidate[0])
        return HandGesture(
            primary.gesture,
            primary.landmarks,
            primary.handedness,
            primary.confidence,
            tuple(candidate[2] for candidate in candidates),
        )

    def close(self) -> None:
        self._hands.close()


def classify_landmarks(
    landmarks: tuple[tuple[float, float, float], ...]
    | list[tuple[float, float, float]],
    aspect_ratio: float = 1.0,
) -> tuple[GestureType, float]:
    """Classify 21 normalized landmarks using palm-relative distances."""
    if len(landmarks) != 21:
        return GestureType.NONE, 0.0

    def distance(first_index: int, second_index: int) -> float:
        return _distance(
            landmarks[first_index],
            landmarks[second_index],
            aspect_ratio,
        )

    palm_scale = max(
        distance(0, 9),
        distance(5, 17),
        1e-6,
    )
    index_tip_distance = distance(8, 0)
    index_pip_distance = distance(6, 0)
    pinch_ratio = distance(4, 8) / palm_scale
    if pinch_ratio <= config.PINCH_DISTANCE_RATIO:
        confidence = min(1.0, 0.75 + (config.PINCH_DISTANCE_RATIO - pinch_ratio))
        return GestureType.PINCH, confidence

    finger_pairs = ((8, 6), (12, 10), (16, 14), (20, 18))
    extended = [
        distance(tip, 0)
        > distance(pip, 0) * config.FINGER_EXTENSION_RATIO
        for tip, pip in finger_pairs
    ]
    folded = [
        distance(tip, 0)
        < distance(pip, 0) * config.FINGER_FOLDED_RATIO
        for tip, pip in finger_pairs
    ]

    thumb_extended = (
        distance(4, 0)
        > distance(3, 0) * config.FINGER_EXTENSION_RATIO
    )
    thumb_spread = (
        distance(4, 5) / palm_scale
        >= config.THUMB_SPREAD_RATIO
    )

    if all(extended) and (thumb_extended or thumb_spread):
        return GestureType.OPEN_PALM, 0.9
    if extended[0] and all(folded[1:]):
        return GestureType.POINTING, 0.9
    if all(folded):
        return GestureType.FIST, 0.85
    if index_tip_distance > index_pip_distance:
        return GestureType.OTHER, 0.5
    return GestureType.OTHER, 0.4


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    aspect_ratio: float = 1.0,
) -> float:
    delta_x = (first[0] - second[0]) * aspect_ratio
    delta_y = first[1] - second[1]
    return float(np.hypot(delta_x, delta_y))


def draw_hand_landmarks(frame: np.ndarray, hand: HandGesture) -> None:
    """Overlay the primary hand skeleton and a fingertip pointing ray."""
    if not hand.detected:
        return
    height, width = frame.shape[:2]
    points = [
        (int(x * width), int(y * height))
        for x, y, _ in hand.landmarks
    ]
    for start, end in (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
    ):
        cv2.line(frame, points[start], points[end], (90, 180, 130), 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(
            frame,
            point,
            config.HAND_LANDMARK_RADIUS,
            (30, 240, 230),
            cv2.FILLED,
            cv2.LINE_AA,
        )
    if hand.gesture == GestureType.POINTING:
        index_tip = points[8]
        index_pip = points[6]
        dx = index_tip[0] - index_pip[0]
        dy = index_tip[1] - index_pip[1]
        magnitude = max(float(np.hypot(dx, dy)), 1.0)
        ray_length = int(np.hypot(width, height) * config.POINTING_RAY_LENGTH_RATIO)
        endpoint = (
            int(index_tip[0] + dx / magnitude * ray_length),
            int(index_tip[1] + dy / magnitude * ray_length),
        )
        cv2.arrowedLine(frame, index_tip, endpoint, (0, 220, 255), 2, cv2.LINE_AA, tipLength=0.04)
