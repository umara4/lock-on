"""Lightweight geometry helpers for camera-space holographic overlays."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from vision.gestures import DetectedHand


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

FINGER_CONNECTIONS: dict[str, tuple[tuple[int, int], ...]] = {
    "thumb": ((1, 2), (2, 3), (3, 4)),
    "index": ((5, 6), (6, 7), (7, 8)),
    "middle": ((9, 10), (10, 11), (11, 12)),
    "ring": ((13, 14), (14, 15), (15, 16)),
    "little": ((17, 18), (18, 19), (19, 20)),
}


def hand_outline(
    landmarks: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """Return a normalized convex hull around landmarks, not a pixel mask."""
    if len(landmarks) != 21:
        return ()
    points = np.asarray([(x, y) for x, y, _ in landmarks], dtype=np.float32)
    if not np.isfinite(points).all():
        return ()
    hull = cv2.convexHull(points).reshape(-1, 2)
    return tuple((float(x), float(y)) for x, y in hull)


class HandLandmarkSmoother:
    """Smooth visible hand landmarks and discard missing or stale identities."""

    def __init__(
        self,
        smoothing: float = 0.62,
        maximum_gap_seconds: float = 0.25,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if maximum_gap_seconds <= 0.0:
            raise ValueError("maximum_gap_seconds must be positive")
        self._smoothing = smoothing
        self._maximum_gap_seconds = maximum_gap_seconds
        self._previous: dict[
            str,
            tuple[float, tuple[tuple[float, float, float], ...]],
        ] = {}

    def update(
        self,
        hands: tuple[DetectedHand, ...],
        timestamp: float,
    ) -> tuple[DetectedHand, ...]:
        if not hands:
            self._previous.clear()
            return ()

        smoothed_hands: list[DetectedHand] = []
        current: dict[
            str,
            tuple[float, tuple[tuple[float, float, float], ...]],
        ] = {}
        used_keys: set[str] = set()
        for index, hand in enumerate(hands):
            if len(hand.landmarks) != 21:
                continue
            base_key = hand.handedness or f"HAND {index + 1}"
            key = base_key
            if key in used_keys:
                key = f"{base_key} {index + 1}"
            used_keys.add(key)
            prior = self._previous.get(key)
            if prior is not None and 0.0 <= timestamp - prior[0] <= self._maximum_gap_seconds:
                weight = self._smoothing
                landmarks = tuple(
                    tuple(
                        old_value + weight * (new_value - old_value)
                        for old_value, new_value in zip(old_point, new_point)
                    )
                    for old_point, new_point in zip(prior[1], hand.landmarks)
                )
                smoothed = replace(hand, landmarks=landmarks)
            else:
                smoothed = hand
            smoothed_hands.append(smoothed)
            current[key] = (timestamp, smoothed.landmarks)

        self._previous = current
        return tuple(smoothed_hands)
