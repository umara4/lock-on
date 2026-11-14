"""Rotation- and scale-tolerant geometric features for MediaPipe hands."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, degrees

import numpy as np

Landmarks = tuple[tuple[float, float, float], ...]

FINGER_JOINTS: tuple[tuple[int, int, int], ...] = (
    (2, 3, 4),
    (5, 6, 8),
    (9, 10, 12),
    (13, 14, 16),
    (17, 18, 20),
)
@dataclass(frozen=True)
class HandFeatures:
    center: tuple[float, float, float]
    palm_scale: float
    finger_extension_ratios: tuple[float, ...]
    finger_states: tuple[bool | None, ...]
    fingertip_distances: tuple[float, ...]
    joint_angles_degrees: tuple[float, ...]
    palm_orientation_degrees: float
    normalized_landmarks: tuple[tuple[float, float, float], ...]


def extract_hand_features(
    landmarks: Landmarks,
    aspect_ratio: float = 1.0,
) -> HandFeatures | None:
    """Derive palm-relative shape features; avoid dependence on pixel scale.

    Euclidean ratios and joint angles are invariant to image translation and
    planar camera rotation. Perspective foreshortening and occlusion remain
    inherent limitations of a monocular 2D landmark input.
    """
    if len(landmarks) != 21:
        return None
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("aspect_ratio must be finite and positive")
    points = np.asarray(landmarks, dtype=np.float64)
    if not np.isfinite(points).all():
        return None

    metric_points = points.copy()
    metric_points[:, 0] *= aspect_ratio
    center = points[[0, 5, 9, 13, 17]].mean(axis=0)
    metric_center = metric_points[[0, 5, 9, 13, 17]].mean(axis=0)
    palm_scale = max(
        float(np.linalg.norm(metric_points[0] - metric_points[9])),
        float(np.linalg.norm(metric_points[5] - metric_points[17])),
        1e-8,
    )
    ratios: list[float] = []
    states: list[bool | None] = []
    angles: list[float] = []
    for base, joint, tip in FINGER_JOINTS:
        first = metric_points[joint] - metric_points[base]
        second = metric_points[tip] - metric_points[joint]
        wrist_distance = max(float(np.linalg.norm(metric_points[0] - metric_points[joint])), 1e-8)
        ratios.append(
            float(np.linalg.norm(metric_points[0] - metric_points[tip]))
            / wrist_distance
        )
        cosine = float(np.dot(first, second)) / max(
            float(np.linalg.norm(first) * np.linalg.norm(second)),
            1e-8,
        )
        angles.append(degrees(acos(float(np.clip(cosine, -1.0, 1.0)))))
        ratio = ratios[-1]
        states.append(True if ratio >= 1.12 else False if ratio <= 0.92 else None)

    tips = metric_points[[4, 8, 12, 16, 20]]
    tip_distances = tuple(
        float(np.linalg.norm(tips[first] - tips[second])) / palm_scale
        for first in range(5)
        for second in range(first + 1, 5)
    )
    normalized = (metric_points - metric_center) / palm_scale
    palm_vector = metric_points[9, :2] - metric_points[0, :2]
    return HandFeatures(
        center=tuple(float(value) for value in center),
        palm_scale=palm_scale,
        finger_extension_ratios=tuple(ratios),
        finger_states=tuple(states),
        fingertip_distances=tip_distances,
        joint_angles_degrees=tuple(angles),
        palm_orientation_degrees=degrees(atan2(float(palm_vector[1]), float(palm_vector[0]))),
        normalized_landmarks=tuple(
            tuple(float(value) for value in point) for point in normalized
        ),
    )


def score_finger_states(
    features: HandFeatures,
    expected: tuple[bool, ...],
) -> float:
    """Return exact fraction of unambiguous per-finger states matching a template."""
    if len(expected) != len(features.finger_states) or not expected:
        raise ValueError("finger template must match the extracted finger count")
    return sum(
        actual is not None and actual == desired
        for actual, desired in zip(features.finger_states, expected)
    ) / len(expected)
