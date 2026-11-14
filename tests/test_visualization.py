"""Tests for landmark-derived outlines and short-term smoothing."""

from __future__ import annotations

import unittest

from interface.visualization import HAND_CONNECTIONS, HandLandmarkSmoother, hand_outline
from vision.gestures import DetectedHand, GestureType


def landmarks(offset_x: float = 0.0, offset_y: float = 0.0) -> tuple[tuple[float, float, float], ...]:
    points = [
        (0.50, 0.82), (0.43, 0.72), (0.38, 0.62), (0.34, 0.52), (0.30, 0.43),
        (0.44, 0.58), (0.43, 0.43), (0.43, 0.31), (0.43, 0.18),
        (0.50, 0.56), (0.50, 0.40), (0.50, 0.27), (0.50, 0.14),
        (0.56, 0.58), (0.57, 0.43), (0.58, 0.32), (0.59, 0.23),
        (0.62, 0.63), (0.65, 0.51), (0.67, 0.41), (0.69, 0.32),
    ]
    return tuple((x + offset_x, y + offset_y, 0.0) for x, y in points)


class HandVisualizationGeometryTests(unittest.TestCase):
    def test_outline_is_a_hull_not_a_bounding_rectangle(self) -> None:
        outline = hand_outline(landmarks())
        self.assertGreaterEqual(len(outline), 5)
        self.assertLessEqual(len(outline), 21)
        self.assertNotEqual(len(outline), 4)
        self.assertEqual(len(HAND_CONNECTIONS), 21)

    def test_multiple_hands_keep_handedness_and_gestures(self) -> None:
        left = DetectedHand(landmarks(), "Left", GestureType.OPEN_PALM, 0.9)
        right = DetectedHand(landmarks(0.25), "Right", GestureType.PINCH, 0.8)
        smoother = HandLandmarkSmoother()
        result = smoother.update((left, right), 1.0)
        self.assertEqual([hand.handedness for hand in result], ["Left", "Right"])
        self.assertEqual([hand.gesture for hand in result], [GestureType.OPEN_PALM, GestureType.PINCH])

    def test_smoothing_reduces_jitter_but_follows_fast_motion(self) -> None:
        smoother = HandLandmarkSmoother(smoothing=0.5)
        first = DetectedHand(landmarks(), "Left")
        moved = DetectedHand(landmarks(0.1), "Left")
        smoother.update((first,), 1.0)
        smoothed = smoother.update((moved,), 1.05)[0]
        self.assertAlmostEqual(smoothed.landmarks[0][0], 0.55)
        self.assertEqual(smoothed.landmarks[0][1], moved.landmarks[0][1])

    def test_missing_hand_is_removed_immediately_and_stale_state_is_not_reused(self) -> None:
        smoother = HandLandmarkSmoother(smoothing=0.5, maximum_gap_seconds=0.2)
        first = DetectedHand(landmarks(), "Right")
        moved = DetectedHand(landmarks(0.1), "Right")
        smoother.update((first,), 1.0)
        self.assertEqual(smoother.update((), 1.05), ())
        returned = smoother.update((moved,), 1.1)[0]
        self.assertEqual(returned.landmarks, moved.landmarks)
        smoother.update((first,), 2.0)
        after_gap = smoother.update((moved,), 2.5)[0]
        self.assertEqual(after_gap.landmarks, moved.landmarks)

    def test_partial_and_out_of_frame_landmarks_still_produce_outline(self) -> None:
        points = list(landmarks())
        points[8] = (-0.08, 0.12, 0.0)
        outline = hand_outline(tuple(points))
        self.assertGreaterEqual(len(outline), 3)
        self.assertTrue(any(x < 0 for x, _ in outline))
        self.assertEqual(hand_outline(tuple(points[:20])), ())


if __name__ == "__main__":
    unittest.main()
