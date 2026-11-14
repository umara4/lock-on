"""Synthetic, camera-independent tests for configurable hand-sign matching."""

from __future__ import annotations

from math import cos, sin
import unittest

from tests.test_gestures import hand_pose
from vision.anime_gestures import AnimeGestureRecognizer, PoseStatus
from vision.gestures import DetectedHand, GestureType


def focus_pose(
    *,
    center: tuple[float, float] = (0.5, 0.5),
    scale: float = 1.0,
    angle: float = 0.0,
    mirror: bool = False,
) -> tuple[tuple[float, float, float], ...]:
    landmarks = list(hand_pose(GestureType.OPEN_PALM))
    landmarks[3] = (0.48, 0.8, 0.0)
    landmarks[4] = (0.5, 0.85, 0.0)
    landmarks[15] = (0.48, 0.77, 0.0)
    landmarks[16] = (0.49, 0.79, 0.0)
    landmarks[19] = (0.48, 0.77, 0.0)
    landmarks[20] = (0.49, 0.79, 0.0)
    c, s = cos(angle), sin(angle)
    result = []
    for x, y, z in landmarks:
        dx = (x - 0.5) * scale
        dy = (y - 0.5) * scale
        rotated_x = c * dx - s * dy + center[0]
        rotated_y = s * dx + c * dy + center[1]
        result.append((1.0 - rotated_x if mirror else rotated_x, rotated_y, z * scale))
    return tuple(result)


def detected(
    pose: tuple[tuple[float, float, float], ...],
    handedness: str,
    gesture: GestureType = GestureType.OTHER,
) -> DetectedHand:
    return DetectedHand(pose, handedness, gesture, 0.9, 0.96)


class AnimeGestureRecognitionTests(unittest.TestCase):
    def test_focus_pose_requires_hold_then_emits_once_and_releases(self) -> None:
        recognizer = AnimeGestureRecognizer(
            cooldown_seconds=0.0,
            maximum_lost_seconds=0.6,
        )
        hand = detected(focus_pose(), "Left")

        self.assertEqual(recognizer.update((hand,), 1.0, (640, 640)).pose.status, PoseStatus.RECOGNIZING)
        self.assertEqual(recognizer.update((hand,), 1.34, (640, 640)).events, ())
        recognized = recognizer.update((hand,), 1.36, (640, 640))
        self.assertEqual(recognized.pose.pose_id, "two_finger_focus")
        self.assertEqual(recognized.pose.status, PoseStatus.RECOGNIZED)
        self.assertEqual([event.kind for event in recognized.events], ["recognized"])

        held = recognizer.update((hand,), 1.6, (640, 640))
        self.assertEqual(held.pose.status, PoseStatus.HELD)
        self.assertEqual(held.events, ())

        released = recognizer.update((), 1.7, (640, 640))
        self.assertEqual(released.pose.status, PoseStatus.RELEASED)
        self.assertEqual([event.kind for event in released.events], ["released"])

    def test_focus_matching_is_tolerant_to_mirroring_rotation_and_scale(self) -> None:
        recognizer = AnimeGestureRecognizer(
            cooldown_seconds=0.0,
            maximum_lost_seconds=0.6,
        )
        mirrored = detected(
            focus_pose(center=(0.45, 0.52), scale=0.65, angle=0.22, mirror=True),
            "Right",
        )
        recognizer.update((mirrored,), 2.0, (640, 640))
        result = recognizer.update((mirrored,), 2.4, (640, 640))
        self.assertEqual(result.pose.pose_id, "two_finger_focus")
        self.assertEqual(result.pose.hand_ids, ("hand-1",))
        self.assertEqual(result.hands[0].handedness, "Right")

    def test_basic_system_gesture_is_not_an_anime_pose(self) -> None:
        recognizer = AnimeGestureRecognizer(cooldown_seconds=0.0)
        open_palm = detected(hand_pose(GestureType.OPEN_PALM), "Left", GestureType.OPEN_PALM)
        for timestamp in (1.0, 1.5, 2.0):
            snapshot = recognizer.update((open_palm,), timestamp, (640, 640))
        self.assertIsNone(snapshot.pose)
        self.assertEqual(snapshot.events, ())

    def test_two_hand_seal_requires_both_close_hands(self) -> None:
        recognizer = AnimeGestureRecognizer(cooldown_seconds=0.0)
        left = detected(focus_pose(center=(0.43, 0.5), scale=0.45), "Left")
        right = detected(focus_pose(center=(0.57, 0.5), scale=0.45), "Right")
        single = recognizer.update((left,), 1.0, (640, 640))
        self.assertNotEqual(
            single.pose.pose_id if single.pose else None,
            "two_hand_seal",
        )

        recognizer.update((left, right), 2.0, (640, 640))
        result = recognizer.update((right, left), 2.5, (640, 640))
        self.assertEqual(result.pose.pose_id, "two_hand_seal")
        self.assertEqual(len(result.pose.hand_ids), 2)
        self.assertEqual(len({hand.identity for hand in result.hands}), 2)

    def test_two_hand_seal_rejects_missing_or_spatially_separated_hand(self) -> None:
        recognizer = AnimeGestureRecognizer(cooldown_seconds=0.0)
        left = detected(focus_pose(center=(0.2, 0.5), scale=0.25), "Left")
        right = detected(focus_pose(center=(0.8, 0.5), scale=0.25), "Right")
        for hands, timestamp in (((left,), 1.0), ((left, right), 2.0)):
            snapshot = recognizer.update(hands, timestamp, (640, 640))
            self.assertNotEqual(
                snapshot.pose.pose_id if snapshot.pose else None,
                "two_hand_seal",
            )

    def test_focus_release_then_two_hand_seal_completes_sequence_once(self) -> None:
        recognizer = AnimeGestureRecognizer(cooldown_seconds=0.0)
        focus = detected(focus_pose(), "Left")
        recognizer.update((focus,), 1.0, (640, 640))
        first = recognizer.update((focus,), 1.36, (640, 640))
        self.assertEqual(first.sequence_progress[0].completed_steps, 1)
        recognizer.update((), 1.4, (640, 640))

        left = detected(focus_pose(center=(0.43, 0.5), scale=0.45), "Left")
        right = detected(focus_pose(center=(0.57, 0.5), scale=0.45), "Right")
        recognizer.update((left, right), 1.5, (640, 640))
        complete = recognizer.update((left, right), 2.0, (640, 640))
        self.assertEqual(
            [event.sequence_id for event in complete.sequence_events],
            ["focus_then_seal"],
        )
        held = recognizer.update((left, right), 2.3, (640, 640))
        self.assertEqual(held.sequence_events, ())

    def test_hand_tracks_expire_after_configured_loss_gap(self) -> None:
        recognizer = AnimeGestureRecognizer(maximum_lost_seconds=0.2)
        hand = detected(focus_pose(), "Left")
        initial = recognizer.update((hand,), 1.0, (640, 640)).hands[0]
        missing = recognizer.update((), 1.1, (640, 640)).hands[0]
        self.assertEqual(initial.identity, missing.identity)
        self.assertFalse(missing.visible)
        expired = recognizer.update((), 1.3, (640, 640))
        self.assertEqual(expired.hands, ())


if __name__ == "__main__":
    unittest.main()
