"""Camera-independent tests for gesture geometry and command handling."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from vision.commands import (
    CommandEvent,
    CommandType,
    GestureCommandProcessor,
    SystemCommandRouter,
    select_pointed_track,
)
from vision.gestures import (
    DetectedHand,
    GestureType,
    HandGesture,
    HandGestureRecognizer,
    classify_landmarks,
)
from vision.tracker import TrackedObject


def hand_pose(gesture: GestureType) -> tuple[tuple[float, float, float], ...]:
    points = [(0.5, 0.8, 0.0) for _ in range(21)]
    points[0] = (0.5, 0.85, 0.0)
    points[1] = (0.45, 0.76, 0.0)
    points[2] = (0.41, 0.71, 0.0)
    points[3] = (0.38, 0.68, 0.0)
    points[4] = (0.35, 0.65, 0.0)

    finger_points = {
        5: ((0.42, 0.62), (0.40, 0.48), (0.39, 0.36), (0.38, 0.20)),
        9: ((0.48, 0.59), (0.47, 0.44), (0.47, 0.32), (0.47, 0.16)),
        13: ((0.55, 0.60), (0.56, 0.46), (0.57, 0.36), (0.58, 0.22)),
        17: ((0.62, 0.63), (0.65, 0.52), (0.67, 0.42), (0.69, 0.31)),
    }
    for mcp, coordinates in finger_points.items():
        pip, dip, tip = coordinates[1:]
        points[mcp] = (*coordinates[0], 0.0)
        if gesture == GestureType.OPEN_PALM or (
            gesture == GestureType.POINTING and mcp == 5
        ):
            points[mcp + 1] = (*pip, 0.0)
            points[mcp + 2] = (*dip, 0.0)
            points[mcp + 3] = (*tip, 0.0)
        else:
            points[mcp + 1] = (*pip, 0.0)
            points[mcp + 2] = (0.48, 0.77, 0.0)
            points[mcp + 3] = (0.49, 0.79, 0.0)

    if gesture == GestureType.OPEN_PALM:
        points[4] = (0.28, 0.65, 0.0)
    elif gesture == GestureType.PINCH:
        points[4] = points[8]
    elif gesture == GestureType.FIST:
        points[4] = (0.25, 0.79, 0.0)
    else:
        points[4] = (0.47, 0.73, 0.0)
    return tuple(points)


def tracked_object(track_id: int, center: tuple[int, int]) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label="person",
        confidence=0.9,
        bbox=(center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10),
        center=center,
        position_history=(center,),
        timestamp=1.0,
        age_seconds=1.0,
        frames_tracked=5,
        missed_frames=0,
    )


class FakeTracker:
    def __init__(self) -> None:
        self.selected_id: int | None = None
        self.reset_count = 0

    def select(self, track_id: int | None) -> bool:
        self.selected_id = track_id
        return True

    def cycle_selection(self, track_ids: list[int]) -> int | None:
        if not track_ids:
            return self.selected_id
        self.selected_id = track_ids[0]
        return self.selected_id

    def reset(self) -> None:
        self.selected_id = None
        self.reset_count += 1


class GestureClassificationTests(unittest.TestCase):
    def test_classifies_four_required_gestures(self) -> None:
        expected = (
            (GestureType.POINTING, GestureType.POINTING),
            (GestureType.OPEN_PALM, GestureType.OPEN_PALM),
            (GestureType.PINCH, GestureType.PINCH),
            (GestureType.FIST, GestureType.FIST),
        )
        for requested, wanted in expected:
            with self.subTest(gesture=requested):
                result, confidence = classify_landmarks(hand_pose(requested))
                self.assertEqual(result, wanted)
                self.assertGreater(confidence, 0.0)

    def test_no_hand_state_does_not_create_gesture(self) -> None:
        self.assertEqual(classify_landmarks([]), (GestureType.NONE, 0.0))
        update = GestureCommandProcessor(stable_frames=2).update(
            HandGesture(), 0.0
        )
        self.assertEqual(update.raw_gesture, GestureType.NONE)
        self.assertEqual(update.events, ())

    def test_recognizer_exposes_both_handed_hands_for_visualization(self) -> None:
        def landmarks_for(scale: float, offset: float) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    x=(x - 0.5) * scale + 0.5 + offset,
                    y=(y - 0.5) * scale + 0.5,
                    z=z,
                )
                for x, y, z in hand_pose(GestureType.OPEN_PALM)
            ]

        results = SimpleNamespace(
            multi_hand_landmarks=[
                SimpleNamespace(landmark=landmarks_for(1.0, -0.2)),
                SimpleNamespace(landmark=landmarks_for(0.55, 0.2)),
            ],
            multi_handedness=[
                SimpleNamespace(classification=[SimpleNamespace(label="Left")]),
                SimpleNamespace(classification=[SimpleNamespace(label="Right")]),
            ],
        )

        class FakeHands:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

            def process(self, _frame: np.ndarray) -> SimpleNamespace:
                del _frame
                return results

            def close(self) -> None:
                pass

        fake_mediapipe = SimpleNamespace(
            solutions=SimpleNamespace(
                hands=SimpleNamespace(Hands=FakeHands),
            )
        )
        with mock.patch.dict("sys.modules", {"mediapipe": fake_mediapipe}):
            recognizer = HandGestureRecognizer()
            detected = recognizer.process(np.zeros((240, 320, 3), dtype=np.uint8))
            recognizer.close()
        self.assertEqual(len(detected.hands), 2)
        self.assertEqual([hand.handedness for hand in detected.hands], ["Right", "Left"])
        self.assertEqual(detected.handedness, "Right")
        self.assertTrue(all(isinstance(hand, DetectedHand) for hand in detected.hands))
        self.assertTrue(all(hand.gesture == GestureType.OPEN_PALM for hand in detected.hands))

    def test_pointing_selects_nearest_object_along_finger_ray(self) -> None:
        landmarks = list(hand_pose(GestureType.POINTING))
        landmarks[6] = (0.20, 0.50, 0.0)
        landmarks[8] = (0.35, 0.50, 0.0)
        hand = HandGesture(GestureType.POINTING, tuple(landmarks))
        tracks = [
            tracked_object(1, (130, 100)),
            tracked_object(2, (105, 145)),
            tracked_object(3, (50, 100)),
        ]
        selected, ray = select_pointed_track(tracks, hand, (200, 200, 3))
        self.assertEqual(selected, 1)
        self.assertIsNotNone(ray)
        self.assertEqual(ray[0], (70, 100))

    def test_pointing_rejects_objects_outside_direction_tolerance(self) -> None:
        landmarks = list(hand_pose(GestureType.POINTING))
        landmarks[6] = (0.20, 0.50, 0.0)
        landmarks[8] = (0.35, 0.50, 0.0)
        hand = HandGesture(GestureType.POINTING, tuple(landmarks))
        selected, _ = select_pointed_track(
            [tracked_object(4, (130, 180))],
            hand,
            (200, 200, 3),
        )
        self.assertIsNone(selected)

    def test_pointing_selects_objects_in_a_different_ray_direction(self) -> None:
        landmarks = list(hand_pose(GestureType.POINTING))
        landmarks[6] = (0.50, 0.55, 0.0)
        landmarks[8] = (0.50, 0.35, 0.0)
        hand = HandGesture(GestureType.POINTING, tuple(landmarks))
        selected, ray = select_pointed_track(
            [tracked_object(5, (100, 30)), tracked_object(6, (150, 100))],
            hand,
            (200, 200, 3),
        )
        self.assertEqual(selected, 5)
        self.assertEqual(ray[0], (100, 70))


class GestureCommandTests(unittest.TestCase):
    def test_stability_cooldown_and_single_activation(self) -> None:
        processor = GestureCommandProcessor(
            stable_frames=3,
            cooldown_seconds=0.5,
            history_length=4,
        )
        open_hand = HandGesture(GestureType.OPEN_PALM, hand_pose(GestureType.OPEN_PALM))
        results = [processor.update(open_hand, time) for time in (0.0, 0.1, 0.2, 0.3)]
        self.assertEqual(results[0].events, ())
        self.assertEqual(results[2].events, (CommandEvent(CommandType.TOGGLE_TRACKING),))
        self.assertEqual(results[3].events, ())

        no_hand = HandGesture()
        for time in (0.4, 0.5, 0.6):
            processor.update(no_hand, time)
        released = [
            processor.update(open_hand, time)
            for time in (0.7, 0.8, 0.9)
        ]
        self.assertEqual(released[-1].events, (CommandEvent(CommandType.TOGGLE_TRACKING),))

    def test_router_applies_gesture_commands_and_keyboard_fallbacks(self) -> None:
        router = SystemCommandRouter()
        tracker = FakeTracker()
        tracks = [tracked_object(7, (80, 90))]
        frame_shape = (200, 200, 3)

        router.handle(CommandEvent(CommandType.TOGGLE_TRACKING), tracker, tracks, frame_shape)
        self.assertTrue(router.state.tracking_paused)
        router.handle(CommandEvent(CommandType.TOGGLE_INTERCEPTION), tracker, tracks, frame_shape)
        self.assertTrue(router.state.interception_active)

        tracker.select(7)
        router.handle(CommandEvent(CommandType.TOGGLE_TARGET_LOCK), tracker, tracks, frame_shape)
        self.assertTrue(router.state.target_locked)
        router.handle(CommandEvent(CommandType.CYCLE_TARGET), tracker, tracks, frame_shape)
        self.assertEqual(router.state.feedback, "UNLOCK TARGET BEFORE CYCLING")
        router.handle_key(ord("l"), tracker, tracks, frame_shape)
        self.assertFalse(router.state.target_locked)

        self.assertTrue(router.handle_key(ord(" "), tracker, tracks, frame_shape))
        self.assertFalse(router.state.tracking_paused)
        self.assertTrue(router.handle_key(ord("r"), tracker, tracks, frame_shape))
        self.assertEqual(tracker.reset_count, 1)
        self.assertFalse(router.state.interception_active)

    def test_locked_target_is_not_replaced_by_pointing(self) -> None:
        router = SystemCommandRouter()
        tracker = FakeTracker()
        tracks = [tracked_object(3, (130, 100)), tracked_object(8, (100, 30))]
        frame_shape = (200, 200, 3)
        landmarks = list(hand_pose(GestureType.POINTING))
        landmarks[6] = (0.20, 0.50, 0.0)
        landmarks[8] = (0.35, 0.50, 0.0)
        router.set_hand(HandGesture(GestureType.POINTING, tuple(landmarks)))
        router.handle(
            CommandEvent(CommandType.POINT_SELECT),
            tracker,
            tracks,
            frame_shape,
        )
        self.assertEqual(tracker.selected_id, 3)

        router.handle(
            CommandEvent(CommandType.TOGGLE_TARGET_LOCK),
            tracker,
            tracks,
            frame_shape,
        )
        landmarks[6] = (0.50, 0.55, 0.0)
        landmarks[8] = (0.50, 0.35, 0.0)
        router.set_hand(HandGesture(GestureType.POINTING, tuple(landmarks)))
        router.handle(
            CommandEvent(CommandType.POINT_SELECT),
            tracker,
            tracks,
            frame_shape,
        )
        self.assertEqual(tracker.selected_id, 3)
        self.assertEqual(router.state.locked_target_id, 3)


if __name__ == "__main__":
    unittest.main()
