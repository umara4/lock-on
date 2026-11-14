"""Synthetic camera-independent tests for ability pose and movement gates."""

from __future__ import annotations

import tempfile
import json
import unittest
from pathlib import Path

from tests.test_anime_gestures import detected, focus_pose
from tests.test_gestures import hand_pose
from vision.ability_calibration import AbilityCalibrationStore
from vision.ability_recognition import AbilityRecognizer
from vision.anime_gestures import AnimeGestureRecognizer
from vision.gestures import GestureType


def cupped_pose(
    center: tuple[float, float],
    scale: float = 0.55,
    depth_offset: float = 0.0,
) -> tuple[tuple[float, float, float], ...]:
    points = list(hand_pose(GestureType.OPEN_PALM))
    for base, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
        bx, by, bz = points[base]
        tx, ty, tz = points[tip]
        points[tip] = (
            bx + (tx - bx) * 0.55,
            by + (ty - by) * 0.55,
            bz + (tz - bz) * 0.55,
        )
    return tuple(
        (
            center[0] + (x - 0.5) * scale,
            center[1] + (y - 0.5) * scale,
            z * scale + depth_offset,
        )
        for x, y, z in points
    )


def ability_hands(
    recognizer: AnimeGestureRecognizer,
    timestamp: float,
    poses: tuple[tuple[tuple[float, float, float], ...], ...],
):
    observations = tuple(
        detected(pose, "Left" if index == 0 else "Right")
        for index, pose in enumerate(poses)
    )
    return recognizer.update(observations, timestamp, (640, 480))


def upper_body_pose(mirrored: bool = False) -> tuple[tuple[float, float, float, float], ...]:
    landmarks = [(0.5, 0.5, 0.0, 1.0) for _ in range(33)]
    values = (
        {
            11: (0.65, 0.4), 12: (0.35, 0.4),
            13: (0.75, 0.55), 14: (0.25, 0.55),
            15: (0.42, 0.65), 16: (0.58, 0.65),
        }
        if mirrored
        else {
            11: (0.35, 0.4), 12: (0.65, 0.4),
            13: (0.25, 0.55), 14: (0.75, 0.55),
            15: (0.58, 0.65), 16: (0.42, 0.65),
        }
    )
    for index, (x, y) in values.items():
        landmarks[index] = (x, y, 0.0, 0.98)
    return tuple(landmarks)


class AbilityRecognitionTests(unittest.TestCase):
    def test_chidori_requires_focus_hold_then_timestamped_thrust(self) -> None:
        hands = AnimeGestureRecognizer(maximum_lost_seconds=1.0)
        abilities = AbilityRecognizer()
        start = detected(focus_pose(center=(0.30, 0.50)), "Right")
        for timestamp in (1.0, 1.15, 1.3, 1.4):
            snapshot = hands.update((start,), timestamp, (640, 480))
            result = abilities.update(snapshot, timestamp)
        self.assertEqual(result.selected_ability_id, "chidori")
        self.assertEqual(result.phase, "CHARGING")
        self.assertFalse(any(event.phase == "activated" for event in result.events))

        thrust = detected(focus_pose(center=(0.58, 0.50)), "Right")
        snapshot = hands.update((thrust,), 1.55, (640, 480))
        result = abilities.update(snapshot, 1.55)
        self.assertTrue(any(event.phase == "activated" for event in result.events))
        self.assertEqual(result.selected_ability_id, "chidori")

    def test_kamehameha_requires_two_hands_and_synchronized_motion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            hands = AnimeGestureRecognizer(maximum_lost_seconds=1.0)
            store = AbilityCalibrationStore(Path(directory) / "kamehameha.json")
            initial_poses = (cupped_pose((0.34, 0.48)), cupped_pose((0.46, 0.48)))
            initial = ability_hands(hands, 2.0, initial_poses)
            store.record("kamehameha", initial.hands, 2, 640 / 480)
            abilities = AbilityRecognizer(store)
            for timestamp in (2.1, 2.3, 2.5, 2.8):
                snapshot = ability_hands(hands, timestamp, initial_poses)
                result = abilities.update(
                    snapshot,
                    timestamp,
                    upper_body_pose(),
                    (640, 480),
                )
            self.assertEqual(result.selected_ability_id, "kamehameha")
            self.assertEqual(result.phase, "CHARGING")
            self.assertFalse(any(event.phase == "activated" for event in result.events))

            thrust_poses = (
                cupped_pose((0.34, 0.48), depth_offset=-0.16),
                cupped_pose((0.46, 0.48), depth_offset=-0.16),
            )
            snapshot = ability_hands(hands, 2.95, thrust_poses)
            result = abilities.update(
                snapshot,
                2.95,
                upper_body_pose(),
                (640, 480),
            )
            self.assertTrue(any(event.phase == "activated" for event in result.events))

    def test_kamehameha_does_not_activate_from_visible_hands_without_calibration(self) -> None:
        hands = AnimeGestureRecognizer()
        gestures = (hand_pose(GestureType.OPEN_PALM),)
        result = AbilityRecognizer().update(
            ability_hands(hands, 1.0, gestures),
            1.0,
            (),
            (640, 480),
        )
        self.assertIsNone(result.selected_ability_id)

    def test_domain_expansion_is_single_hand_and_requires_recorded_21_point_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AbilityCalibrationStore(Path(directory) / "calibration.json")
            hands = AnimeGestureRecognizer(maximum_lost_seconds=1.0)
            sign = detected(focus_pose(center=(0.5, 0.5), scale=0.8), "Right")
            first = hands.update((sign,), 3.0, (640, 480))
            store.record("domain_expansion", first.hands, 1, 640 / 480)
            restored = AbilityCalibrationStore(Path(directory) / "calibration.json")
            abilities = AbilityRecognizer(restored)
            events = []
            for timestamp in (3.1, 3.4, 3.7, 3.9):
                result = abilities.update(
                    hands.update((sign,), timestamp, (640, 480)),
                    timestamp,
                    frame_size=(640, 480),
                )
                events.extend(result.events)
            self.assertEqual(result.selected_ability_id, "domain_expansion")
            self.assertTrue(any(event.phase == "activated" for event in events))

            extra_hand = detected(focus_pose(center=(0.65, 0.5)), "Left")
            two_hands = hands.update((sign, extra_hand), 3.95, (640, 480))
            self.assertIsNone(
                AbilityRecognizer(restored).update(
                    two_hands,
                    3.95,
                    frame_size=(640, 480),
                ).selected_ability_id
            )

            wrong = detected(hand_pose(GestureType.FIST), "Right", GestureType.FIST)
            released = abilities.update(
                hands.update((wrong,), 4.0, (640, 480)),
                4.0,
                frame_size=(640, 480),
            )
            self.assertTrue(any(event.phase == "released" for event in released.events))

    def test_crossed_arms_recognition_is_stable_and_mirror_invariant(self) -> None:
        hands = AnimeGestureRecognizer(maximum_lost_seconds=1.0)
        abilities = AbilityRecognizer()
        two_hands = (
            hand_pose(GestureType.OPEN_PALM),
            tuple((x + 0.08, y, z) for x, y, z in hand_pose(GestureType.OPEN_PALM)),
        )
        activations = []
        for timestamp in (2.0, 2.3, 2.6, 2.8):
            result = abilities.update(
                ability_hands(hands, timestamp, two_hands),
                timestamp,
                upper_body_pose(mirrored=timestamp == 2.8),
                (640, 480),
            )
            activations.extend(
                event for event in result.events if event.phase == "activated"
            )
        self.assertEqual(len(activations), 1)
        self.assertEqual(activations[0].ability_id, "shadow_clone")

        invalid_pose = list(upper_body_pose())
        invalid_pose[15] = (0.36, 0.65, 0.0, 0.98)
        invalid_pose[16] = (0.40, 0.65, 0.0, 0.98)
        fresh = AnimeGestureRecognizer()
        rejected = AbilityRecognizer().update(
            ability_hands(fresh, 1.0, two_hands),
            1.0,
            tuple(invalid_pose),
            (640, 480),
        )
        self.assertIsNone(rejected.selected_ability_id)

    def test_calibration_samples_persist_and_gate_other_pose_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ability-calibration.json"
            recognizer = AnimeGestureRecognizer()
            pose = detected(focus_pose(), "Left")
            observation = recognizer.update((pose,), 1.0, (640, 480)).hands
            store = AbilityCalibrationStore(path, minimum_similarity=0.8)
            store.record("chidori", observation)

            reloaded = AbilityCalibrationStore(path, minimum_similarity=0.8)
            self.assertEqual(reloaded.sample_counts, {"chidori": 1})
            self.assertTrue(reloaded.accepts("chidori", observation))
            wrong_pose = detected(hand_pose(GestureType.FIST), "Left")
            wrong = AnimeGestureRecognizer().update(
                (wrong_pose,),
                2.0,
                (640, 480),
            )
            self.assertIsNone(
                AbilityRecognizer(reloaded).update(wrong, 2.0).selected_ability_id
            )

    def test_movement_sequences_and_threshold_persist_without_landmarks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ability-calibration.json"
            store = AbilityCalibrationStore(path)
            recognizer = AnimeGestureRecognizer()
            store.begin_movement_recording("chidori", 1.0)
            for timestamp, center_x in ((1.0, 0.35), (1.1, 0.45), (1.2, 0.55)):
                observations = recognizer.update(
                    (detected(focus_pose(center=(center_x, 0.5)), "Left"),),
                    timestamp,
                    (640, 480),
                ).hands
                store.append_movement_frame(observations, timestamp)
            self.assertEqual(store.finish_movement_recording(), 1)
            self.assertLess(store.movement_threshold("chidori", 0.7), 0.7)
            store.set_minimum_similarity(0.84)

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("landmarks", json.dumps(payload))
            self.assertEqual(len(payload["movement_samples"][0]["frames"]), 3)
            restored = AbilityCalibrationStore(path)
            self.assertEqual(restored.movement_sample_counts, {"chidori": 1})
            self.assertEqual(restored.minimum_similarity, 0.84)

    def test_kamehameha_movement_calibration_includes_forward_depth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AbilityCalibrationStore(Path(directory) / "movement.json")
            recognizer = AnimeGestureRecognizer()
            store.begin_movement_recording("kamehameha", 1.0)
            for timestamp, depth in (
                (1.0, 0.0),
                (1.1, -0.08),
                (1.2, -0.16),
            ):
                hands = ability_hands(
                    recognizer,
                    timestamp,
                    (
                        cupped_pose((0.42, 0.48), depth_offset=depth),
                        cupped_pose((0.55, 0.48), depth_offset=depth),
                    ),
                ).hands
                store.append_movement_frame(hands, timestamp)

            self.assertEqual(store.finish_movement_recording(), 1)
            threshold = store.movement_threshold("kamehameha", 0.7)
            self.assertGreaterEqual(threshold, 0.2)
            self.assertNotEqual(threshold, 0.7)
            movement = json.loads(
                (Path(directory) / "movement.json").read_text(encoding="utf-8")
            )["movement_samples"][0]["frames"]
            self.assertNotEqual(
                movement[0]["hands"][0]["center_3d"][2],
                movement[-1]["hands"][0]["center_3d"][2],
            )

    def test_spirit_bomb_requires_both_open_hands_raised_and_held(self) -> None:
        def raised_open_hand(
            center_x: float,
            vertical_shift: float,
        ) -> tuple[tuple[float, float, float], ...]:
            return tuple(
                (
                    center_x + (x - 0.5) * 0.6,
                    0.5 + (y - 0.5) * 0.6 + vertical_shift,
                    z * 0.6,
                )
                for x, y, z in hand_pose(GestureType.OPEN_PALM)
            )

        hands = AnimeGestureRecognizer(maximum_lost_seconds=1.0)
        abilities = AbilityRecognizer()
        events = []
        for timestamp, vertical_shift in (
            (1.0, 0.0),
            (1.2, -0.2),
            (1.6, -0.2),
            (2.0, -0.2),
            (2.2, -0.2),
        ):
            left = detected(
                raised_open_hand(0.43, vertical_shift),
                "Left",
                GestureType.OPEN_PALM,
            )
            right = detected(
                raised_open_hand(0.57, vertical_shift),
                "Right",
                GestureType.OPEN_PALM,
            )
            result = abilities.update(
                hands.update((left, right), timestamp, (640, 480)),
                timestamp,
            )
            events.extend(result.events)
        self.assertEqual(result.selected_ability_id, "spirit_bomb")
        self.assertTrue(any(event.phase == "activated" for event in events))



if __name__ == "__main__":
    unittest.main()
