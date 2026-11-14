"""Tests that application orchestration reuses vision and motion components."""

from __future__ import annotations

from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from application.pipeline import PipelineCommand, VisionPipeline
from interface.effects import EffectType
from tests.test_anime_gestures import detected, focus_pose
from tests.test_gestures import hand_pose
from vision.gestures import GestureType, HandGesture
from vision.tracker import TrackedObject, TrackerTiming


def tracked(timestamp: float) -> TrackedObject:
    x = int(50 + timestamp * 20)
    return TrackedObject(
        track_id=7,
        label="person",
        confidence=0.92,
        bbox=(x - 10, 20, x + 10, 60),
        center=(x, 40),
        position_history=((x, 40),),
        timestamp=timestamp,
        age_seconds=timestamp,
        frames_tracked=1,
        missed_frames=0,
    )


class FakeTracker:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.selected_id: int | None = None
        self._tracks: list[TrackedObject] = []
        self.position_histories: dict[int, tuple[tuple[int, int], ...]] = {}
        self.last_timing = TrackerTiming()

    def update(self, _frame: np.ndarray, timestamp: float) -> list[TrackedObject]:
        item = tracked(timestamp)
        self._tracks = [item]
        self.position_histories = {item.track_id: (item.center,)}
        return self._tracks

    def reset(self) -> None:
        self.selected_id = None
        self._tracks = []
        self.position_histories = {}

    def select(self, track_id: int | None) -> bool:
        if track_id is not None and not any(t.track_id == track_id for t in self._tracks):
            return False
        self.selected_id = track_id
        return True

    def cycle_selection(self, visible_ids: list[int]) -> int | None:
        if visible_ids:
            self.selected_id = visible_ids[0]
        return self.selected_id

    @property
    def retained_tracks(self) -> list[TrackedObject]:
        return list(self._tracks)


class FakeGestureRecognizer:
    calls = 0

    def __init__(self) -> None:
        self.calls = 0

    def process(self, _frame: np.ndarray) -> HandGesture:
        self.calls += 1
        return HandGesture()

    def close(self) -> None:
        pass


class FakeBodyPoseTracker:
    def process(
        self,
        _frame: np.ndarray,
    ) -> tuple[tuple[float, float, float, float], ...]:
        return ()

    def close(self) -> None:
        pass


class PipelineTests(unittest.TestCase):
    def create_pipeline(self) -> tuple[VisionPipeline, FakeGestureRecognizer]:
        recognizer = FakeGestureRecognizer()
        patches = (
            mock.patch(
                "application.pipeline.YOLODetector",
                return_value=SimpleNamespace(model=object()),
            ),
            mock.patch("application.pipeline.ByteTrackTracker", FakeTracker),
            mock.patch(
                "application.pipeline.HandGestureRecognizer",
                return_value=recognizer,
            ),
            mock.patch(
                "application.pipeline.BodyPoseTracker",
                FakeBodyPoseTracker,
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        pipeline = VisionPipeline(
            "model.pt",
            0.3,
            "cpu",
            "tracker.yaml",
            classification_enabled=False,
        )
        return pipeline, recognizer

    def test_frame_updates_tracking_motion_and_selectable_target(self) -> None:
        pipeline, _ = self.create_pipeline()
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        pipeline.process_frame(frame, 1.0)
        pipeline.process_frame(frame, 1.1)
        snapshot = pipeline.process_frame(frame, 1.2)
        self.assertEqual(len(snapshot.tracks), 1)
        self.assertEqual(snapshot.motion_estimates[7].measurement_count, 3)

        selected = pipeline.apply_command(PipelineCommand.SELECT_TARGET, 1.2, 7)
        self.assertEqual(selected.selected_id, 7)
        self.assertEqual(selected.selected_track.track_id, 7)

    def test_worker_settings_and_gesture_sampling_are_configurable(self) -> None:
        pipeline, recognizer = self.create_pipeline()
        with mock.patch("application.pipeline.config.GESTURE_PROCESSING_INTERVAL_SECONDS", 0.1):
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            pipeline.process_frame(frame, 1.0)
            pipeline.process_frame(frame, 1.05)
            pipeline.process_frame(frame, 1.11)
        self.assertEqual(recognizer.calls, 2)

        snapshot = pipeline.apply_command(PipelineCommand.TOGGLE_PAUSE, 1.2)
        self.assertTrue(snapshot.tracking_paused)
        snapshot = pipeline.apply_command(PipelineCommand.TOGGLE_DETECTION, 1.2)
        self.assertFalse(snapshot.detection_enabled)

    def test_anime_controls_are_independent_from_system_gesture_controls(self) -> None:
        pipeline, recognizer = self.create_pipeline()
        system_off = pipeline.apply_command(PipelineCommand.TOGGLE_GESTURES, 1.0)
        self.assertFalse(system_off.gestures_enabled)
        self.assertTrue(system_off.anime_gestures_enabled)

        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        anime_only = pipeline.process_frame(frame, 1.1)
        self.assertFalse(anime_only.gestures_enabled)
        self.assertTrue(anime_only.anime_gestures_enabled)
        self.assertEqual(recognizer.calls, 1)

        anime_off = pipeline.apply_command(PipelineCommand.TOGGLE_ANIME_GESTURES, 1.2)
        self.assertFalse(anime_off.anime_gestures_enabled)
        self.assertFalse(anime_off.gestures_enabled)
        effects_off = pipeline.apply_command(PipelineCommand.TOGGLE_ANIME_EFFECTS, 1.3)
        self.assertFalse(effects_off.anime_effects_enabled)

    def test_recognized_pose_triggers_effects_without_system_command(self) -> None:
        pipeline, _ = self.create_pipeline()
        focus = detected(focus_pose(), "Left")

        class FocusRecognizer:
            def process(self, _frame: np.ndarray) -> HandGesture:
                return HandGesture(
                    GestureType.OTHER,
                    focus.landmarks,
                    focus.handedness,
                    focus.confidence,
                    (focus,),
                )

            def close(self) -> None:
                pass

        pipeline.gesture_recognizer = FocusRecognizer()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        pipeline.process_frame(frame, 1.0)
        snapshot = pipeline.process_frame(frame, 1.4)

        self.assertEqual(snapshot.gesture_update.events, ())
        self.assertTrue(snapshot.anime_gestures_enabled)
        self.assertEqual(
            {effect.effect_type for effect in snapshot.effects},
            {EffectType.ENERGY_CHARGE, EffectType.AURA},
        )
        self.assertEqual(
            snapshot.ability_recognition.selected_ability_id,
            "chidori",
        )
        self.assertTrue(snapshot.ability_effects)

    def test_ability_master_toggle_is_independent_of_system_gesture_toggle(self) -> None:
        pipeline, _ = self.create_pipeline()
        snapshot = pipeline.apply_command(PipelineCommand.TOGGLE_ABILITIES, 1.0)
        self.assertFalse(snapshot.abilities_enabled)
        self.assertTrue(snapshot.gestures_enabled)
        self.assertTrue(snapshot.anime_gestures_enabled)

    def test_calibration_command_persists_derived_features(self) -> None:
        focus = detected(focus_pose(), "Left")

        class FocusRecognizer:
            def process(self, _frame: np.ndarray) -> HandGesture:
                return HandGesture(
                    GestureType.OTHER,
                    focus.landmarks,
                    focus.handedness,
                    focus.confidence,
                    (focus,),
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ability-calibration.json"
            with mock.patch(
                "application.pipeline.config.ABILITY_CALIBRATION_PATH",
                str(path),
            ):
                pipeline, _ = self.create_pipeline()
            pipeline.gesture_recognizer = FocusRecognizer()
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            pipeline.process_frame(frame, 2.0)
            pipeline.process_frame(frame, 2.4)
            snapshot = pipeline.apply_command(
                PipelineCommand.RECORD_ABILITY_SAMPLE,
                2.5,
                ability_id="chidori",
            )
            self.assertEqual(snapshot.ability_samples, {"chidori": 1})
            self.assertTrue(path.is_file())

    def test_pipeline_records_timestamped_movement_sequence(self) -> None:
        focus = detected(focus_pose(), "Left")

        class FocusRecognizer:
            def process(self, _frame: np.ndarray) -> HandGesture:
                return HandGesture(
                    GestureType.OTHER,
                    focus.landmarks,
                    focus.handedness,
                    focus.confidence,
                    (focus,),
                )

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ability-calibration.json"
            with mock.patch(
                "application.pipeline.config.ABILITY_CALIBRATION_PATH",
                str(path),
            ):
                pipeline, _ = self.create_pipeline()
            pipeline.gesture_recognizer = FocusRecognizer()
            frame = np.zeros((640, 640, 3), dtype=np.uint8)
            pipeline.apply_command(
                PipelineCommand.TOGGLE_ABILITY_MOVEMENT_RECORDING,
                3.0,
                ability_id="chidori",
            )
            for timestamp in (3.1, 3.2, 3.3, 3.4):
                pipeline.process_frame(frame, timestamp)
            snapshot = pipeline.apply_command(
                PipelineCommand.TOGGLE_ABILITY_MOVEMENT_RECORDING,
                3.5,
                ability_id="chidori",
            )
            self.assertFalse(snapshot.ability_calibration_recording)
            self.assertEqual(snapshot.ability_movement_samples, {"chidori": 1})
            saved = Path(path).read_text(encoding="utf-8")
            self.assertNotIn('"landmarks"', saved)

    def test_two_hand_ability_poses_do_not_fire_single_hand_system_commands(self) -> None:
        left = detected(
            hand_pose(GestureType.FIST),
            "Left",
            GestureType.FIST,
        )
        right = detected(
            tuple((x + 0.1, y, z) for x, y, z in hand_pose(GestureType.FIST)),
            "Right",
            GestureType.FIST,
        )

        class TwoFistRecognizer:
            def process(self, _frame: np.ndarray) -> HandGesture:
                return HandGesture(
                    GestureType.FIST,
                    left.landmarks,
                    left.handedness,
                    left.confidence,
                    (left, right),
                )

            def close(self) -> None:
                pass

        pipeline, _ = self.create_pipeline()
        pipeline.gesture_recognizer = TwoFistRecognizer()
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        for timestamp in (4.0, 4.15, 4.3):
            snapshot = pipeline.process_frame(frame, timestamp)
        self.assertEqual(snapshot.gesture_update.events, ())
        self.assertFalse(snapshot.interception_active)

    def test_classification_can_be_enabled_without_loading_model_on_pipeline_thread(self) -> None:
        pipeline, _ = self.create_pipeline()
        snapshot = pipeline.apply_command(PipelineCommand.TOGGLE_CLASSIFICATION, 1.0)
        self.assertTrue(snapshot.classification_enabled)
        self.assertIsNone(pipeline.recognition_manager._worker)
        snapshot = pipeline.apply_command(PipelineCommand.TOGGLE_CLASSIFICATION, 1.1)
        self.assertFalse(snapshot.classification_enabled)
