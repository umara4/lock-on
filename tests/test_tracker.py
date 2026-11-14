"""Tests for ByteTrack result handling, histories, selection, and trail drawing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from interface.hud import draw_tracking
from vision.tracker import (
    ByteTrackTracker,
    TrackHistoryManager,
    TrackObservation,
    TrackedObject,
)


def observation(
    track_id: int,
    x: int,
    timestamp: float,
    label: str = "person",
) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bbox=(x - 5, 5, x + 5, 15),
        center=(x, 10),
        timestamp=timestamp,
    )


class FakeBoxes:
    def __init__(
        self,
        coordinates: list[list[int]],
        confidences: list[float],
        classes: list[int],
        track_ids: list[int] | None,
    ) -> None:
        self.xyxy = np.asarray(coordinates, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)
        self.id = (
            None
            if track_ids is None
            else np.asarray(track_ids, dtype=np.float32)
        )


class FakeModel:
    def __init__(self, frames: list[FakeBoxes]) -> None:
        self.frames = frames
        self.calls: list[dict[str, object]] = []

    def track(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        boxes = self.frames.pop(0)
        return [SimpleNamespace(boxes=boxes, names={0: "person"})]


class TrackHistoryTests(unittest.TestCase):
    def test_same_class_objects_keep_distinct_ids_and_update_positions(self) -> None:
        manager = TrackHistoryManager(history_length=10, max_missed_frames=3)
        first = manager.update(
            [observation(8, 20, 1.0), observation(3, 80, 1.0)], 1.0
        )
        second = manager.update(
            [observation(8, 30, 2.0), observation(3, 70, 2.0)], 2.0
        )

        self.assertEqual({track.track_id for track in first}, {3, 8})
        by_id = {track.track_id: track for track in second}
        self.assertEqual(by_id[8].position_history, ((20, 10), (30, 10)))
        self.assertEqual(by_id[3].position_history, ((80, 10), (70, 10)))
        self.assertEqual(by_id[8].label, by_id[3].label)
        self.assertEqual(by_id[8].frames_tracked, 2)

    def test_history_is_bounded_and_stale_tracks_are_removed(self) -> None:
        manager = TrackHistoryManager(history_length=2, max_missed_frames=1)
        manager.update([observation(5, 10, 0.0)], 0.0)
        manager.update([observation(5, 20, 1.0)], 1.0)
        current = manager.update([observation(5, 30, 2.0)], 2.0)[0]
        self.assertEqual(current.position_history, ((20, 10), (30, 10)))
        self.assertEqual(current.age_seconds, 2.0)

        manager.update([], 3.0)
        self.assertIn(5, manager.position_histories)
        retained = manager.retained_tracks[0]
        self.assertEqual(retained.missed_frames, 1)
        self.assertEqual(retained.age_seconds, 3.0)
        manager.update([], 4.0)
        self.assertNotIn(5, manager.position_histories)

    def test_reacquisition_resets_consecutive_detection_count(self) -> None:
        manager = TrackHistoryManager(history_length=5, max_missed_frames=2)
        manager.update([observation(5, 10, 0.0)], 0.0)
        manager.update([], 1.0)
        resumed = manager.update([observation(5, 20, 2.0)], 2.0)[0]
        self.assertEqual(resumed.frames_tracked, 1)
        self.assertEqual(resumed.missed_frames, 0)
        self.assertEqual(resumed.age_seconds, 2.0)

    def test_selection_cycles_and_survives_temporary_loss(self) -> None:
        manager = TrackHistoryManager(history_length=5, max_missed_frames=2)
        manager.update([observation(2, 20, 0.0), observation(9, 90, 0.0)], 0.0)
        self.assertEqual(manager.cycle_selection([9, 2]), 2)
        self.assertEqual(manager.cycle_selection([2, 9]), 9)

        manager.update([], 1.0)
        self.assertEqual(manager.cycle_selection([]), 9)
        self.assertTrue(manager.select(9))
        manager.update([], 2.0)
        manager.update([], 3.0)
        self.assertIsNone(manager.selected_id)

    def test_byte_track_adapter_uses_persistent_tracking_api(self) -> None:
        frames = [
            FakeBoxes([[10, 5, 30, 25], [70, 5, 90, 25]], [0.9, 0.8], [0, 0], [4, 7]),
            FakeBoxes([[15, 5, 35, 25], [65, 5, 85, 25]], [0.9, 0.8], [0, 0], [4, 7]),
            FakeBoxes([], [], [], None),
        ]
        model = FakeModel(frames)
        with tempfile.TemporaryDirectory() as directory:
            tracker_config = Path(directory) / "bytetrack.yaml"
            tracker_config.write_text("tracker_type: bytetrack\n", encoding="utf-8")
            tracker = ByteTrackTracker(
                model,
                tracker_config,
                inference_confidence=0.1,
                history_length=10,
                max_missed_frames=3,
            )
            frame = np.zeros((40, 100, 3), dtype=np.uint8)
            first = tracker.update(frame, 1.0)
            second = tracker.update(frame, 2.0)
            tracker.update(frame, 3.0)

        self.assertEqual([track.track_id for track in first], [4, 7])
        self.assertEqual([track.track_id for track in second], [4, 7])
        self.assertEqual(second[0].center, (25, 15))
        self.assertIn(4, tracker.position_histories)
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(all(call["persist"] is True for call in model.calls))
        self.assertTrue(all(call["conf"] == 0.1 for call in model.calls))
        self.assertTrue(
            all(call["imgsz"] == tracker.inference_size for call in model.calls)
        )
        self.assertGreaterEqual(tracker.last_timing.ultralytics_track_seconds, 0.0)
        self.assertGreaterEqual(tracker.last_timing.adapter_seconds, 0.0)

    def test_reset_clears_local_and_ultralytics_tracker_state(self) -> None:
        class FakeByteTracker:
            reset_called = False

            def reset(self) -> None:
                self.reset_called = True

        model = FakeModel([])
        byte_tracker = FakeByteTracker()
        model.predictor = SimpleNamespace(trackers=[byte_tracker])
        with tempfile.TemporaryDirectory() as directory:
            tracker_config = Path(directory) / "bytetrack.yaml"
            tracker_config.write_text("tracker_type: bytetrack\n", encoding="utf-8")
            tracker = ByteTrackTracker(
                model,
                tracker_config,
                inference_confidence=0.1,
                history_length=10,
                max_missed_frames=3,
            )
            tracker.history.update([observation(4, 20, 1.0)], 1.0)
            tracker.select(4)
            tracker.reset()

        self.assertEqual(tracker.position_histories, {})
        self.assertIsNone(tracker.selected_id)
        self.assertTrue(byte_tracker.reset_called)

    def test_trails_render_and_selected_trail_has_distinct_color(self) -> None:
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        track = TrackedObject(
            track_id=1,
            label="person",
            confidence=0.9,
            bbox=(30, 25, 50, 45),
            center=(40, 35),
            position_history=((20, 35), (40, 35)),
            timestamp=1.0,
            age_seconds=1.0,
            frames_tracked=2,
            missed_frames=0,
        )
        output = draw_tracking(
            frame,
            [track],
            {1: track.position_history},
            selected_id=1,
            confidence_threshold=0.35,
        )

        self.assertEqual(output.shape, (60, 100 + 270, 3))
        np.testing.assert_array_equal(output[35, 25], np.array([0, 210, 255]))
        self.assertGreater(int(np.count_nonzero(output[:, 100:])), 0)
        self.assertGreater(
            cv2.countNonZero(cv2.cvtColor(output[:, 100:], cv2.COLOR_BGR2GRAY)),
            0,
        )

    def test_trail_remains_visible_during_temporary_loss(self) -> None:
        frame = np.zeros((60, 100, 3), dtype=np.uint8)
        output = draw_tracking(
            frame,
            [],
            {1: ((10, 30), (40, 30))},
            selected_id=1,
            confidence_threshold=0.35,
        )
        self.assertTrue(np.any(output[30, 15:35]))


if __name__ == "__main__":
    unittest.main()
