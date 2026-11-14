"""Tests for crop validation, temporal consensus, and asynchronous track identity."""

from __future__ import annotations

import threading
import time
import unittest
from typing import Callable
from unittest import mock

import numpy as np

from vision.recognition import (
    ClassificationCandidate,
    ClassificationStatus,
    CropClassifier,
    TrackRecognitionManager,
    extract_detection_crop,
)
from vision.tracker import TrackedObject


def track(track_id: int, frames: int = 3, label: str = "object") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label=label,
        confidence=0.88,
        bbox=(20, 20, 100, 100),
        center=(60, 60),
        position_history=((60, 60),),
        timestamp=1.0,
        age_seconds=0.5,
        frames_tracked=frames,
        missed_frames=0,
    )


class FakeClassifier:
    model_id = "test/fake-classifier"

    def __init__(
        self,
        candidates: tuple[ClassificationCandidate, ...] = (
            ClassificationCandidate("ballpoint pen", 0.82),
            ClassificationCandidate("pencil", 0.08),
        ),
    ) -> None:
        self.candidates = candidates
        self.calls = 0

    def classify(self, crop: np.ndarray) -> tuple[ClassificationCandidate, ...]:
        self.calls += 1
        if crop.size == 0:
            raise ValueError("empty crop")
        return self.candidates


class RecognitionTests(unittest.TestCase):
    def make_manager(
        self,
        classifier_factory: Callable[[], CropClassifier] | None = None,
        max_pending_jobs: int = 4,
        requests_per_second: float = 100.0,
    ) -> TrackRecognitionManager:
        return TrackRecognitionManager(
            classifier_factory=classifier_factory or FakeClassifier,
            max_pending_jobs=max_pending_jobs,
            requests_per_second=requests_per_second,
        )

    @staticmethod
    def wait_for(manager: TrackRecognitionManager, track_id: int, count: int = 1) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = manager.snapshot().get(track_id)
            if manager.completed_count >= count and state is not None and state.status != ClassificationStatus.CLASSIFYING:
                return
            time.sleep(0.005)
        raise AssertionError("classification did not finish in time")

    def test_crop_clips_to_frame_and_copies_memory(self) -> None:
        frame = np.ones((100, 120, 3), dtype=np.uint8)
        crop = extract_detection_crop(
            frame,
            (-10, 5, 25, 50),
            minimum_size=20,
            padding_ratio=0,
        )
        self.assertIsNotNone(crop)
        assert crop is not None
        self.assertEqual(crop.shape, (45, 25, 3))
        self.assertFalse(np.shares_memory(crop, frame))

    def test_invalid_and_tiny_crops_are_rejected(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(extract_detection_crop(frame, (30, 30, 20, 40)))
        self.assertIsNone(extract_detection_crop(frame, (2, 2, 12, 12)))
        self.assertIsNone(extract_detection_crop(np.zeros((0, 10, 3)), (0, 0, 8, 8)))

    def test_refined_label_requires_temporal_agreement_and_preserves_detection(self) -> None:
        classifier = FakeClassifier()
        manager = self.make_manager(lambda: classifier)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(7, label="toothbrush")], {7}, 1.0, 7)
        self.wait_for(manager, 7, 1)
        first = manager.snapshot()[7]
        self.assertEqual(first.detector_label, "toothbrush")
        self.assertEqual(first.status, ClassificationStatus.UNCERTAIN)
        self.assertIsNone(first.refined_label)

        with mock.patch("vision.recognition.config.CLASSIFICATION_RETRY_SECONDS", 0.1):
            manager.update(frame, [track(7, label="toothbrush")], {7}, 2.0, 7)
            self.wait_for(manager, 7, 2)
        refined = manager.snapshot()[7]
        self.assertEqual(refined.detector_label, "toothbrush")
        self.assertEqual(refined.status, ClassificationStatus.REFINED)
        self.assertEqual(refined.refined_label, "ballpoint pen")
        self.assertEqual(refined.display_label, "ballpoint pen")
        self.assertEqual(classifier.calls, 2)

    def test_matching_labels_are_confirmed_not_rewritten(self) -> None:
        classifier = FakeClassifier(
            (
                ClassificationCandidate("keyboard, computer keyboard", 0.83),
                ClassificationCandidate("typewriter keyboard", 0.12),
            )
        )
        manager = self.make_manager(lambda: classifier)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(3, label="keyboard")], {3}, 1.0, 3)
        self.wait_for(manager, 3, 1)
        with mock.patch("vision.recognition.config.CLASSIFICATION_RETRY_SECONDS", 0.1):
            manager.update(frame, [track(3, label="keyboard")], {3}, 2.0, 3)
            self.wait_for(manager, 3, 2)
        result = manager.snapshot()[3]
        self.assertEqual(result.status, ClassificationStatus.CONFIRMED)
        self.assertEqual(result.detector_label, "keyboard")
        self.assertEqual(result.display_label, "keyboard")

    def test_empty_classifier_output_is_uncertain_not_unsupported(self) -> None:
        manager = self.make_manager(lambda: FakeClassifier(()))
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(12)], {12}, 1.0, 12)
        self.wait_for(manager, 12, 1)
        result = manager.snapshot()[12]
        self.assertEqual(result.status, ClassificationStatus.UNCERTAIN)
        self.assertEqual(result.error, "Classifier returned no valid candidate labels.")

    def test_detector_label_change_invalidates_temporal_consensus(self) -> None:
        classifier = FakeClassifier()
        manager = self.make_manager(lambda: classifier)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(13, label="first label")], {13}, 1.0, 13)
        self.wait_for(manager, 13, 1)
        initial_generation = manager._states[13].generation
        with mock.patch("vision.recognition.config.CLASSIFICATION_RETRY_SECONDS", 0.1):
            manager.update(frame, [track(13, label="replacement label")], {13}, 2.0, 13)
            self.wait_for(manager, 13, 2)
        result = manager.snapshot()[13]
        self.assertGreater(manager._states[13].generation, initial_generation)
        self.assertEqual(result.detector_label, "replacement label")
        self.assertEqual(result.status, ClassificationStatus.UNCERTAIN)
        self.assertIsNone(result.refined_label)

    def test_full_queue_is_bounded_and_dropped_track_is_not_retried_each_frame(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingClassifier(FakeClassifier):
            def classify(self, crop: np.ndarray) -> tuple[ClassificationCandidate, ...]:
                entered.set()
                release.wait(1.0)
                return super().classify(crop)

        manager = self.make_manager(BlockingClassifier, max_pending_jobs=1)
        self.addCleanup(manager.close)
        self.addCleanup(release.set)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(1)], {1}, 1.0, 1)
        self.assertTrue(entered.wait(1.0))
        manager.update(frame, [track(1), track(2)], {1, 2}, 2.0, 2)
        manager.update(frame, [track(1), track(2), track(3)], {1, 2, 3}, 3.0, 3)
        self.assertEqual(manager._jobs.qsize(), 1)
        self.assertEqual(manager.dropped_count, 1)
        manager.update(
            frame,
            [track(1), track(2), track(3)],
            {1, 2, 3},
            3.01,
            3,
        )
        self.assertEqual(manager.dropped_count, 1)

    def test_cache_avoids_reclassifying_each_frame(self) -> None:
        classifier = FakeClassifier()
        manager = self.make_manager(lambda: classifier)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(4)], {4}, 1.0, 4)
        self.wait_for(manager, 4, 1)
        for timestamp in (1.1, 1.2, 2.0, 3.0):
            manager.update(frame, [track(4)], {4}, timestamp, 4)
        self.assertEqual(classifier.calls, 1)

    def test_multiple_tracks_get_independent_cached_results(self) -> None:
        classifier = FakeClassifier()
        manager = self.make_manager(lambda: classifier, requests_per_second=100.0)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        with mock.patch("vision.recognition.config.CLASSIFICATION_MAX_REQUESTS_PER_SECOND", 100.0):
            manager.update(
                frame,
                [track(1), track(2)],
                {1, 2},
                10.0,
                2,
            )
            manager.update(frame, [track(1), track(2)], {1, 2}, 10.02, 2)
        self.wait_for(manager, 2, 2)
        states = manager.snapshot()
        self.assertEqual(set(states), {1, 2})
        self.assertEqual(states[1].detector_label, "object")
        self.assertEqual(states[2].detector_label, "object")

    def test_stale_results_after_reset_cannot_update_reused_id(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        class BlockingClassifier(FakeClassifier):
            def classify(self, crop: np.ndarray) -> tuple[ClassificationCandidate, ...]:
                entered.set()
                release.wait(1.0)
                return super().classify(crop)

        manager = self.make_manager(BlockingClassifier)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(9)], {9}, 1.0, 9)
        self.assertTrue(entered.wait(1.0))
        manager.reset()
        manager.update(frame, [track(9, label="new object")], {9}, 2.0, 9)
        release.set()
        self.wait_for(manager, 9, 1)
        result = manager.snapshot()[9]
        self.assertEqual(result.detector_label, "new object")
        self.assertEqual(result.status, ClassificationStatus.UNCERTAIN)

    def test_classifier_initialization_failure_is_visible(self) -> None:
        def fail_to_load() -> FakeClassifier:
            raise FileNotFoundError("weights unavailable")

        manager = self.make_manager(fail_to_load)
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(5)], {5}, 1.0, 5)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            result = manager.snapshot().get(5)
            if result is not None and result.error:
                break
            time.sleep(0.005)
        result = manager.snapshot()[5]
        self.assertEqual(result.status, ClassificationStatus.UNCERTAIN)
        self.assertIn("weights unavailable", result.error or "")

    def test_disabled_manager_does_not_create_jobs_or_states(self) -> None:
        classifier = FakeClassifier()
        manager = TrackRecognitionManager(classifier_factory=lambda: classifier, enabled=False)
        self.addCleanup(manager.close)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        self.assertEqual(manager.update(frame, [track(8)], {8}, 1.0, 8), {})
        self.assertIsNone(manager._worker)
        self.assertEqual(classifier.calls, 0)

    def test_track_retention_expiry_reuses_id_with_a_new_generation(self) -> None:
        manager = self.make_manager()
        self.addCleanup(manager.close)
        frame = np.full((120, 120, 3), 127, dtype=np.uint8)
        manager.update(frame, [track(11)], {11}, 1.0, 11)
        old_generation = manager._states[11].generation
        manager.update(frame, [], set(), 2.0, None)
        manager.update(frame, [track(11, label="replacement")], {11}, 3.0, 11)
        self.assertGreater(manager._states[11].generation, old_generation)
        self.assertEqual(manager.snapshot()[11].detector_label, "replacement")


if __name__ == "__main__":
    unittest.main()
