"""Tests for recording structured feature samples without persisting video."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tests.test_anime_gestures import focus_pose
from vision.gesture_templates import GestureSampleRecorder
from vision.gestures import DetectedHand, GestureType


class GestureTemplateRecorderTests(unittest.TestCase):
    def test_records_single_and_two_hand_features_without_raw_landmarks(self) -> None:
        recorder = GestureSampleRecorder(maximum_samples_per_gesture=2)
        first = DetectedHand(focus_pose(center=(0.42, 0.5)), "Left", GestureType.OTHER)
        second = DetectedHand(focus_pose(center=(0.58, 0.5)), "Right", GestureType.OTHER)
        recorder.record("focus", "Focus sign", (first,), 1.0, 1.0)
        sample = recorder.record("seal", "Two-hand seal", (first, second), 1.1, 1.0)

        self.assertEqual(len(sample.feature_vectors), 2)
        self.assertIsNotNone(sample.relative_hand_features)
        payload = recorder.to_payload()
        self.assertEqual(payload["schema"], "jarvis.gesture-features.v1")
        serialized = json.dumps(payload)
        self.assertNotIn('"landmarks"', serialized)
        self.assertNotIn("frame", payload)
        self.assertIn("sequences", payload)
        self.assertIn("action_mappings", payload)

    def test_sample_limit_and_json_save(self) -> None:
        recorder = GestureSampleRecorder(maximum_samples_per_gesture=1)
        hand = DetectedHand(focus_pose(), "Left", GestureType.OTHER)
        recorder.record("focus", "Focus", (hand,), 1.0)
        recorder.record("focus", "Focus", (hand,), 2.0)
        self.assertEqual(recorder.sample_counts, {"focus": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gesture-samples.json"
            recorder.save(path)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["samples"][0]["timestamp"], 2.0)


if __name__ == "__main__":
    unittest.main()
