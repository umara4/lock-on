"""Tests for COCO and optional YOLO-World detector selection."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from vision.detector import YOLODetector


class DetectorTests(unittest.TestCase):
    def test_coco_detector_remains_the_default_model_path(self) -> None:
        model = object()
        yolo = mock.Mock(return_value=model)
        fake_ultralytics = types.SimpleNamespace(YOLO=yolo)
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            detector = YOLODetector("coco.pt", 0.3)
        yolo.assert_called_once_with("coco.pt")
        self.assertIs(detector.model, model)
        self.assertEqual(detector.detector_type, "coco")

    def test_world_detector_sets_the_requested_text_vocabulary(self) -> None:
        model = mock.Mock()
        world = mock.Mock(return_value=model)
        fake_ultralytics = types.SimpleNamespace(YOLO=mock.Mock(), YOLOWorld=world)
        classes = ("thermometer", "stethoscope")
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            detector = YOLODetector(
                "world.pt",
                0.3,
                detector_type="world",
                vocabulary=classes,
            )
        world.assert_called_once_with("world.pt")
        model.set_classes.assert_called_once_with(list(classes))
        self.assertEqual(detector.detector_type, "world")

    def test_world_detector_rejects_an_empty_vocabulary(self) -> None:
        fake_ultralytics = types.SimpleNamespace(YOLO=mock.Mock(), YOLOWorld=mock.Mock())
        with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
            with self.assertRaisesRegex(ValueError, "non-empty"):
                YOLODetector("world.pt", 0.3, detector_type="world", vocabulary=())


if __name__ == "__main__":
    unittest.main()
