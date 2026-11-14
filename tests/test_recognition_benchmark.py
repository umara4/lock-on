"""Tests for labeled crop-classifier benchmark metrics."""

from __future__ import annotations

import unittest

from benchmark_recognition import evaluate_predictions
from vision.recognition import ClassificationCandidate


class RecognitionBenchmarkTests(unittest.TestCase):
    def test_topk_metrics_match_imagenet_synonyms_and_folder_names(self) -> None:
        report = evaluate_predictions(
            [
                (
                    "computer_keyboard",
                    (
                        ClassificationCandidate("keyboard, computer keyboard", 0.7),
                        ClassificationCandidate("typewriter keyboard", 0.2),
                    ),
                ),
                (
                    "cell_phone",
                    (
                        ClassificationCandidate("remote control", 0.4),
                        ClassificationCandidate("cellular telephone", 0.3),
                    ),
                ),
            ]
        )
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["top1_accuracy"], 0.5)
        self.assertEqual(report["top5_accuracy"], 1.0)
        self.assertEqual(report["per_class"]["cell_phone"]["top5_accuracy"], 1.0)

    def test_empty_evaluation_has_no_accuracy_claim(self) -> None:
        report = evaluate_predictions([])
        self.assertIsNone(report["top1_accuracy"])
        self.assertIsNone(report["top5_accuracy"])


if __name__ == "__main__":
    unittest.main()
