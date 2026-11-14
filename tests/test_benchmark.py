"""Tests for reproducible benchmark helpers and synthetic accuracy reports."""

from __future__ import annotations

from contextlib import redirect_stderr
from io import StringIO
import unittest
from unittest import mock

from benchmark import _summarize, _synthetic_accuracy, _synthetic_anime_benchmark, main


class BenchmarkHelperTests(unittest.TestCase):
    def test_summary_reports_mean_median_percentile_and_range(self) -> None:
        summary = _summarize([float(value) for value in range(1, 21)])
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["mean_ms"], 10.5)
        self.assertEqual(summary["median_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19.0)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 20.0)
        self.assertIsNone(_summarize([]))

    def test_seeded_synthetic_accuracy_has_all_prediction_horizons(self) -> None:
        report = _synthetic_accuracy()
        errors = report["prediction_error"]
        self.assertEqual(set(errors), {"0.5s", "1s", "2s"})
        self.assertTrue(all(result["samples"] > 0 for result in errors.values()))
        interception = report["analytic_interception_cases"]
        self.assertTrue(interception["reachable_plan_found"])
        self.assertAlmostEqual(interception["reachable_plan_time_seconds"], 1.0)
        self.assertTrue(interception["unreachable_plan_rejected"])

    def test_synthetic_accuracy_is_reproducible(self) -> None:
        self.assertEqual(_synthetic_accuracy(), _synthetic_accuracy())

    def test_synthetic_anime_benchmark_reports_each_component(self) -> None:
        report = _synthetic_anime_benchmark(iterations=10)
        measured = report["mean_median_p95_latency_ms"]
        self.assertEqual(
            set(measured),
            {
                "landmark_feature_extraction",
                "static_finger_template_match",
                "single_hand_recognition",
                "two_hand_recognition",
                "sequence_recognition",
                "effect_state_update",
                "effect_vector_render",
                "ability_pose_and_movement_recognition",
                "ability_effect_state_update",
                "ability_effect_vector_render",
            },
        )
        self.assertTrue(all(result is not None for result in measured.values()))
        self.assertIn("not labeled real-hand", report["accuracy_claim"])

    def test_invalid_video_source_exits_with_an_actionable_error(self) -> None:
        stderr = StringIO()
        with (
            mock.patch(
                "sys.argv",
                ["benchmark.py", "--source", "missing-benchmark-video.mp4"],
            ),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            main()
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("video file does not exist", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
