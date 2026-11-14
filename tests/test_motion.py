"""Synthetic trajectory tests and pixel-space prediction error measurements."""

from __future__ import annotations

import math
import unittest

import numpy as np

from interface.hud import draw_status, draw_tracking
from prediction.kalman import ConstantVelocityKalman
from prediction.motion import MotionEstimate, MotionEstimator, PredictionStatus
from vision.tracker import TrackedObject


def track(track_id: int, x: int, y: int, timestamp: float, label: str = "person") -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        label=label,
        confidence=0.9,
        bbox=(x - 5, y - 5, x + 5, y + 5),
        center=(x, y),
        position_history=((x, y),),
        timestamp=timestamp,
        age_seconds=timestamp,
        frames_tracked=1,
        missed_frames=0,
    )


class KalmanFilterTests(unittest.TestCase):
    def test_variable_delta_time_recovers_horizontal_velocity(self) -> None:
        kalman = ConstantVelocityKalman(measurement_noise=0.01, process_noise=0.1)
        kalman.update((0.0, 20.0), 0.0)
        estimate = kalman.update((30.0, 20.0), 0.3)
        self.assertAlmostEqual(estimate.velocity[0], 100.0, delta=1.0)
        self.assertAlmostEqual(estimate.velocity[1], 0.0, delta=0.1)
        future = kalman.predict(0.5)
        self.assertAlmostEqual(future[0], 80.0, delta=0.01)
        self.assertAlmostEqual(future[1], 20.0, delta=0.01)

    def test_reset_and_input_validation(self) -> None:
        kalman = ConstantVelocityKalman(measurement_noise=1.0, process_noise=1.0)
        kalman.update((1.0, 2.0), 1.0)
        kalman.reset()
        self.assertFalse(kalman.initialized)
        with self.assertRaises(RuntimeError):
            kalman.predict(0.5)
        kalman.update((1.0, 2.0), 1.0)
        with self.assertRaises(ValueError):
            kalman.update((2.0, 3.0), 1.0)


class MotionEstimatorTests(unittest.TestCase):
    def make_estimator(self, **kwargs: object) -> MotionEstimator:
        settings: dict[str, object] = {
            "measurement_noise": 1.0,
            "process_noise": 10.0,
            "max_speed": 1_000.0,
            "stationary_speed": 2.0,
            "max_lost_seconds": 1.0,
            "low_confidence_residual": 100.0,
        }
        settings.update(kwargs)
        return MotionEstimator(
            **settings,
        )

    def test_stationary_object_converges_to_zero_speed(self) -> None:
        estimator = self.make_estimator()
        for index in range(20):
            estimates = estimator.update(
                [track(1, 100, 80, index * 0.1)],
                index * 0.1,
            )
        estimate = estimates[1]
        self.assertTrue(estimate.stationary)
        self.assertLess(estimate.speed, 2.0)
        self.assertEqual(estimate.status, PredictionStatus.TRACKING)

    def test_horizontal_vertical_and_diagonal_constant_velocity(self) -> None:
        cases = (
            (100.0, 0.0),
            (0.0, 60.0),
            (60.0, 80.0),
        )
        for track_id, (vx, vy) in enumerate(cases, start=1):
            with self.subTest(vx=vx, vy=vy):
                estimator = self.make_estimator()
                estimate = None
                for index in range(25):
                    timestamp = index * 0.1
                    x = round(100 + vx * timestamp)
                    y = round(100 + vy * timestamp)
                    estimate = estimator.update(
                        [track(track_id, x, y, timestamp)],
                        timestamp,
                    )[track_id]
                self.assertIsNotNone(estimate)
                self.assertAlmostEqual(estimate.velocity[0], vx, delta=4.0)
                self.assertAlmostEqual(estimate.velocity[1], vy, delta=4.0)
                self.assertAlmostEqual(
                    estimate.speed,
                    math.hypot(vx, vy),
                    delta=5.0,
                )
                self.assertAlmostEqual(
                    estimate.predictions[0.5][0],
                    estimate.filtered_position[0] + estimate.velocity[0] * 0.5,
                )

    def test_acceleration_and_direction_change_are_reported(self) -> None:
        estimator = self.make_estimator()
        for index in range(15):
            timestamp = index * 0.1
            estimator.update([track(2, 100 + index * 10, 100, timestamp)], timestamp)
        before = estimator.estimates[2]
        self.assertGreater(before.velocity[0], 50.0)

        direction_changed = False
        for index in range(15, 25):
            timestamp = index * 0.1
            estimate = estimator.update(
                [track(2, 240 - (index - 15) * 10, 100, timestamp)],
                timestamp,
            )[2]
            direction_changed = direction_changed or estimate.direction_changed
        self.assertLess(estimate.velocity[0], 0.0)
        self.assertTrue(direction_changed)
        self.assertLess(estimate.acceleration[0], 0.0)

    def test_noisy_measurements_are_smoothed(self) -> None:
        estimator = self.make_estimator(measurement_noise=16.0, process_noise=2.0)
        rng = np.random.default_rng(42)
        raw_errors: list[float] = []
        filtered_errors: list[float] = []
        for index in range(60):
            timestamp = index * 0.1
            true_x = 50 + 35 * timestamp
            measured_x = true_x + float(rng.normal(0.0, 4.0))
            estimate = estimator.update(
                [track(4, round(measured_x), 80, timestamp)],
                timestamp,
            )[4]
            if index > 5:
                raw_errors.append(abs(measured_x - true_x))
                filtered_errors.append(abs(estimate.filtered_position[0] - true_x))
        self.assertLess(np.mean(filtered_errors), np.mean(raw_errors))

    def test_variable_frame_timing_is_used_for_velocity(self) -> None:
        estimator = self.make_estimator()
        timestamp = 0.0
        estimate = None
        for interval in (0.05, 0.18, 0.07, 0.22, 0.11, 0.15, 0.09, 0.2):
            timestamp += interval
            estimate = estimator.update(
                [track(5, round(20 + 40 * timestamp), 50, timestamp)],
                timestamp,
            )[5]
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.velocity[0], 40.0, delta=5.0)

    def test_temporary_loss_preserves_filter_then_permanent_loss_removes_it(self) -> None:
        estimator = self.make_estimator()
        for index in range(8):
            timestamp = index * 0.1
            estimator.update([track(6, 20 + index * 5, 50, timestamp)], timestamp)
        before_loss = estimator.estimates[6].velocity

        during_loss = estimator.update([], 0.9)[6]
        self.assertLess(during_loss.seconds_since_measurement, 1.0)
        self.assertNotEqual(during_loss.status, PredictionStatus.LOST)
        after_loss = estimator.update([], 1.5)[6]
        self.assertEqual(after_loss.status, PredictionStatus.LOST)
        recovered = estimator.update([track(6, 65, 50, 1.6)], 1.6)[6]
        self.assertEqual(recovered.measurement_count, 9)
        self.assertNotEqual(recovered.status, PredictionStatus.INITIALIZING)
        self.assertGreater(before_loss[0], 0.0)

        estimator.update([], 2.7)
        self.assertNotIn(6, estimator.estimates)

    def test_multiple_objects_keep_independent_motion_states(self) -> None:
        estimator = self.make_estimator()
        for index in range(20):
            timestamp = index * 0.1
            estimates = estimator.update(
                [
                    track(10, 20 + index * 8, 40, timestamp, "person"),
                    track(11, 100, 30 + index * 4, timestamp, "car"),
                ],
                timestamp,
            )
        self.assertGreater(estimates[10].velocity[0], 50.0)
        self.assertAlmostEqual(estimates[10].velocity[1], 0.0, delta=1.0)
        self.assertAlmostEqual(estimates[11].velocity[0], 0.0, delta=1.0)
        self.assertGreater(estimates[11].velocity[1], 20.0)

    def test_id_reuse_with_new_class_reinitializes_filter(self) -> None:
        estimator = self.make_estimator()
        estimator.update([track(12, 30, 40, 0.0)], 0.0)
        estimator.update([track(12, 40, 40, 0.1)], 0.1)
        replacement = estimator.update(
            [track(12, 300, 200, 0.2, "car")],
            0.2,
        )[12]
        self.assertEqual(replacement.measurement_count, 1)
        self.assertEqual(replacement.measured_position, (300.0, 200.0))
        self.assertEqual(replacement.status, PredictionStatus.INITIALIZING)

    def test_excessive_velocity_is_clamped_and_marked(self) -> None:
        estimator = self.make_estimator(max_speed=100.0)
        estimator.update([track(13, 0, 0, 0.0)], 0.0)
        estimate = estimator.update([track(13, 500, 0, 0.1)], 0.1)[13]
        self.assertTrue(estimate.velocity_clamped)
        self.assertAlmostEqual(estimate.speed, 100.0, delta=0.01)
        self.assertEqual(estimate.predictions[0.5][0], estimate.filtered_position[0] + 50.0)

    def test_prediction_visualization_and_status_render(self) -> None:
        estimate = MotionEstimate(
            track_id=14,
            timestamp=1.0,
            measured_position=(40.0, 50.0),
            position_history=(),
            filtered_position=(42.0, 50.0),
            velocity=(30.0, 0.0),
            speed=30.0,
            acceleration=(0.0, 0.0),
            direction_degrees=0.0,
            stationary=False,
            predictions={0.5: (57.0, 50.0), 1.0: (72.0, 50.0), 2.0: (102.0, 50.0)},
            status=PredictionStatus.PREDICTING,
            measurement_count=10,
            residual_pixels=1.0,
            seconds_since_measurement=0.0,
            direction_changed=False,
            velocity_clamped=False,
        )
        frame = np.zeros((480, 320, 3), dtype=np.uint8)
        rendered = draw_tracking(
            frame,
            [],
            {},
            selected_id=14,
            confidence_threshold=0.35,
            motion_estimate=estimate,
        )
        draw_status(rendered, 25.0, False, selected_id=14, motion_estimate=estimate)
        self.assertEqual(rendered.shape, (480, 320 + 270, 3))
        self.assertGreater(np.count_nonzero(rendered[:, :320]), 0)
        self.assertGreater(np.count_nonzero(rendered[:, 320:]), 0)

    def test_prediction_error_metrics_for_constant_velocity(self) -> None:
        estimator = self.make_estimator(
            measurement_noise=2.0,
            process_noise=8.0,
            low_confidence_residual=100.0,
        )
        errors: dict[float, list[float]] = {0.5: [], 1.0: [], 2.0: []}
        speed = (45.0, 20.0)
        for index in range(100):
            timestamp = index * 0.05
            measured = track(
                20,
                round(80 + speed[0] * timestamp),
                round(40 + speed[1] * timestamp),
                timestamp,
            )
            estimate = estimator.update([measured], timestamp)[20]
            if index < 20:
                continue
            for horizon in errors:
                predicted = estimate.predictions[horizon]
                truth = (
                    80 + speed[0] * (timestamp + horizon),
                    40 + speed[1] * (timestamp + horizon),
                )
                errors[horizon].append(
                    math.hypot(predicted[0] - truth[0], predicted[1] - truth[1])
                )

        for horizon, samples in errors.items():
            mean_error = float(np.mean(samples))
            rmse = float(np.sqrt(np.mean(np.square(samples))))
            print(
                f"constant-velocity {horizon:.1f}s prediction: "
                f"mean position error={mean_error:.2f}px, RMSE={rmse:.2f}px"
            )
            self.assertLess(mean_error, 8.0)
            self.assertLess(rmse, 9.0)


if __name__ == "__main__":
    unittest.main()
