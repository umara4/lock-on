"""Reproducible tests for interception planning and closed-loop simulation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from interface.hud import draw_simulation_panel
from prediction.motion import MotionEstimate, MotionSample, PredictionStatus
from robotics.interceptor import (
    InterceptionSimulator,
    InterceptionStatus,
    calculate_interception,
)


def estimate(
    timestamp: float,
    position: tuple[float, float],
    velocity: tuple[float, float] = (0.0, 0.0),
    *,
    track_id: int = 1,
    status: PredictionStatus = PredictionStatus.PREDICTING,
    measurements: int = 10,
) -> MotionEstimate:
    return MotionEstimate(
        track_id=track_id,
        timestamp=timestamp,
        measured_position=position,
        position_history=(MotionSample(timestamp, position),),
        filtered_position=position,
        velocity=velocity,
        speed=math.hypot(*velocity),
        acceleration=(0.0, 0.0),
        direction_degrees=None,
        stationary=math.hypot(*velocity) < 1e-6,
        predictions={},
        status=status,
        measurement_count=measurements,
        residual_pixels=0.0,
        seconds_since_measurement=0.0,
        direction_changed=False,
        velocity_clamped=False,
    )


class InterceptionCalculationTests(unittest.TestCase):
    def test_stationary_target(self) -> None:
        plan = calculate_interception((0.0, 0.0), (100.0, 0.0), (0.0, 0.0), 50.0)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(plan.time_seconds, 2.0)
        self.assertEqual(plan.position, (100.0, 0.0))

    def test_target_moving_toward_robot(self) -> None:
        plan = calculate_interception((0.0, 0.0), (100.0, 0.0), (-10.0, 0.0), 50.0)
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(plan.time_seconds, 100.0 / 60.0)

    def test_target_moving_away_and_unreachable(self) -> None:
        reachable = calculate_interception(
            (0.0, 0.0), (100.0, 0.0), (10.0, 0.0), 50.0
        )
        impossible = calculate_interception(
            (0.0, 0.0), (100.0, 0.0), (60.0, 0.0), 50.0
        )
        self.assertIsNotNone(reachable)
        self.assertAlmostEqual(reachable.time_seconds, 2.5)
        self.assertIsNone(impossible)

    def test_perpendicular_and_diagonal_motion(self) -> None:
        perpendicular = calculate_interception(
            (0.0, 0.0), (100.0, 0.0), (0.0, 30.0), 50.0
        )
        diagonal = calculate_interception(
            (0.0, 0.0), (100.0, 100.0), (20.0, -10.0), 80.0
        )
        self.assertIsNotNone(perpendicular)
        self.assertAlmostEqual(perpendicular.time_seconds, 2.5)
        self.assertIsNotNone(diagonal)
        self.assertAlmostEqual(
            math.dist(diagonal.position, (0.0, 0.0)) / diagonal.time_seconds,
            80.0,
            delta=1e-6,
        )

    def test_response_delay_and_maximum_horizon(self) -> None:
        plan = calculate_interception(
            (0.0, 0.0), (100.0, 0.0), (10.0, 0.0), 50.0, response_delay=0.5
        )
        too_late = calculate_interception(
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 0.0),
            50.0,
            response_delay=0.5,
            max_horizon=2.0,
        )
        self.assertIsNotNone(plan)
        self.assertAlmostEqual(plan.time_seconds, 3.125)
        self.assertIsNone(too_late)

    def test_robot_speed_changes_feasibility_and_eta(self) -> None:
        slow = calculate_interception((0.0, 0.0), (100.0, 0.0), (0.0, 0.0), 25.0)
        fast = calculate_interception((0.0, 0.0), (100.0, 0.0), (0.0, 0.0), 100.0)
        impossible = calculate_interception((0.0, 0.0), (100.0, 0.0), (60.0, 0.0), 50.0)
        self.assertAlmostEqual(slow.time_seconds, 4.0)
        self.assertAlmostEqual(fast.time_seconds, 1.0)
        self.assertIsNone(impossible)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_interception((0, 0), (1, 0), (0, 0), 0.0)


class InterceptionControllerTests(unittest.TestCase):
    def make_simulator(self, **kwargs: object) -> InterceptionSimulator:
        settings: dict[str, object] = {
            "robot_speed": 100.0,
            "response_delay": 0.0,
            "tolerance": 8.0,
            "max_horizon": 5.0,
        }
        settings.update(kwargs)
        return InterceptionSimulator(**settings)

    def test_robot_moves_at_bounded_speed_and_toward_planned_point(self) -> None:
        simulator = self.make_simulator()
        simulator.step(estimate(0.0, (200.0, 180.0)), 0.0, True, 320, 240)
        previous = simulator.snapshot.robot_position
        current = simulator.step(estimate(0.1, (200.0, 180.0)), 0.1, True, 320, 240)
        travel = math.dist(previous, current.robot_position)
        self.assertLessEqual(travel, 100.0 * 0.1 + 1e-6)
        self.assertLess(math.dist(current.robot_position, (200.0, 180.0)), math.dist(previous, (200.0, 180.0)))
        self.assertEqual(current.status, InterceptionStatus.APPROACHING)

    def test_response_delay_prevents_early_motion(self) -> None:
        simulator = self.make_simulator(response_delay=0.3)
        simulator.step(estimate(0.0, (200.0, 180.0)), 0.0, True, 320, 240)
        initial = simulator.snapshot.robot_position
        for timestamp in (0.1, 0.2):
            snapshot = simulator.step(
                estimate(timestamp, (200.0, 180.0)),
                timestamp,
                True,
                320,
                240,
            )
            self.assertEqual(snapshot.robot_position, initial)
        moved = simulator.step(estimate(0.4, (200.0, 180.0)), 0.4, True, 320, 240)
        self.assertNotEqual(moved.robot_position, initial)

    def test_no_feasible_plan_reports_impossible(self) -> None:
        simulator = self.make_simulator(robot_speed=40.0)
        snapshot = simulator.step(
            estimate(0.0, (20.0, 200.0), (-100.0, 0.0)),
            0.0,
            True,
            320,
            240,
        )
        self.assertEqual(snapshot.status, InterceptionStatus.IMPOSSIBLE)
        self.assertIsNone(snapshot.interception_point)

    def test_low_confidence_estimate_stops_planning(self) -> None:
        simulator = self.make_simulator()
        snapshot = simulator.step(
            estimate(
                0.0,
                (200.0, 180.0),
                status=PredictionStatus.LOW_CONFIDENCE,
            ),
            0.0,
            True,
            320,
            240,
        )
        self.assertEqual(snapshot.status, InterceptionStatus.LOW_CONFIDENCE)
        self.assertIsNone(snapshot.interception_point)

    def test_target_loss_stops_controller(self) -> None:
        simulator = self.make_simulator()
        simulator.step(estimate(0.0, (200.0, 180.0)), 0.0, True, 320, 240)
        snapshot = simulator.step(None, 0.1, True, 320, 240)
        self.assertEqual(snapshot.status, InterceptionStatus.TARGET_LOST)
        self.assertEqual(snapshot.target_id, 1)
        self.assertIsNone(snapshot.interception_point)
        self.assertEqual(len(snapshot.target_history), 1)

    def test_simulation_hud_renders_appended_panel(self) -> None:
        simulator = self.make_simulator()
        snapshot = simulator.step(
            estimate(0.0, (200.0, 180.0), (10.0, 0.0)),
            0.0,
            True,
            320,
            240,
        )
        frame = np.zeros((240, 590, 3), dtype=np.uint8)
        rendered = draw_simulation_panel(frame, snapshot)
        self.assertEqual(rendered.shape, (240, 590 + 330, 3))
        self.assertGreater(np.count_nonzero(rendered[:, 590:]), 0)

    def test_direction_change_recalculates_interception(self) -> None:
        simulator = self.make_simulator(robot_speed=150.0)
        simulator.step(
            estimate(0.0, (200.0, 180.0), (-20.0, 0.0)),
            0.0,
            True,
            320,
            240,
        )
        before = simulator.snapshot.interception_point
        changed = simulator.step(
            estimate(0.1, (198.0, 180.0), (20.0, 0.0)),
            0.1,
            True,
            320,
            240,
        )
        self.assertIsNotNone(before)
        self.assertIsNotNone(changed.interception_point)
        self.assertNotEqual(changed.interception_point, before)

    def test_closed_loop_verifies_interception_at_same_time(self) -> None:
        simulator = self.make_simulator(robot_speed=100.0, tolerance=8.0)
        snapshot = simulator.step(estimate(0.0, (150.0, 180.0)), 0.0, True, 320, 240)
        timestamp = 0.0
        while timestamp < 5.0 and snapshot.status != InterceptionStatus.INTERCEPTED:
            timestamp = round(timestamp + 1.0 / 30.0, 8)
            snapshot = simulator.step(
                estimate(timestamp, (150.0, 180.0)),
                timestamp,
                True,
                320,
                240,
            )
        self.assertEqual(snapshot.status, InterceptionStatus.INTERCEPTED)
        self.assertIsNotNone(snapshot.success_time)
        self.assertLessEqual(snapshot.distance_pixels, 8.0)
        self.assertLessEqual(
            math.dist(snapshot.robot_position, (150.0, 180.0)),
            8.0,
        )

    def test_interception_against_moving_target(self) -> None:
        simulator = self.make_simulator(robot_speed=120.0, tolerance=10.0)
        timestamp = 0.0
        snapshot = simulator.step(
            estimate(timestamp, (220.0, 180.0), (-15.0, 0.0)),
            timestamp,
            True,
            320,
            240,
        )
        while timestamp < 6.0 and snapshot.status != InterceptionStatus.INTERCEPTED:
            timestamp = round(timestamp + 1.0 / 30.0, 8)
            target_position = (220.0 - 15.0 * timestamp, 180.0)
            snapshot = simulator.step(
                estimate(timestamp, target_position, (-15.0, 0.0)),
                timestamp,
                True,
                320,
                240,
            )
        self.assertEqual(snapshot.status, InterceptionStatus.INTERCEPTED)
        self.assertLessEqual(snapshot.distance_pixels, 10.0)

    def test_synthetic_interception_accuracy_metrics(self) -> None:
        scenarios = (
            ((180.0, 50.0), (0.0, 0.0)),
            ((240.0, 130.0), (-30.0, 0.0)),
            ((250.0, 160.0), (0.0, -20.0)),
        )
        successes = 0
        position_errors: list[float] = []
        time_errors: list[float] = []
        for scenario_index, (initial_position, velocity) in enumerate(scenarios, start=1):
            simulator = self.make_simulator(
                robot_speed=180.0,
                tolerance=10.0,
                response_delay=0.1,
                max_horizon=5.0,
            )
            initial_robot = (160.0, 216.0)
            expected = calculate_interception(
                initial_robot,
                initial_position,
                velocity,
                180.0,
                response_delay=0.1,
                max_horizon=5.0,
            )
            self.assertIsNotNone(expected)
            timestamp = 0.0
            snapshot = simulator.step(
                estimate(timestamp, initial_position, velocity, track_id=scenario_index),
                timestamp,
                True,
                320,
                240,
            )
            while timestamp < 5.0 and snapshot.status != InterceptionStatus.INTERCEPTED:
                timestamp = round(timestamp + 1.0 / 30.0, 8)
                target_position = (
                    initial_position[0] + velocity[0] * timestamp,
                    initial_position[1] + velocity[1] * timestamp,
                )
                snapshot = simulator.step(
                    estimate(
                        timestamp,
                        target_position,
                        velocity,
                        track_id=scenario_index,
                    ),
                    timestamp,
                    True,
                    320,
                    240,
                )
            if snapshot.status == InterceptionStatus.INTERCEPTED:
                successes += 1
                position_errors.append(snapshot.distance_pixels)
                time_errors.append(abs(snapshot.success_time - expected.time_seconds))

        impossible_plan = calculate_interception(
            (160.0, 216.0),
            (20.0, 200.0),
            (-100.0, 0.0),
            50.0,
        )
        correctly_impossible = int(impossible_plan is None)
        self.assertEqual(successes, len(scenarios))
        self.assertEqual(correctly_impossible, 1)
        self.assertLessEqual(max(position_errors), 10.0)
        print(
            "synthetic interception: "
            f"success rate={successes}/{len(scenarios)}, "
            f"mean position error={sum(position_errors) / len(position_errors):.2f}px, "
            f"mean time error={sum(time_errors) / len(time_errors):.3f}s, "
            f"impossible correctly identified={correctly_impossible}/1"
        )

    def test_disable_and_pause_do_not_reset_tracking_estimates(self) -> None:
        simulator = self.make_simulator()
        active = simulator.step(estimate(0.0, (200.0, 180.0)), 0.0, True, 320, 240)
        paused = simulator.step(
            estimate(0.1, (200.0, 180.0)),
            0.1,
            True,
            320,
            240,
            paused=True,
        )
        self.assertEqual(paused, active)
        idle = simulator.step(None, 0.2, False, 320, 240)
        self.assertEqual(idle.status, InterceptionStatus.IDLE)
        self.assertEqual(idle.robot_position, active.robot_position)


if __name__ == "__main__":
    unittest.main()
