"""Image-space interception planning and a bounded-speed 2D robot simulator."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import ceil, copysign, hypot, isfinite, sqrt

import config
from prediction.motion import MotionEstimate, MotionSample, PredictionStatus

Point = tuple[float, float]


class InterceptionStatus(str, Enum):
    IDLE = "IDLE"
    INTERCEPTED = "INTERCEPTED"
    APPROACHING = "APPROACHING"
    POSSIBLE = "INTERCEPTION POSSIBLE"
    IMPOSSIBLE = "INTERCEPTION IMPOSSIBLE"
    TARGET_LOST = "TARGET LOST"
    LOW_CONFIDENCE = "LOW CONFIDENCE"


@dataclass(frozen=True)
class InterceptionPlan:
    """Earliest feasible meet point, with time measured from the current instant."""

    time_seconds: float
    position: Point


@dataclass(frozen=True)
class SimulationSnapshot:
    """Immutable state for rendering and inspecting one simulation update."""

    status: InterceptionStatus
    simulation_time: float
    width: int
    height: int
    robot_start: Point
    robot_position: Point
    robot_path: tuple[Point, ...]
    target_id: int | None
    target_position: Point | None
    target_velocity: Point | None
    target_history: tuple[MotionSample, ...]
    predicted_trajectory: tuple[Point, ...]
    interception_point: Point | None
    interception_time: float | None
    distance_pixels: float | None
    response_delay_remaining: float
    success_time: float | None
    success_position: Point | None


def calculate_interception(
    robot_position: Point,
    target_position: Point,
    target_velocity: Point,
    robot_speed: float,
    response_delay: float = 0.0,
    max_horizon: float = config.ROBOTICS_MAX_INTERCEPTION_HORIZON_SECONDS,
) -> InterceptionPlan | None:
    """Solve for the earliest time a speed-limited robot can meet a linear target.

    For ``tau = t - response_delay``, solve
    ``|target_position + velocity*t - robot_position| = robot_speed*tau``.
    The quadratic is solved for its earliest non-negative real root. ``None``
    means no exact meet is reachable within the configured horizon.
    """
    values = (*robot_position, *target_position, *target_velocity, robot_speed, response_delay, max_horizon)
    if not all(isfinite(value) for value in values):
        raise ValueError("interception inputs must be finite")
    if robot_speed <= 0.0 or response_delay < 0.0 or max_horizon <= 0.0:
        raise ValueError("speed and horizon must be positive; delay must not be negative")

    relative_x = target_position[0] + target_velocity[0] * response_delay - robot_position[0]
    relative_y = target_position[1] + target_velocity[1] * response_delay - robot_position[1]
    vx, vy = target_velocity
    a = vx * vx + vy * vy - robot_speed * robot_speed
    b = 2.0 * (relative_x * vx + relative_y * vy)
    c = relative_x * relative_x + relative_y * relative_y
    epsilon = 1e-10

    if c <= epsilon:
        travel_time = 0.0
    elif abs(a) <= epsilon:
        if b >= -epsilon:
            return None
        travel_time = -c / b
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < -epsilon:
            return None
        root = sqrt(max(0.0, discriminant))
        q = -0.5 * (b + copysign(root, b))
        roots = (q / a, c / q) if abs(q) > epsilon else (-b / (2.0 * a),)
        feasible = [value for value in roots if value >= -epsilon]
        if not feasible:
            return None
        travel_time = max(0.0, min(feasible))

    intercept_time = response_delay + travel_time
    if intercept_time > max_horizon:
        return None
    return InterceptionPlan(
        time_seconds=intercept_time,
        position=(
            target_position[0] + vx * intercept_time,
            target_position[1] + vy * intercept_time,
        ),
    )


class InterceptionSimulator:
    """Run a software-only point-effector controller against a selected target."""

    def __init__(
        self,
        robot_speed: float = config.ROBOTICS_MAX_SPEED_PIXELS_PER_SECOND,
        response_delay: float = config.ROBOTICS_RESPONSE_DELAY_SECONDS,
        tolerance: float = config.ROBOTICS_INTERCEPTION_TOLERANCE_PIXELS,
        max_horizon: float = config.ROBOTICS_MAX_INTERCEPTION_HORIZON_SECONDS,
        minimum_measurements: int = config.ROBOTICS_MIN_PREDICTION_MEASUREMENTS,
        controller_gain: float = config.ROBOTICS_CONTROLLER_GAIN,
        update_rate_hz: float = config.ROBOTICS_UPDATE_RATE_HZ,
        minimum_confidence_status: tuple[PredictionStatus, ...] = (
            PredictionStatus.TRACKING,
            PredictionStatus.PREDICTING,
        ),
        history_length: int = config.ROBOTICS_PATH_HISTORY_LENGTH,
    ) -> None:
        if (
            robot_speed <= 0.0
            or response_delay < 0.0
            or tolerance <= 0.0
            or controller_gain <= 0.0
            or update_rate_hz <= 0.0
        ):
            raise ValueError("speed, tolerance, gain, and update rate must be positive")
        if max_horizon <= 0.0 or minimum_measurements < 1 or history_length < 2:
            raise ValueError("horizon, measurement count, and history length must be positive")
        self.robot_speed = robot_speed
        self.response_delay = response_delay
        self.tolerance = tolerance
        self.max_horizon = max_horizon
        self.minimum_measurements = minimum_measurements
        self.controller_gain = controller_gain
        self.update_rate_hz = update_rate_hz
        self.minimum_confidence_status = minimum_confidence_status
        self._robot_fraction = config.ROBOTICS_START_POSITION_RATIO
        self._robot_start: Point = (0.0, 0.0)
        self._robot_position: Point = (0.0, 0.0)
        self._robot_path: deque[Point] = deque(maxlen=history_length)
        self._target_history_length = config.ROBOTICS_TARGET_HISTORY_LENGTH
        self._target_history: deque[MotionSample] = deque(
            maxlen=self._target_history_length
        )
        self._width = 1
        self._height = 1
        self._simulation_time = 0.0
        self._last_timestamp: float | None = None
        self._enabled = False
        self._target_id: int | None = None
        self._last_target_id: int | None = None
        self._response_remaining = 0.0
        self._plan: InterceptionPlan | None = None
        self._intercepted = False
        self._success_time: float | None = None
        self._success_position: Point | None = None
        self._snapshot = self._make_snapshot(
            InterceptionStatus.IDLE, None, None, None, (), None
        )

    @property
    def snapshot(self) -> SimulationSnapshot:
        return self._snapshot

    def reset(self) -> None:
        """Return the robot and simulation clock to their initial state."""
        self._robot_start = self._initial_position(self._width, self._height)
        self._robot_position = self._robot_start
        self._robot_path.clear()
        self._robot_path.append(self._robot_start)
        self._target_history.clear()
        self._simulation_time = 0.0
        self._last_timestamp = None
        self._enabled = False
        self._target_id = None
        self._last_target_id = None
        self._response_remaining = 0.0
        self._plan = None
        self._intercepted = False
        self._success_time = None
        self._success_position = None
        self._snapshot = self._make_snapshot(
            InterceptionStatus.IDLE, None, None, None, (), None
        )

    def step(
        self,
        estimate: MotionEstimate | None,
        timestamp: float,
        enabled: bool,
        width: int,
        height: int,
        paused: bool = False,
    ) -> SimulationSnapshot:
        """Advance with real elapsed time and the latest filtered target estimate."""
        if not isfinite(timestamp) or width < 1 or height < 1:
            raise ValueError("timestamp and simulation dimensions must be valid")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("simulation timestamp must not move backwards")

        if (width, height) != (self._width, self._height):
            self._width, self._height = width, height
            if not self._robot_path:
                self._robot_start = self._initial_position(width, height)
                self._robot_position = self._robot_start
                self._robot_path.append(self._robot_start)
            else:
                self._robot_position = self._clamp(self._robot_position)

        if self._last_timestamp is None:
            elapsed = 0.0
            self._robot_start = self._initial_position(width, height)
            self._robot_position = self._robot_start
            self._robot_path.clear()
            self._robot_path.append(self._robot_start)
        else:
            elapsed = timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        if not enabled:
            if not paused:
                self._simulation_time += elapsed
            self._enabled = False
            self._target_id = None
            self._plan = None
            self._intercepted = False
            self._response_remaining = 0.0
            self._snapshot = self._make_snapshot(
                InterceptionStatus.IDLE, None, None, None, (), None
            )
            return self._snapshot

        if paused:
            return self._snapshot
        self._simulation_time += elapsed

        if not self._enabled:
            self._enabled = True
            self._target_id = None
            self._response_remaining = self.response_delay
            self._plan = None
            self._intercepted = False

        if estimate is None:
            if self._target_id is not None:
                self._last_target_id = self._target_id
            self._target_id = None
            self._plan = None
            self._intercepted = False
            self._snapshot = self._make_snapshot(
                InterceptionStatus.TARGET_LOST,
                None,
                None,
                None,
                tuple(self._target_history),
                None,
            )
            return self._snapshot

        if estimate.track_id != self._target_id:
            previous_target_id = (
                self._target_id
                if self._target_id is not None
                else self._last_target_id
            )
            if previous_target_id != estimate.track_id:
                self._target_history.clear()
            self._last_target_id = self._target_id
            self._target_id = estimate.track_id
            self._response_remaining = self.response_delay
            self._plan = None
            self._intercepted = False
            self._success_time = None
            self._success_position = None

        target_position = self._current_position(estimate, timestamp)
        target_velocity = estimate.velocity
        if (
            not self._target_history
            or timestamp > self._target_history[-1].timestamp
        ):
            self._target_history.append(
                MotionSample(timestamp=timestamp, position=target_position)
            )
        if (
            estimate.status not in self.minimum_confidence_status
            or estimate.measurement_count < self.minimum_measurements
            or estimate.seconds_since_measurement > config.ROBOTICS_MAX_MEASUREMENT_AGE_SECONDS
        ):
            self._plan = None
            self._snapshot = self._make_snapshot(
                InterceptionStatus.LOW_CONFIDENCE,
                estimate.track_id,
                target_position,
                target_velocity,
                tuple(self._target_history),
                None,
            )
            return self._snapshot

        if self._intercepted:
            self._snapshot = self._make_snapshot(
                InterceptionStatus.INTERCEPTED,
                estimate.track_id,
                target_position,
                target_velocity,
                tuple(self._target_history),
                self._plan,
            )
            return self._snapshot

        status = InterceptionStatus.POSSIBLE
        steps = min(
            max(1, ceil(elapsed * self.update_rate_hz)),
            config.ROBOTICS_MAX_CATCHUP_STEPS,
        )
        substep = elapsed / steps
        self._plan = None
        for index in range(steps):
            step_start_target = (
                target_position[0] - target_velocity[0] * (elapsed - index * substep),
                target_position[1] - target_velocity[1] * (elapsed - index * substep),
            )
            plan = calculate_interception(
                self._robot_position,
                step_start_target,
                target_velocity,
                self.robot_speed,
                self._response_remaining,
                self.max_horizon,
            )
            if plan is None or not self._inside_environment(plan.position):
                self._plan = None
                status = InterceptionStatus.IMPOSSIBLE
                break

            self._plan = plan
            if self._intercepted:
                status = InterceptionStatus.INTERCEPTED
                continue

            response_before_step = self._response_remaining
            move_elapsed = max(0.0, substep - response_before_step)
            self._response_remaining = max(
                0.0, response_before_step - substep
            )
            destination_distance = hypot(
                plan.position[0] - self._robot_position[0],
                plan.position[1] - self._robot_position[1],
            )
            controller_distance = min(
                self.robot_speed * move_elapsed,
                self.controller_gain * destination_distance,
            )
            if controller_distance > 0.0:
                self._move_toward(plan.position, controller_distance)

            target_at_end = (
                step_start_target[0] + target_velocity[0] * substep,
                step_start_target[1] + target_velocity[1] * substep,
            )
            distance = hypot(
                self._robot_position[0] - target_at_end[0],
                self._robot_position[1] - target_at_end[1],
            )
            if distance <= self.tolerance:
                self._intercepted = True
                self._success_time = (
                    self._simulation_time - elapsed + (index + 1) * substep
                )
                self._success_position = (
                    (self._robot_position[0] + target_at_end[0]) / 2.0,
                    (self._robot_position[1] + target_at_end[1]) / 2.0,
                )
                status = InterceptionStatus.INTERCEPTED
            elif move_elapsed > 0.0:
                status = InterceptionStatus.APPROACHING
            else:
                status = InterceptionStatus.POSSIBLE

        self._snapshot = self._make_snapshot(
            status,
            estimate.track_id,
            target_position,
            target_velocity,
            tuple(self._target_history),
            self._plan if status != InterceptionStatus.IMPOSSIBLE else None,
        )
        return self._snapshot

    def _move_toward(self, destination: Point, max_distance: float) -> None:
        dx = destination[0] - self._robot_position[0]
        dy = destination[1] - self._robot_position[1]
        distance = hypot(dx, dy)
        if distance <= 1e-9:
            return
        travel = min(distance, max_distance)
        self._robot_position = self._clamp(
            (
                self._robot_position[0] + dx * travel / distance,
                self._robot_position[1] + dy * travel / distance,
            )
        )
        self._robot_path.append(self._robot_position)

    def _current_position(self, estimate: MotionEstimate, timestamp: float) -> Point:
        age = max(0.0, min(timestamp - estimate.timestamp, self.max_horizon))
        return (
            estimate.filtered_position[0] + estimate.velocity[0] * age,
            estimate.filtered_position[1] + estimate.velocity[1] * age,
        )

    def _initial_position(self, width: int, height: int) -> Point:
        return (
            max(0.0, min(width - 1.0, width * self._robot_fraction[0])),
            max(0.0, min(height - 1.0, height * self._robot_fraction[1])),
        )

    def _clamp(self, point: Point) -> Point:
        return (
            max(0.0, min(self._width - 1.0, point[0])),
            max(0.0, min(self._height - 1.0, point[1])),
        )

    def _inside_environment(self, point: Point) -> bool:
        return 0.0 <= point[0] < self._width and 0.0 <= point[1] < self._height

    def _make_snapshot(
        self,
        status: InterceptionStatus,
        target_id: int | None,
        target_position: Point | None,
        target_velocity: Point | None,
        target_history: tuple[MotionSample, ...],
        plan: InterceptionPlan | None,
    ) -> SimulationSnapshot:
        distance = (
            hypot(
                self._robot_position[0] - target_position[0],
                self._robot_position[1] - target_position[1],
            )
            if target_position is not None
            else None
        )
        trajectory: tuple[Point, ...] = ()
        if target_position is not None and target_velocity is not None:
            trajectory = tuple(
                (
                    target_position[0] + target_velocity[0] * horizon,
                    target_position[1] + target_velocity[1] * horizon,
                )
                for horizon in config.ROBOTICS_TRAJECTORY_HORIZONS_SECONDS
            )
        return SimulationSnapshot(
            status=status,
            simulation_time=self._simulation_time,
            width=self._width,
            height=self._height,
            robot_start=self._robot_start,
            robot_position=self._robot_position,
            robot_path=tuple(self._robot_path),
            target_id=target_id if target_id is not None else self._last_target_id,
            target_position=target_position,
            target_velocity=target_velocity,
            target_history=target_history[-self._target_history_length :],
            predicted_trajectory=trajectory,
            interception_point=plan.position if plan is not None else None,
            interception_time=plan.time_seconds if plan is not None else None,
            distance_pixels=distance,
            response_delay_remaining=self._response_remaining,
            success_time=self._success_time,
            success_position=self._success_position,
        )
