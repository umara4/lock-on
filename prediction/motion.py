"""Per-track motion estimation and constant-velocity future predictions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees

import numpy as np

import config
from prediction.kalman import ConstantVelocityKalman, KalmanEstimate
from vision.tracker import TrackedObject


class PredictionStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    TRACKING = "TRACKING"
    PREDICTING = "PREDICTING"
    LOW_CONFIDENCE = "LOW CONFIDENCE"
    LOST = "LOST"


@dataclass(frozen=True)
class MotionSample:
    timestamp: float
    position: tuple[float, float]


@dataclass(frozen=True)
class MotionEstimate:
    track_id: int
    timestamp: float
    measured_position: tuple[float, float]
    position_history: tuple[MotionSample, ...]
    filtered_position: tuple[float, float]
    velocity: tuple[float, float]
    speed: float
    acceleration: tuple[float, float]
    direction_degrees: float | None
    stationary: bool
    predictions: dict[float, tuple[float, float]]
    status: PredictionStatus
    measurement_count: int
    residual_pixels: float
    seconds_since_measurement: float
    direction_changed: bool
    velocity_clamped: bool


@dataclass
class _MotionState:
    label: str
    kalman: ConstantVelocityKalman
    last_seen: float
    measurement_count: int = 0
    acceleration: tuple[float, float] = (0.0, 0.0)
    residuals: deque[float] | None = None
    positions: deque[MotionSample] | None = None
    estimate: MotionEstimate | None = None


class MotionEstimator:
    """Track independent Kalman state, velocity, and reliability per track ID."""

    def __init__(
        self,
        horizons: tuple[float, ...] = config.PREDICTION_HORIZONS_SECONDS,
        measurement_noise: float = config.KALMAN_MEASUREMENT_NOISE,
        process_noise: float = config.KALMAN_PROCESS_NOISE,
        max_speed: float = config.MOTION_MAX_SPEED_PIXELS_PER_SECOND,
        stationary_speed: float = config.MOTION_STATIONARY_SPEED_PIXELS_PER_SECOND,
        max_lost_seconds: float = config.MOTION_MAX_LOST_SECONDS,
        low_confidence_residual: float = config.MOTION_LOW_CONFIDENCE_RESIDUAL_PIXELS,
        history_length: int = config.MOTION_HISTORY_LENGTH,
    ) -> None:
        if not horizons or any(horizon <= 0 for horizon in horizons):
            raise ValueError("prediction horizons must be positive")
        if max_speed <= 0 or stationary_speed < 0 or max_lost_seconds <= 0:
            raise ValueError("motion limits must be positive (stationary speed may be zero)")
        if history_length < 2:
            raise ValueError("history_length must be at least 2")
        self.horizons = tuple(sorted(set(float(horizon) for horizon in horizons)))
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.max_speed = max_speed
        self.stationary_speed = stationary_speed
        self.max_lost_seconds = max_lost_seconds
        self.low_confidence_residual = low_confidence_residual
        self.history_length = history_length
        self._states: dict[int, _MotionState] = {}

    @property
    def estimates(self) -> dict[int, MotionEstimate]:
        """Return the current immutable estimate for every retained track."""
        return {
            track_id: state.estimate
            for track_id, state in self._states.items()
            if state.estimate is not None
        }

    def update(
        self,
        tracks: list[TrackedObject],
        timestamp: float,
        paused: bool = False,
    ) -> dict[int, MotionEstimate]:
        """Update measured tracks and expire objects lost for too long."""
        if paused:
            return self.estimates

        seen_ids: set[int] = set()
        for track in tracks:
            seen_ids.add(track.track_id)
            state = self._states.get(track.track_id)
            if state is not None and (
                state.label != track.label
                or timestamp - state.last_seen > self.max_lost_seconds
            ):
                del self._states[track.track_id]
                state = None

            if state is None:
                state = self._new_state(track.label, timestamp)
                self._states[track.track_id] = state

            self._update_track(state, track, timestamp)

        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if track_id not in seen_ids
            and timestamp - state.last_seen > self.max_lost_seconds
        ]
        for track_id in stale_ids:
            del self._states[track_id]

        for track_id, state in self._states.items():
            if track_id not in seen_ids and state.estimate is not None:
                state.estimate = self._with_status(
                    state.estimate,
                    PredictionStatus.LOST
                    if timestamp - state.last_seen
                    >= config.MOTION_LOST_STATUS_SECONDS
                    else state.estimate.status,
                    timestamp - state.last_seen,
                )
        return self.estimates

    def reset(self) -> None:
        """Clear all per-ID filters and estimates."""
        self._states.clear()

    def _new_state(self, label: str, timestamp: float) -> _MotionState:
        return _MotionState(
            label=label,
            kalman=ConstantVelocityKalman(
                self.measurement_noise,
                self.process_noise,
                config.KALMAN_INITIAL_VELOCITY_VARIANCE,
            ),
            last_seen=timestamp,
            residuals=deque(maxlen=self.history_length),
        )

    def _update_track(
        self,
        state: _MotionState,
        track: TrackedObject,
        timestamp: float,
    ) -> None:
        previous_estimate = state.estimate
        kalman_estimate = state.kalman.update(track.center, timestamp)
        state.measurement_count += 1
        state.last_seen = timestamp
        if state.residuals is None:
            state.residuals = deque(maxlen=self.history_length)
        if state.positions is None:
            state.positions = deque(maxlen=self.history_length)
        previous_position = state.positions[-1] if state.positions else None
        state.positions.append(
            MotionSample(
                timestamp=timestamp,
                position=(float(track.center[0]), float(track.center[1])),
            )
        )
        state.residuals.append(kalman_estimate.residual_magnitude)

        raw_velocity = np.asarray(kalman_estimate.velocity, dtype=np.float64)
        raw_speed = float(np.linalg.norm(raw_velocity))
        velocity_clamped = raw_speed > self.max_speed
        if velocity_clamped:
            raw_velocity *= self.max_speed / raw_speed
        velocity = (float(raw_velocity[0]), float(raw_velocity[1]))
        speed = float(np.linalg.norm(raw_velocity))

        direction_changed = False
        if previous_estimate is not None:
            previous_velocity = np.asarray(previous_estimate.velocity)
            previous_speed = float(np.linalg.norm(previous_velocity))
            interval = (
                timestamp - previous_position.timestamp
                if previous_position is not None
                else 0.0
            )
            observed_velocity = np.zeros(2, dtype=np.float64)
            if interval > 1e-6 and previous_position is not None:
                observed_velocity = (
                    np.asarray(track.center, dtype=np.float64)
                    - np.asarray(previous_position.position, dtype=np.float64)
                ) / interval
            observed_speed = float(np.linalg.norm(observed_velocity))
            if (
                previous_speed >= self.stationary_speed
                and observed_speed >= self.stationary_speed
            ):
                cosine = float(
                    np.dot(previous_velocity, observed_velocity)
                    / (previous_speed * observed_speed)
                )
                direction_changed = cosine < config.MOTION_DIRECTION_CHANGE_COSINE

            dt = timestamp - previous_estimate.timestamp
            if dt > 1e-6:
                acceleration_array = (
                    np.asarray(velocity) - np.asarray(previous_estimate.velocity)
                ) / dt
                state.acceleration = (
                    float(acceleration_array[0]),
                    float(acceleration_array[1]),
                )

        average_residual = (
            float(np.mean(state.residuals)) if state.residuals else 0.0
        )
        status = self._status(
            state.measurement_count,
            speed,
            average_residual,
            direction_changed,
        )
        predictions = {
            horizon: self._predict(kalman_estimate, velocity, horizon)
            for horizon in self.horizons
        }
        direction = degrees(atan2(velocity[1], velocity[0])) if speed > 1e-6 else None
        state.estimate = MotionEstimate(
            track_id=track.track_id,
            timestamp=timestamp,
            measured_position=(float(track.center[0]), float(track.center[1])),
            position_history=tuple(state.positions),
            filtered_position=kalman_estimate.position,
            velocity=velocity,
            speed=speed,
            acceleration=state.acceleration,
            direction_degrees=direction,
            stationary=speed < self.stationary_speed,
            predictions=predictions,
            status=status,
            measurement_count=state.measurement_count,
            residual_pixels=average_residual,
            seconds_since_measurement=0.0,
            direction_changed=direction_changed,
            velocity_clamped=velocity_clamped,
        )

    def _predict(
        self,
        estimate: KalmanEstimate,
        velocity: tuple[float, float],
        horizon: float,
    ) -> tuple[float, float]:
        return (
            estimate.position[0] + velocity[0] * horizon,
            estimate.position[1] + velocity[1] * horizon,
        )

    def _status(
        self,
        measurement_count: int,
        speed: float,
        average_residual: float,
        direction_changed: bool,
    ) -> PredictionStatus:
        if measurement_count < config.MOTION_MIN_PREDICTION_MEASUREMENTS:
            return PredictionStatus.INITIALIZING
        if (
            average_residual > self.low_confidence_residual
            or direction_changed
        ):
            return PredictionStatus.LOW_CONFIDENCE
        if speed < self.stationary_speed:
            return PredictionStatus.TRACKING
        return PredictionStatus.PREDICTING

    @staticmethod
    def _with_status(
        estimate: MotionEstimate,
        status: PredictionStatus,
        seconds_since_measurement: float,
    ) -> MotionEstimate:
        return MotionEstimate(
            track_id=estimate.track_id,
            timestamp=estimate.timestamp,
            measured_position=estimate.measured_position,
            position_history=estimate.position_history,
            filtered_position=estimate.filtered_position,
            velocity=estimate.velocity,
            speed=estimate.speed,
            acceleration=estimate.acceleration,
            direction_degrees=estimate.direction_degrees,
            stationary=estimate.stationary,
            predictions=estimate.predictions,
            status=status,
            measurement_count=estimate.measurement_count,
            residual_pixels=estimate.residual_pixels,
            seconds_since_measurement=seconds_since_measurement,
            direction_changed=estimate.direction_changed,
            velocity_clamped=estimate.velocity_clamped,
        )
