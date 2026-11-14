"""Variable-time-step constant-velocity Kalman filter for image coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KalmanEstimate:
    position: tuple[float, float]
    velocity: tuple[float, float]
    residual_magnitude: float
    timestamp: float


class ConstantVelocityKalman:
    """Estimate [x, y, vx, vy] from noisy [x, y] pixel measurements.

    The transition matrix uses the actual time between observations:
    ``F(dt) = [[1,0,dt,0], [0,1,0,dt], [0,0,1,0], [0,0,0,1]]``.
    Process noise models independent white acceleration on each axis with
    variance ``q``; each axis therefore receives the standard
    ``q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]`` covariance block. Measurement
    noise is isotropic with covariance ``R = r * I``.
    """

    def __init__(
        self,
        measurement_noise: float,
        process_noise: float,
        initial_velocity_variance: float = 10_000.0,
    ) -> None:
        if measurement_noise <= 0:
            raise ValueError("measurement_noise must be positive")
        if process_noise < 0:
            raise ValueError("process_noise must not be negative")
        if initial_velocity_variance <= 0:
            raise ValueError("initial_velocity_variance must be positive")
        self.measurement_noise = measurement_noise
        self.process_noise = process_noise
        self.initial_velocity_variance = initial_velocity_variance
        self._state: FloatArray | None = None
        self._covariance: FloatArray | None = None
        self._timestamp: float | None = None
        self._measurement = np.zeros((2, 4), dtype=np.float64)
        self._measurement[0, 0] = 1.0
        self._measurement[1, 1] = 1.0
        self._measurement_covariance = (
            np.eye(2, dtype=np.float64) * measurement_noise
        )
        self._identity = np.eye(4, dtype=np.float64)

    @property
    def initialized(self) -> bool:
        return self._state is not None

    @property
    def timestamp(self) -> float | None:
        return self._timestamp

    def reset(self) -> None:
        self._state = None
        self._covariance = None
        self._timestamp = None

    def update(self, position: tuple[float, float], timestamp: float) -> KalmanEstimate:
        """Predict to ``timestamp`` and incorporate a measured pixel center."""
        measurement = np.asarray(position, dtype=np.float64)
        if measurement.shape != (2,) or not np.isfinite(measurement).all():
            raise ValueError("position must contain two finite coordinates")
        if self._timestamp is not None and timestamp <= self._timestamp:
            raise ValueError("timestamp must increase between measurements")

        if not self.initialized:
            self._state = np.array(
                [measurement[0], measurement[1], 0.0, 0.0],
                dtype=np.float64,
            )
            self._covariance = np.diag(
                [
                    self.measurement_noise,
                    self.measurement_noise,
                    self.initial_velocity_variance,
                    self.initial_velocity_variance,
                ]
            )
            self._timestamp = timestamp
            return self.estimate(0.0)

        if self._state is None or self._covariance is None or self._timestamp is None:
            raise RuntimeError("Kalman state was not initialized")
        dt = timestamp - self._timestamp
        transition = self._transition_matrix(dt)
        process_covariance = self._process_covariance(dt)

        predicted_state = transition @ self._state
        predicted_covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )
        residual = measurement - self._measurement @ predicted_state
        innovation_covariance = (
            self._measurement @ predicted_covariance @ self._measurement.T
            + self._measurement_covariance
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            self._measurement @ predicted_covariance,
        ).T

        self._state = predicted_state + kalman_gain @ residual
        residual_projection = self._identity - kalman_gain @ self._measurement
        self._covariance = (
            residual_projection
            @ predicted_covariance
            @ residual_projection.T
            + kalman_gain
            @ self._measurement_covariance
            @ kalman_gain.T
        )
        self._timestamp = timestamp
        return self.estimate(float(np.linalg.norm(residual)))

    def predict(
        self,
        horizon_seconds: float,
    ) -> tuple[float, float]:
        """Return a future image-space position without mutating filter state."""
        if horizon_seconds < 0:
            raise ValueError("horizon_seconds must not be negative")
        if self._state is None or self._timestamp is None:
            raise RuntimeError("filter has no measurements")
        future_state = self._transition_matrix(horizon_seconds) @ self._state
        return float(future_state[0]), float(future_state[1])

    def estimate(self, residual_magnitude: float = 0.0) -> KalmanEstimate:
        if self._state is None or self._timestamp is None:
            raise RuntimeError("filter has no measurements")
        return KalmanEstimate(
            position=(float(self._state[0]), float(self._state[1])),
            velocity=(float(self._state[2]), float(self._state[3])),
            residual_magnitude=residual_magnitude,
            timestamp=self._timestamp,
        )

    @staticmethod
    def _transition_matrix(dt: float) -> FloatArray:
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _process_covariance(self, dt: float) -> FloatArray:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        block = self.process_noise * np.array(
            [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]],
            dtype=np.float64,
        )
        covariance = np.zeros((4, 4), dtype=np.float64)
        covariance[np.ix_([0, 2], [0, 2])] = block
        covariance[np.ix_([1, 3], [1, 3])] = block
        return covariance
