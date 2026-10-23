from dataclasses import dataclass

import numpy as np

from wavetrace.domain.localize import Localization


@dataclass
class TrackState:
    """One filtered step of a moving target. The filter fuses PREDICTION (motion) with the MEASUREMENT
    (MUSIC), so the track is continuous and `measured=False` marks a frame whose measurement was gated
    out as an impossible jump (the track coasted on its motion model)."""

    timestamp: float
    angle_deg: float
    range_m: float        # filtered range (nan when range tracking is off)
    angular_rate: float   # deg/s
    radial_rate: float    # m/s (0 when range off)
    x_m: float
    y_m: float
    measured: bool        # True if the measurement passed the gate and updated the state
    confidence: float     # the measurement confidence that set the Kalman gain this step


class Tracker:
    """Constant-velocity Kalman filter over (angle, range); state = [angle, range, d_angle, d_range].
    Measurement noise shrinks with confidence (floored at confidence_floor to avoid infinite trust);
    a χ² innovation gate (9.0 ≈ 98.9% at 2 DOF) rejects impossible jumps. angle_std / range_std are
    in deg / m; the *_acceleration_std process-noise bounds are in deg/s² / m/s². O(1) per step (4×4)."""

    def __init__(self, *, angle_std=3.0, range_std=4.0, angle_acceleration_std=40.0,
                 range_acceleration_std=3.0, confidence_floor=0.05, gate=9.0, range_enabled=True):
        self._angle_variance = float(angle_std) ** 2
        self._range_variance = float(range_std) ** 2
        self._angle_process_variance = float(angle_acceleration_std) ** 2
        self._range_process_variance = float(range_acceleration_std) ** 2
        self._confidence_floor = float(confidence_floor)
        self._gate = float(gate)
        self._is_range_enabled = bool(range_enabled)
        self._x = None          # state x = [angle, range, d_angle, d_range], set by the first measurement
        self._P = None          # state covariance P (4, 4)
        self._last_timestamp_s = None

    def update(self, localization: Localization) -> TrackState:
        measured_angle_deg = float(localization.peak_angle_deg)
        measured_range_m = float(localization.peak_range_m)
        has_range = self._is_range_enabled and not np.isnan(measured_range_m)
        confidence = float(localization.confidence)
        timestamp_s = float(localization.timestamp)

        if self._x is None:
            self._x = np.array([measured_angle_deg, (measured_range_m if has_range else 0.0), 0.0, 0.0])
            self._P = np.diag([self._angle_variance, self._range_variance, 1e3, 1e3])
            self._last_timestamp_s = timestamp_s
            return self._emit(timestamp_s, confidence, measured=True, has_range=has_range)

        time_step_s = max(timestamp_s - self._last_timestamp_s, 1e-3)
        self._last_timestamp_s = timestamp_s
        # predict: x = F·x, P = F·P·F^T + Q, F = [[I, Δt·I], [0, I]] (constant velocity; Q = the
        # max-acceleration motion bound)
        F = np.array([[1, 0, time_step_s, 0], [0, 1, 0, time_step_s], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(time_step_s)

        # update: z = H·x + noise, R = σ² / confidence (angle, + range when available), so higher
        # confidence -> smaller R -> higher Kalman gain
        confidence_factor = max(confidence, self._confidence_floor)
        if has_range:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
            z = np.array([measured_angle_deg, measured_range_m])
            R = np.diag([self._angle_variance / confidence_factor, self._range_variance / confidence_factor])
        else:
            H = np.array([[1, 0, 0, 0]], float)
            z = np.array([measured_angle_deg])
            R = np.array([[self._angle_variance / confidence_factor]])
        y = z - H @ self._x                     # innovation y = z - H·x (measurement - prediction)
        S = H @ self._P @ H.T + R               # innovation covariance S = H·P·H^T + R
        S_inv = np.linalg.inv(S)
        mahalanobis_squared = float(y @ S_inv @ y)
        measured = mahalanobis_squared <= self._gate
        if measured:
            K = self._P @ H.T @ S_inv            # Kalman gain K = P·H^T·S⁻¹ (the confidence-weighted blend)
            self._x = self._x + K @ y
            self._P = (np.eye(4) - K @ H) @ self._P
        return self._emit(timestamp_s, confidence, measured=measured, has_range=has_range)

    def run(self, localizations) -> list[TrackState]:
        return [self.update(localization) for localization in localizations]

    def _process_noise(self, time_step_s):
        """Discrete white-noise-acceleration Q per coordinate (couples position & its velocity)."""
        # Q = q·[[Δt⁴/4, Δt³/2], [Δt³/2, Δt²]] per (position, velocity) pair; q = acceleration variance
        position_noise_scale, cross_noise_scale, velocity_noise_scale = (
            time_step_s ** 4 / 4.0, time_step_s ** 3 / 2.0, time_step_s ** 2)
        Q = np.zeros((4, 4))
        Q[0, 0], Q[0, 2], Q[2, 0], Q[2, 2] = (
            self._angle_process_variance * position_noise_scale,
            self._angle_process_variance * cross_noise_scale,
            self._angle_process_variance * cross_noise_scale,
            self._angle_process_variance * velocity_noise_scale)
        Q[1, 1], Q[1, 3], Q[3, 1], Q[3, 3] = (
            self._range_process_variance * position_noise_scale,
            self._range_process_variance * cross_noise_scale,
            self._range_process_variance * cross_noise_scale,
            self._range_process_variance * velocity_noise_scale)
        return Q

    def _emit(self, timestamp_s, confidence, *, measured, has_range):
        angle_deg, range_m = float(self._x[0]), float(self._x[1])
        radius = range_m if has_range else 1.0
        angle_rad = np.deg2rad(angle_deg)
        return TrackState(
            timestamp=timestamp_s, angle_deg=angle_deg, range_m=(range_m if has_range else float("nan")),
            angular_rate=float(self._x[2]), radial_rate=(float(self._x[3]) if has_range else 0.0),
            x_m=radius * float(np.sin(angle_rad)), y_m=radius * float(np.cos(angle_rad)),
            measured=measured, confidence=confidence,
        )
