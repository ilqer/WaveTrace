"""Temporal fusion of heatmap frames: per-cell Bayesian filter + 2D position Kalman smoother.

OccupancyGrid is a grid-wide Kalman predict/update: PREDICT (decay toward empty + small spatial blur;
the blur is the process noise, modelling motion uncertainty) then UPDATE (blend the new measurement
weighted by confidence). _GridKalman smooths the peak cell so the indicator doesn't jump between frames.

Not wavetrace.localize.Tracker: that one tracks azimuth and range, while this tracks grid cells directly.
O(G²) per OccupancyGrid step.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OccupancyOptions:
    """`decay`: higher = forget faster. `blur`: 4-neighbour spatial coupling each predict step, i.e.
    motion uncertainty. `measurement_weight_floor`: minimum measurement weight so even
    low-confidence frames still update."""

    grid: int = 16
    decay: float = 0.1
    blur: float = 0.5
    measurement_weight_floor: float = 0.2

    def __post_init__(self) -> None:
        if self.grid <= 0:
            raise ValueError("grid must be positive")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if self.blur < 0.0:
            raise ValueError("blur must be non-negative")
        if not 0.0 <= self.measurement_weight_floor <= 1.0:
            raise ValueError("measurement_weight_floor must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class KalmanOptions:
    acceleration_std: float = 2.0
    measurement_std: float = 1.5
    gate: float = 9.0

    def __post_init__(self) -> None:
        if self.acceleration_std <= 0:
            raise ValueError("acceleration_std must be positive")
        if self.measurement_std <= 0:
            raise ValueError("measurement_std must be positive")
        if self.gate <= 0:
            raise ValueError("gate must be positive")


class OccupancyGrid:
    def __init__(self, options: OccupancyOptions = OccupancyOptions()):
        self.g = options.grid
        self.decay = options.decay
        self.blur = options.blur
        self.floor = options.measurement_weight_floor
        self.state = np.zeros((self.g, self.g), dtype=np.float32)

    def _predict(self) -> None:
        decayed = self.state * (1.0 - self.decay)
        blur_weight = self.blur
        blurred = decayed.copy()
        blurred[1:, :] += blur_weight * decayed[:-1, :]
        blurred[:-1, :] += blur_weight * decayed[1:, :]
        blurred[:, 1:] += blur_weight * decayed[:, :-1]
        blurred[:, :-1] += blur_weight * decayed[:, 1:]
        self.state = np.clip(blurred / (1 + 4 * blur_weight), 0.0, 1.0)

    def update(self, measurement, confidence=1.0) -> np.ndarray:
        """Predict, then blend in the G×G measurement weighted by confidence."""
        self._predict()
        measured_grid = np.asarray(measurement, dtype=np.float32).reshape(self.g, self.g)
        measurement_weight = max(self.floor, float(confidence))
        self.state = np.clip((1 - measurement_weight) * self.state + measurement_weight * measured_grid,
                             0.0, 1.0)
        return self.state

    def peak(self) -> tuple[int, int, float]:
        i, j = np.unravel_index(int(np.argmax(self.state)), self.state.shape)
        return int(i), int(j), float(self.state[i, j])

    def reset(self) -> None:
        self.state[:] = 0.0


class _GridKalman:
    """Constant-velocity 2D Kalman in grid coordinates (row, col).
    State x = [row, col, row_rate, col_rate] in cells and cells/s, covariance P."""

    def __init__(self, options: KalmanOptions = KalmanOptions()):
        self._qa = options.acceleration_std ** 2    # σ_a², process (acceleration) variance
        self._rs = options.measurement_std ** 2     # σ_r², measurement variance in cells²
        self._gate = options.gate
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._last_timestamp_s: float | None = None

    def update(self, row: float, col: float, timestamp_s: float,
               confidence=1.0) -> tuple[float, float, bool]:
        if self._x is None:
            self._x = np.array([row, col, 0.0, 0.0])
            self._P = np.diag([self._rs, self._rs, 100.0, 100.0])
            self._last_timestamp_s = float(timestamp_s)
            return row, col, True
        elapsed_s = max(float(timestamp_s) - self._last_timestamp_s, 1e-3)
        self._last_timestamp_s = float(timestamp_s)
        # constant-velocity model: x' = F x with F = [[I, Δt·I], [0, I]], Δt = elapsed_s
        F = np.array([[1, 0, elapsed_s, 0], [0, 1, 0, elapsed_s], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        # white-noise acceleration, per axis: Q = σ_a² · [[Δt⁴/4, Δt³/2], [Δt³/2, Δt²]]
        position_factor, cross_factor, velocity_factor = (
            elapsed_s ** 4 / 4, elapsed_s ** 3 / 2, elapsed_s ** 2)
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = self._qa * position_factor; Q[i, i+2] = self._qa * cross_factor
            Q[i+2, i] = self._qa * cross_factor; Q[i+2, i+2] = self._qa * velocity_factor
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        # low confidence inflates the measurement noise: R = (σ_r² / confidence) · I
        clamped_confidence = max(float(confidence), 0.05)
        H = np.eye(2, 4)     # z = H x observes (row, col) only
        z = np.array([float(row), float(col)])
        R = np.eye(2) * (self._rs / clamped_confidence)
        innovation = z - H @ self._x
        S = H @ self._P @ H.T + R     # innovation covariance S = H P Hᵀ + R
        S_inv = np.linalg.inv(S)
        # Mahalanobis gate: accept only if innovationᵀ S⁻¹ innovation <= gate (rejects impossible jumps)
        mahalanobis_squared = float(innovation @ S_inv @ innovation)
        is_measurement_accepted = mahalanobis_squared <= self._gate
        if is_measurement_accepted:
            K = self._P @ H.T @ S_inv     # Kalman gain K = P Hᵀ S⁻¹; then x += K·innovation, P = (I - K H) P
            self._x = self._x + K @ innovation
            self._P = (np.eye(4) - K @ H) @ self._P
        return float(self._x[0]), float(self._x[1]), is_measurement_accepted


class HeatmapTrack:
    """cell_size_m: grid cell side in metres (default 0.25 m for a 4 m × 4 m room with G=16)."""

    def __init__(self, *, cell_size_m=0.25, occupancy: OccupancyOptions = OccupancyOptions(),
                 kalman: KalmanOptions = KalmanOptions()):
        self.g = occupancy.grid
        self.cell = float(cell_size_m)
        self._kalman_options = kalman
        self.occ = OccupancyGrid(occupancy)
        self._kalman = _GridKalman(kalman)

    def update(self, measurement, confidence: float, timestamp_s: float) -> dict:
        fused = self.occ.update(measurement, confidence)
        peak_row, peak_col, peak_value = self.occ.peak()
        filtered_row, filtered_col, is_measurement_accepted = self._kalman.update(
            float(peak_row), float(peak_col), timestamp_s, confidence)
        # grid cell -> metres (row 0 = far, last row = near; col centre = 0)
        track_x_m = (filtered_col - self.g / 2) * self.cell
        track_y_m = (self.g - filtered_row) * self.cell
        return {
            "grid": fused.flatten().tolist(),
            "peak": [peak_row, peak_col, peak_value],
            "track": {"x": round(track_x_m, 3), "y": round(track_y_m, 3),
                      "measured": is_measurement_accepted},
        }

    def reset(self) -> None:
        self.occ.reset()
        self._kalman = _GridKalman(self._kalman_options)
