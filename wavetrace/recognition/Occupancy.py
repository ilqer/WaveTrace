"""Temporal fusion of heatmap frames: per-cell Bayesian filter + 2D position Kalman smoother.

Two layers:
  * OccupancyGrid — per-cell predict/update filter: PREDICT (decay toward empty + small spatial blur
    models motion uncertainty) then UPDATE (blend the new measurement weighted by confidence).
    This is a grid-wide Kalman predict/update; the blur is the process noise.
  * _GridKalman — constant-velocity 2D Kalman in (row, col) grid coordinates, tracking the peak cell.
    Smooths the peak indicator so it doesn't jump between frames (anti-teleport).

Note: we do NOT use Localize.Tracker here — that tracks azimuth+range (AoA, parked per MINDMAP).
This module tracks grid cell coordinates directly. O(G²) per OccupancyGrid step.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OccupancyOptions:
    """Tuning knobs for `OccupancyGrid` / `HeatmapTrack` (§1.6: >3 related constructor parameters).

    `decay`: higher = forget faster. `blur`: 4-neighbour spatial coupling each predict step, i.e.
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
    """Tuning knobs for `_GridKalman`, the constant-velocity peak smoother."""

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
    """Per-cell predict/update filter over a G×G heatmap. See `OccupancyOptions` for the knobs."""

    def __init__(self, options: OccupancyOptions = OccupancyOptions()):
        self.g = options.grid
        self.decay = options.decay
        self.blur = options.blur
        self.floor = options.measurement_weight_floor
        self.state = np.zeros((self.g, self.g), dtype=np.float32)

    def _predict(self) -> None:
        s = self.state * (1.0 - self.decay)
        # 4-neighbour blur: spreads probability to neighbours = motion uncertainty
        b = self.blur
        blurred = s.copy()
        blurred[1:, :] += b * s[:-1, :]
        blurred[:-1, :] += b * s[1:, :]
        blurred[:, 1:] += b * s[:, :-1]
        blurred[:, :-1] += b * s[:, 1:]
        self.state = np.clip(blurred / (1 + 4 * b), 0.0, 1.0)

    def update(self, measurement, confidence=1.0) -> np.ndarray:
        """One step: predict then blend in the G×G measurement weighted by confidence.
        Returns the fused grid."""
        self._predict()
        m = np.asarray(measurement, dtype=np.float32).reshape(self.g, self.g)
        w = max(self.floor, float(confidence))
        self.state = np.clip((1 - w) * self.state + w * m, 0.0, 1.0)
        return self.state

    def peak(self) -> tuple[int, int, float]:
        """(row, col, value) of the hottest cell."""
        i, j = np.unravel_index(int(np.argmax(self.state)), self.state.shape)
        return int(i), int(j), float(self.state[i, j])

    def reset(self) -> None:
        self.state[:] = 0.0


class _GridKalman:
    """Constant-velocity 2D Kalman in grid coordinates (row, col).
    State = [r, c, dr, dc]. Fuses measurement with motion model; gates impossible jumps."""

    def __init__(self, options: KalmanOptions = KalmanOptions()):
        self._qa = options.acceleration_std ** 2
        self._rs = options.measurement_std ** 2
        self._gate = options.gate
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._t: float | None = None

    def update(self, row: float, col: float, t: float,
               confidence=1.0) -> tuple[float, float, bool]:
        """Returns (filtered_row, filtered_col, measurement_accepted)."""
        if self._x is None:
            self._x = np.array([row, col, 0.0, 0.0])
            self._P = np.diag([self._rs, self._rs, 100.0, 100.0])
            self._t = float(t)
            return row, col, True
        dt = max(float(t) - self._t, 1e-3); self._t = float(t)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        d4, d3, d2 = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = self._qa * d4; Q[i, i+2] = self._qa * d3
            Q[i+2, i] = self._qa * d3; Q[i+2, i+2] = self._qa * d2
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        sf = max(float(confidence), 0.05)
        H = np.eye(2, 4)
        z = np.array([float(row), float(col)])
        R = np.eye(2) * (self._rs / sf)
        innov = z - H @ self._x
        S = H @ self._P @ H.T + R
        Sinv = np.linalg.inv(S)
        maha2 = float(innov @ Sinv @ innov)
        measured = maha2 <= self._gate
        if measured:
            K = self._P @ H.T @ Sinv
            self._x = self._x + K @ innov
            self._P = (np.eye(4) - K @ H) @ self._P
        return float(self._x[0]), float(self._x[1]), measured


class HeatmapTrack:
    """Couples OccupancyGrid (per-cell Bayesian filter) with _GridKalman (peak smoother).
    The UI shows both the fused grid AND a non-jumping peak marker.

    cell_size_m: grid cell side in metres (default 0.25 m for a 4 m × 4 m room with G=16).
    grid_origin: (row_origin, col_origin) = the grid index corresponding to (0, 0) in metres.
    """

    def __init__(self, *, cell_size_m=0.25, occupancy: OccupancyOptions = OccupancyOptions(),
                 kalman: KalmanOptions = KalmanOptions()):
        self.g = occupancy.grid
        self.cell = float(cell_size_m)
        self._kalman_options = kalman
        self.occ = OccupancyGrid(occupancy)
        self._kalman = _GridKalman(kalman)

    def update(self, measurement, confidence: float, t: float) -> dict:
        """One step: fuse measurement into the grid, smooth the peak. Returns the telemetry dict."""
        fused = self.occ.update(measurement, confidence)
        pi, pj, pval = self.occ.peak()
        fr, fc, measured = self._kalman.update(float(pi), float(pj), t, confidence)
        # grid cell -> metres (row 0 = far, last row = near; col centre = 0)
        xM = (fc - self.g / 2) * self.cell
        yM = (self.g - fr) * self.cell
        return {
            "grid": fused.flatten().tolist(),
            "peak": [pi, pj, pval],
            "track": {"x": round(xM, 3), "y": round(yM, 3), "measured": measured},
        }

    def reset(self) -> None:
        self.occ.reset()
        self._kalman = _GridKalman(self._kalman_options)
