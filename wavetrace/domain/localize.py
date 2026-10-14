"""Spatial localization — angle-of-arrival (AoA) "where is the person" heatmap.

The presence/weapon heads answer "is something here"; this answers "WHERE". It needs >= 2 RX
antennas on one radio (2-antenna ESP32-S3/C6, or a Raspberry-Pi NIC) — the inter-antenna phase
progression of one arriving path encodes its azimuth.

Two estimators:
  * PER-FRAME track (`locate`) — a fast 1-D AoA pseudo-spectrum (MUSIC default, Bartlett option) with
    subcarriers as snapshots; the azimuth peak + a coarse delay range place a streaming (x, y).
  * AGGREGATE room map (`aggregate`) — a TRUE JOINT 2-D delay-AoA MUSIC (SpotFi-style) over a window
    of frames: each frame is one snapshot of the vectorized (A·S) channel, so the steering vector
    a(θ, τ) spans antennas (angle) AND subcarriers (delay/range) jointly — NOT the separable outer
    product. Forward-backward averaging decorrelates coherent multipath. The 2-D pseudo-spectrum is
    the real (range × angle) heatmap and its argmax is the joint (θ, range) fix.

This is OFFLINE / Pi-side analysis, NOT the <8 ms real-time gate (joint MUSIC eigendecomposes an
A·S × A·S matrix). Range carries an unknown STO offset and, at WiFi bandwidths (20/40 MHz), the delay
phase ramp over room-scale ranges is small — so range stays COARSE/relative; azimuth is the
trustworthy axis. The joint estimator is the correct form; bandwidth (not the algorithm) is the limit.

Conventions: uniform linear array (ULA), antennas 0..A-1 spaced `spacing` wavelengths; azimuth θ is
measured from broadside (0° = straight ahead), +θ to one side. Steering a_m(θ) = exp(j2π·spacing·m·sinθ).
"""

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s


@dataclass
class Localization:
    """One frame's (or one aggregate window's) spatial estimate. All angles in degrees, ranges in m."""

    timestamp: float
    angles_deg: np.ndarray       # (G,) azimuth grid
    angle_spectrum: np.ndarray   # (G,) AoA pseudo-spectrum, normalized to [0, 1]
    ranges_m: np.ndarray         # (R,) range grid (empty if range disabled)
    range_profile: np.ndarray    # (R,) delay-domain power, normalized to [0, 1] (empty if disabled)
    heatmap: np.ndarray          # (R, G) when `aggregate` runs joint MUSIC; (1, G) azimuth spectrum otherwise
    peak_angle_deg: float
    peak_range_m: float          # range of the dominant delay bin (nan if range disabled)
    x_m: float                   # lateral position = peak_range·sin(peak_angle) (1·sin if no range)
    y_m: float                   # forward position = peak_range·cos(peak_angle) (1·cos if no range)
    confidence: float            # spectrum peakedness 1 - mean/max in [0, 1] (sharp peak -> ~1)


def _to_complex_grid(frame) -> np.ndarray:
    """Accept a CsiFrame (has .grid) or a raw (A, S) complex array -> (A, S) complex128."""
    csi_grid = np.asarray(getattr(frame, "grid", frame))
    if csi_grid.ndim != 2:
        raise ValueError(f"Localizer: expected an (A, S) grid, got shape {csi_grid.shape}")
    return csi_grid.astype(np.complex128, copy=False)


class Localizer:
    """Per-frame AoA + range estimator over a ULA. `spacing` is in wavelengths (0.5 is alias-free to
    ±90°); the azimuth grid spans [-90, 90]. MUSIC needs num_antennas > num_sources (noise-subspace
    dim = A - num_sources); bartlett needs no source count. The range axis is coarse at WiFi
    bandwidth; range_enabled=False is azimuth-only."""

    def __init__(self, num_antennas, *, spacing=0.5, method="music", num_sources=1,
                 num_angles=181, subcarrier_spacing_hz=312.5e3, max_range_m=12.0, num_ranges=64,
                 range_enabled=True):
        if num_antennas < 2:
            raise ValueError("Localizer needs >= 2 antennas (no spatial baseline with 1)")
        if method not in ("music", "bartlett"):
            raise ValueError("method must be 'music' or 'bartlett'")
        if method == "music" and not (0 < num_sources < num_antennas):
            raise ValueError("num_sources must satisfy 0 < num_sources < num_antennas for MUSIC")
        self.A = int(num_antennas)
        self.method = method
        self.num_sources = int(num_sources)
        self.range_enabled = bool(range_enabled)
        self._subcarrier_spacing_hz = float(subcarrier_spacing_hz)
        self._max_range_m = float(max_range_m)
        self._num_ranges = int(num_ranges)
        self.angles_deg = np.linspace(-90.0, 90.0, int(num_angles))
        # steering matrix a[m, g] = exp(j·2π·spacing·m·sinθ_g) — precomputed once (A × G).
        antenna_indices = np.arange(self.A)[:, None]
        sin_theta = np.sin(np.deg2rad(self.angles_deg))[None, :]
        self._steering_matrix = np.exp(1j * 2.0 * np.pi * spacing * antenna_indices * sin_theta)

    def angle_spectrum(self, grid) -> np.ndarray:
        """(A, S) complex CSI -> (G,) AoA pseudo-spectrum normalized to [0, 1]. Subcarriers are the
        covariance snapshots. O(A²S + A³ + A·G)."""
        csi_grid = _to_complex_grid(grid)
        if csi_grid.shape[0] != self.A:
            raise ValueError(f"Localizer: grid has {csi_grid.shape[0]} antennas, expected {self.A}")
        # R = H·H^H / S: (A, A) spatial covariance over subcarrier snapshots
        R = (csi_grid @ csi_grid.conj().T) / csi_grid.shape[1]
        if self.method == "bartlett":
            # P(θ) = a(θ)^H R a(θ): conventional beamformer energy per look direction
            spectrum = np.einsum("ag,ab,bg->g", self._steering_matrix.conj(), R, self._steering_matrix).real
        else:
            # MUSIC: P(θ) = 1 / ||E_n^H a(θ)||², E_n = the A - num_sources noise-subspace eigenvectors of R
            _, eigenvectors = np.linalg.eigh(R)  # ascending eigenvalues -> noise subspace = leading cols
            E_n = eigenvectors[:, : self.A - self.num_sources]
            noise_projection = E_n.conj().T @ self._steering_matrix            # (A-ns, G)
            noise_projection_power = np.sum(np.abs(noise_projection) ** 2, axis=0)  # -> 0 at a true arrival
            spectrum = 1.0 / np.maximum(noise_projection_power, 1e-12)
        lowest, highest = float(spectrum.min()), float(spectrum.max())
        return ((spectrum - lowest) / (highest - lowest)) if highest > lowest else np.zeros_like(spectrum)

    def range_profile(self, grid) -> tuple[np.ndarray, np.ndarray]:
        """(A, S) -> (ranges_m, profile) over the delay domain. IFFT across antenna-averaged
        subcarriers -> a power-vs-delay profile; bin r maps to range c·r/(S·Δf). Coarse (carries an
        STO offset). O(S log S)."""
        csi_grid = _to_complex_grid(grid)
        frequency_response = csi_grid.mean(axis=0)                  # (S,) antenna-averaged complex CFR
        S = frequency_response.size
        delay_power = np.abs(np.fft.ifft(frequency_response)) ** 2  # (S,) power vs delay bin
        ranges_m = SPEED_OF_LIGHT * np.arange(S) / (S * self._subcarrier_spacing_hz)  # range_r = c·r / (S·Δf)
        is_within_max_range = ranges_m <= self._max_range_m
        ranges_m, delay_power = ranges_m[is_within_max_range], delay_power[is_within_max_range]
        highest = float(delay_power.max())
        profile = (delay_power / highest) if highest > 0 else delay_power
        return ranges_m, profile

    def locate(self, frame, timestamp=0.0) -> Localization:
        """Per-frame STREAMING estimate: 1-D AoA spectrum + a coarse delay range to place (x, y). The
        per-frame heatmap is the (1, G) azimuth spectrum (a single frame can't resolve a room map —
        that is `aggregate`'s joint 2-D MUSIC). O(A²S + A³ + A·G)."""
        spectrum = self.angle_spectrum(frame)
        peak_angle_deg = float(self.angles_deg[int(np.argmax(spectrum))])
        mean_over_max = float(spectrum.mean()) / float(spectrum.max()) if spectrum.max() > 0 else 1.0
        confidence = max(0.0, 1.0 - mean_over_max)

        if self.range_enabled:
            ranges, profile = self.range_profile(frame)
            peak_range_m = float(ranges[int(np.argmax(profile))]) if ranges.size else float("nan")
            radius = peak_range_m
        else:
            ranges, profile, peak_range_m, radius = np.empty(0), np.empty(0), float("nan"), 1.0

        peak_angle_rad = np.deg2rad(peak_angle_deg)
        return Localization(
            timestamp=float(timestamp), angles_deg=self.angles_deg, angle_spectrum=spectrum,
            ranges_m=ranges, range_profile=profile, heatmap=spectrum.astype(np.float32)[None, :],
            peak_angle_deg=peak_angle_deg, peak_range_m=peak_range_m,
            x_m=radius * float(np.sin(peak_angle_rad)), y_m=radius * float(np.cos(peak_angle_rad)),
            confidence=confidence,
        )

    def _joint_2d_music(self, grids) -> tuple[np.ndarray, np.ndarray]:
        """SpotFi-style joint (range × angle) MUSIC over a window of frames. Each frame is one
        snapshot of the vectorized (A·S) channel; the steering a(θ, τ)[m, k] = e^{j2π·spacing·m·sinθ}·
        e^{-j2π·Δf·k·τ} spans antennas AND subcarriers JOINTLY (not a separable product). Forward-
        backward averaging decorrelates coherent paths. Returns (ranges_m, P_2d) with P_2d (R, G)
        normalized to [0, 1]. O(A·S·T + (A·S)³ + A·S·G·R)."""
        A = self.A
        S = grids[0].shape[1]
        num_joint_elements = A * S
        X = np.stack([csi_grid.reshape(-1) for csi_grid in grids], axis=1)  # (A·S, T) vec(H) snapshots
        R = (X @ X.conj().T) / X.shape[1]  # R = X·X^H / T
        J = np.fliplr(np.eye(num_joint_elements))  # exchange matrix (anti-diagonal identity)
        R = 0.5 * (R + J @ R.conj() @ J)  # forward-backward smoothing (coherent-path decorrelation)
        _, eigenvectors = np.linalg.eigh(R)                   # ascending -> noise subspace leads
        E_n = eigenvectors[:, : num_joint_elements - self.num_sources]         # (A·S, A·S - ns)

        ranges = np.linspace(0.0, self._max_range_m, self._num_ranges)
        delays_s = ranges / SPEED_OF_LIGHT
        delay_steering = np.exp(-1j * 2.0 * np.pi * self._subcarrier_spacing_hz
                                * np.outer(np.arange(S), delays_s))  # (S, R)
        # a(θ,τ)[m,k] = steering_matrix[m,θ]·delay_steering[k,τ]; build (A,S,G,R) then flatten to
        # (A·S, G·R), col = θ*R+τ
        joint_steering_grid = self._steering_matrix[:, None, :, None] * delay_steering[None, :, None, :]
        joint_steering = joint_steering_grid.reshape(num_joint_elements,
                                                     self.angles_deg.size * self._num_ranges)
        noise_projection = E_n.conj().T @ joint_steering                    # (A·S - ns, G·R)
        # P(θ,τ) = 1 / ||E_n^H a(θ,τ)||²
        P = 1.0 / np.maximum(np.sum(np.abs(noise_projection) ** 2, axis=0), 1e-12)
        P_2d = P.reshape(self.angles_deg.size, self._num_ranges).T  # (R, G)
        lowest, highest = float(P_2d.min()), float(P_2d.max())
        return ranges, ((P_2d - lowest) / (highest - lowest) if highest > lowest else np.zeros_like(P_2d))

    def aggregate(self, frames) -> Localization:
        """ONE stable room map over `frames`. With range enabled (and > num_sources frames) this is
        the JOINT 2-D delay-AoA MUSIC pseudo-spectrum; else it falls back to the time-averaged 1-D
        AoA spectrum (azimuth only). O(F·A·S + (A·S)³ + A·S·G·R)."""
        frames = list(frames)
        if not frames:
            raise ValueError("Localizer.aggregate: no frames")
        timestamp_s = float(getattr(frames[-1], "timestamp", 0.0))

        if self.range_enabled and len(frames) > self.num_sources:
            ranges, room_spectrum = self._joint_2d_music([_to_complex_grid(frame) for frame in frames])
            range_index, angle_index = np.unravel_index(int(np.argmax(room_spectrum)), room_spectrum.shape)
            peak_angle_deg, peak_range_m = float(self.angles_deg[angle_index]), float(ranges[range_index])
            angle_spectrum = room_spectrum.max(axis=0)        # marginal over range
            range_profile = room_spectrum.max(axis=1)         # marginal over angle
            heatmap = room_spectrum.astype(np.float32)
            peak_value = float(room_spectrum.max())
            confidence = max(0.0, 1.0 - float(room_spectrum.mean()) / peak_value) if peak_value > 0 else 0.0
            radius = peak_range_m
        else:
            angle_spectrum = np.zeros_like(self.angles_deg)
            for frame in frames:
                angle_spectrum += self.angle_spectrum(frame)
            angle_spectrum /= len(frames)
            peak_angle_deg = float(self.angles_deg[int(np.argmax(angle_spectrum))])
            peak_value = float(angle_spectrum.max())
            confidence = max(0.0, 1.0 - float(angle_spectrum.mean()) / peak_value) if peak_value > 0 else 0.0
            ranges, range_profile = np.empty(0), np.empty(0)
            peak_range_m, radius = float("nan"), 1.0
            heatmap = angle_spectrum.astype(np.float32)[None, :]

        peak_angle_rad = np.deg2rad(peak_angle_deg)
        return Localization(
            timestamp=timestamp_s, angles_deg=self.angles_deg, angle_spectrum=angle_spectrum, ranges_m=ranges,
            range_profile=range_profile, heatmap=heatmap, peak_angle_deg=peak_angle_deg,
            peak_range_m=peak_range_m, x_m=radius * float(np.sin(peak_angle_rad)),
            y_m=radius * float(np.cos(peak_angle_rad)), confidence=confidence,
        )

    def locate_stream(self, frames):
        """Yield one Localization per frame (frame.timestamp used when present). O(F·(A²S + A·G))."""
        for frame in frames:
            yield self.locate(frame, timestamp=float(getattr(frame, "timestamp", 0.0)))


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
