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

from dataclasses import dataclass, field
import json
from pathlib import Path

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
    heatmap: np.ndarray          # (R, G) separable range×angle map = outer(range_profile, spectrum)
    peak_angle_deg: float        # azimuth of the dominant arrival
    peak_range_m: float          # range of the dominant delay bin (nan if range disabled)
    x_m: float                   # lateral position = peak_range·sin(peak_angle) (1·sin if no range)
    y_m: float                   # forward position = peak_range·cos(peak_angle) (1·cos if no range)
    confidence: float            # spectrum peakedness 1 - mean/max in [0, 1] (sharp peak -> ~1)


def _grid_of(frame) -> np.ndarray:
    """Accept a CsiFrame (has .grid) or a raw (A, S) complex array -> (A, S) complex128."""
    g = np.asarray(getattr(frame, "grid", frame))
    if g.ndim != 2:
        raise ValueError(f"Localizer: expected an (A, S) grid, got shape {g.shape}")
    return g.astype(np.complex128, copy=False)


class Localizer:
    """Per-frame AoA + range estimator over a ULA. Build once (precomputes the steering matrix),
    then call `locate` per frame or `aggregate` over a stream.

    num_antennas: array size A (>= 2 — a single antenna has no spatial baseline).
    spacing:      element spacing in WAVELENGTHS (λ/2 = 0.5 is the standard, alias-free to ±90°).
    method:       "music" (subspace, high-resolution; needs A > num_sources) | "bartlett" (robust,
                  no source-count assumption).
    num_sources:  expected simultaneous arrivals (MUSIC noise-subspace dim = A - num_sources).
    num_angles:   azimuth grid resolution over [-90, 90].
    subcarrier_spacing_hz / max_range_m / num_ranges: the delay-domain range axis of the joint 2-D
                  room map (coarse at WiFi BW). range_enabled=False drops it (azimuth-only)."""

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
        self._df = float(subcarrier_spacing_hz)
        self._max_range = float(max_range_m)
        self._num_ranges = int(num_ranges)
        self.angles_deg = np.linspace(-90.0, 90.0, int(num_angles))
        # steering matrix Asteer[m, g] = exp(j·2π·spacing·m·sinθ_g) — precomputed once (A × G).
        m = np.arange(self.A)[:, None]
        sin_t = np.sin(np.deg2rad(self.angles_deg))[None, :]
        self._steer = np.exp(1j * 2.0 * np.pi * spacing * m * sin_t)  # (A, G) complex128

    # ----- AoA spectrum -------------------------------------------------------------------------

    def angle_spectrum(self, grid) -> np.ndarray:
        """(A, S) complex CSI -> (G,) AoA pseudo-spectrum normalized to [0, 1]. Subcarriers are the
        covariance snapshots. O(A²S + A³ + A·G)."""
        g = _grid_of(grid)
        if g.shape[0] != self.A:
            raise ValueError(f"Localizer: grid has {g.shape[0]} antennas, expected {self.A}")
        R = (g @ g.conj().T) / g.shape[1]  # (A, A) spatial covariance over subcarrier snapshots
        if self.method == "bartlett":
            # P(θ) = a(θ)^H R a(θ): conventional beamformer energy per look direction
            spec = np.einsum("ag,ab,bg->g", self._steer.conj(), R, self._steer).real
        else:  # MUSIC
            _, evecs = np.linalg.eigh(R)  # ascending eigenvalues -> noise subspace = leading cols
            En = evecs[:, : self.A - self.num_sources]
            proj = En.conj().T @ self._steer            # (A-ns, G)
            denom = np.sum(np.abs(proj) ** 2, axis=0)    # ||E_n^H a||²; -> 0 at a true arrival
            spec = 1.0 / np.maximum(denom, 1e-12)
        lo, hi = float(spec.min()), float(spec.max())
        return ((spec - lo) / (hi - lo)) if hi > lo else np.zeros_like(spec)

    # ----- delay-domain range profile -----------------------------------------------------------

    def range_profile(self, grid) -> tuple[np.ndarray, np.ndarray]:
        """(A, S) -> (ranges_m, profile) over the delay domain. IFFT across antenna-averaged
        subcarriers -> a power-vs-delay profile; bin r maps to range c·r/(S·Δf). Coarse (carries an
        STO offset). O(S log S)."""
        g = _grid_of(grid)
        h = g.mean(axis=0)                         # (S,) antenna-averaged complex CFR
        S = h.size
        delay = np.abs(np.fft.ifft(h)) ** 2        # (S,) power vs delay bin
        rng = SPEED_OF_LIGHT * np.arange(S) / (S * self._df)  # bin -> metres
        keep = rng <= self._max_range
        rng, delay = rng[keep], delay[keep]
        hi = float(delay.max())
        prof = (delay / hi) if hi > 0 else delay
        return rng, prof

    # ----- one frame ----------------------------------------------------------------------------

    def locate(self, frame, timestamp=0.0) -> Localization:
        """Per-frame STREAMING estimate: 1-D AoA spectrum + a coarse delay range to place (x, y). The
        per-frame heatmap is the (1, G) azimuth spectrum (a single frame can't resolve a room map —
        that is `aggregate`'s joint 2-D MUSIC). O(A²S + A³ + A·G)."""
        spec = self.angle_spectrum(frame)
        peak_angle = float(self.angles_deg[int(np.argmax(spec))])
        mean_over_max = float(spec.mean()) / float(spec.max()) if spec.max() > 0 else 1.0
        conf = max(0.0, 1.0 - mean_over_max)

        if self.range_enabled:
            ranges, prof = self.range_profile(frame)
            peak_r = float(ranges[int(np.argmax(prof))]) if ranges.size else float("nan")
            radius = peak_r
        else:
            ranges, prof, peak_r, radius = np.empty(0), np.empty(0), float("nan"), 1.0

        th = np.deg2rad(peak_angle)
        return Localization(
            timestamp=float(timestamp), angles_deg=self.angles_deg, angle_spectrum=spec,
            ranges_m=ranges, range_profile=prof, heatmap=spec.astype(np.float32)[None, :],
            peak_angle_deg=peak_angle, peak_range_m=peak_r,
            x_m=radius * float(np.sin(th)), y_m=radius * float(np.cos(th)), confidence=conf,
        )

    # ----- joint 2-D delay-AoA MUSIC (the room map) ---------------------------------------------

    def _joint_2d_music(self, grids) -> tuple[np.ndarray, np.ndarray]:
        """SpotFi-style joint (range × angle) MUSIC over a window of frames. Each frame is one
        snapshot of the vectorized (A·S) channel; the steering a(θ, τ)[m, k] = e^{j2π·spacing·m·sinθ}·
        e^{-j2π·Δf·k·τ} spans antennas AND subcarriers JOINTLY (not a separable product). Forward-
        backward averaging decorrelates coherent paths. Returns (ranges_m, P2d) with P2d (R, G)
        normalized to [0, 1]. O(A·S·T + (A·S)³ + A·S·G·R)."""
        A = self.A
        S = grids[0].shape[1]
        AS = A * S
        X = np.stack([g.reshape(-1) for g in grids], axis=1)  # (A·S, T) vec(H) snapshots
        R = (X @ X.conj().T) / X.shape[1]
        J = np.fliplr(np.eye(AS))
        R = 0.5 * (R + J @ R.conj() @ J)  # forward-backward smoothing (coherent-path decorrelation)
        _, evecs = np.linalg.eigh(R)                          # ascending -> noise subspace leads
        En = evecs[:, : AS - self.num_sources]                # (A·S, A·S - ns)

        ranges = np.linspace(0.0, self._max_range, self._num_ranges)
        tau = ranges / SPEED_OF_LIGHT
        sub = np.exp(-1j * 2.0 * np.pi * self._df * np.outer(np.arange(S), tau))  # (S, R)
        # a(θ,τ)[m,k] = ant[m,θ]·sub[k,τ]; build (A,S,G,R) then flatten to (A·S, G·R), col = θ*R+τ
        a2 = self._steer[:, None, :, None] * sub[None, :, None, :]
        steer = a2.reshape(AS, self.angles_deg.size * self._num_ranges)
        M = En.conj().T @ steer                                # (A·S - ns, G·R)
        P = 1.0 / np.maximum(np.sum(np.abs(M) ** 2, axis=0), 1e-12)
        P2d = P.reshape(self.angles_deg.size, self._num_ranges).T  # (R, G)
        lo, hi = float(P2d.min()), float(P2d.max())
        return ranges, ((P2d - lo) / (hi - lo) if hi > lo else np.zeros_like(P2d))

    def aggregate(self, frames) -> Localization:
        """ONE stable room map over `frames`. With range enabled (and > num_sources frames) this is
        the JOINT 2-D delay-AoA MUSIC pseudo-spectrum; else it falls back to the time-averaged 1-D
        AoA spectrum (azimuth only). O(F·A·S + (A·S)³ + A·S·G·R)."""
        frames = list(frames)
        if not frames:
            raise ValueError("Localizer.aggregate: no frames")
        ts = float(getattr(frames[-1], "timestamp", 0.0))

        if self.range_enabled and len(frames) > self.num_sources:
            ranges, P2d = self._joint_2d_music([_grid_of(fr) for fr in frames])
            ri, gi = np.unravel_index(int(np.argmax(P2d)), P2d.shape)
            peak_angle, peak_r = float(self.angles_deg[gi]), float(ranges[ri])
            angle_spectrum = P2d.max(axis=0)        # marginal over range
            range_profile = P2d.max(axis=1)         # marginal over angle
            heatmap = P2d.astype(np.float32)
            mx = float(P2d.max())
            conf = max(0.0, 1.0 - float(P2d.mean()) / mx) if mx > 0 else 0.0
            radius = peak_r
        else:  # azimuth-only fallback: average the per-frame 1-D AoA spectra
            angle_spectrum = np.zeros_like(self.angles_deg)
            for fr in frames:
                angle_spectrum += self.angle_spectrum(fr)
            angle_spectrum /= len(frames)
            peak_angle = float(self.angles_deg[int(np.argmax(angle_spectrum))])
            mx = float(angle_spectrum.max())
            conf = max(0.0, 1.0 - float(angle_spectrum.mean()) / mx) if mx > 0 else 0.0
            ranges, range_profile = np.empty(0), np.empty(0)
            peak_r, radius = float("nan"), 1.0
            heatmap = angle_spectrum.astype(np.float32)[None, :]

        th = np.deg2rad(peak_angle)
        return Localization(
            timestamp=ts, angles_deg=self.angles_deg, angle_spectrum=angle_spectrum, ranges_m=ranges,
            range_profile=range_profile, heatmap=heatmap, peak_angle_deg=peak_angle,
            peak_range_m=peak_r, x_m=radius * float(np.sin(th)), y_m=radius * float(np.cos(th)),
            confidence=conf,
        )

    def locate_stream(self, frames):
        """Yield one Localization per frame (frame.timestamp used when present). O(F·(A²S + A·G))."""
        for fr in frames:
            yield self.locate(fr, timestamp=float(getattr(fr, "timestamp", 0.0)))


# ----- temporal tracking: constant-velocity Kalman filter ---------------------------------------

@dataclass
class TrackState:
    """One filtered step of a moving target. The filter fuses PREDICTION (motion) with the MEASUREMENT
    (MUSIC), so the track is continuous and `measured=False` marks a frame whose measurement was gated
    out as an impossible jump (the track coasted on its motion model)."""

    timestamp: float
    angle_deg: float      # filtered azimuth
    range_m: float        # filtered range (nan when range tracking is off)
    angular_rate: float   # deg/s
    radial_rate: float    # m/s (0 when range off)
    x_m: float
    y_m: float
    measured: bool        # True if the measurement passed the gate and updated the state
    confidence: float     # the measurement confidence that set the Kalman gain this step


class Tracker:
    """Constant-velocity Kalman filter over the Localizer's (angle, range) measurements — the
    "it can't teleport" model the user asked for. State = [angle, range, d_angle, d_range].

    PREDICT (calculation): a person moves continuously, so the next state = motion-extrapolated
    previous state; the process noise (`accel_*`, a max-acceleration bound) is what limits how fast
    the track may move. UPDATE (measurement): the MUSIC measurement is fused with a noise covariance
    that SHRINKS WITH CONFIDENCE — high confidence -> high Kalman gain (trust the measurement), low
    confidence -> low gain (trust the motion model). A χ² gate on the innovation rejects measurements
    that imply a jump no real motion could produce (anti-teleport). O(1) per step (4×4 matrices).

    angle_std / range_std: measurement std at FULL confidence (deg / m). accel_angle / accel_range:
    process-noise acceleration std (deg/s² / m/s²) — the motion bound. conf_floor: min effective
    confidence (avoids divide-by-zero / infinite trust in noise). gate: χ² innovation gate (9.0 ≈
    98.9% for 2 DOF). range_enabled=False tracks azimuth only (range measurements ignored)."""

    def __init__(self, *, angle_std=3.0, range_std=4.0, accel_angle=40.0, accel_range=3.0,
                 conf_floor=0.05, gate=9.0, range_enabled=True):
        self._angle_var = float(angle_std) ** 2
        self._range_var = float(range_std) ** 2
        self._qa = float(accel_angle) ** 2
        self._qr = float(accel_range) ** 2
        self._conf_floor = float(conf_floor)
        self._gate = float(gate)
        self._range_enabled = bool(range_enabled)
        self._x = None          # state (4,), lazily initialized from the first measurement
        self._P = None          # covariance (4, 4)
        self._t = None

    def update(self, localization: Localization) -> TrackState:
        """One predict+update step from a `Localization` measurement. Returns the filtered TrackState."""
        z_angle = float(localization.peak_angle_deg)
        z_range = float(localization.peak_range_m)
        has_range = self._range_enabled and not np.isnan(z_range)
        conf = float(localization.confidence)
        t = float(localization.timestamp)

        if self._x is None:  # initialize on the first measurement (no motion history yet)
            self._x = np.array([z_angle, (z_range if has_range else 0.0), 0.0, 0.0])
            self._P = np.diag([self._angle_var, self._range_var, 1e3, 1e3])
            self._t = t
            return self._emit(t, conf, measured=True, has_range=has_range)

        dt = max(t - self._t, 1e-3)
        self._t = t
        # ----- predict (constant velocity; process noise = the max-acceleration motion bound) -----
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._process_noise(dt)

        # ----- update (measure angle, + range when available; R shrinks with confidence) ----------
        sf = max(conf, self._conf_floor)  # confidence factor: higher conf -> smaller R -> higher gain
        if has_range:
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
            z = np.array([z_angle, z_range])
            R = np.diag([self._angle_var / sf, self._range_var / sf])
        else:
            H = np.array([[1, 0, 0, 0]], float)
            z = np.array([z_angle])
            R = np.array([[self._angle_var / sf]])
        y = z - H @ self._x                     # innovation (measurement - prediction)
        S = H @ self._P @ H.T + R
        Sinv = np.linalg.inv(S)
        maha2 = float(y @ Sinv @ y)             # squared Mahalanobis distance of the innovation
        measured = maha2 <= self._gate          # gate: reject impossible jumps (anti-teleport)
        if measured:
            K = self._P @ H.T @ Sinv            # Kalman gain (the confidence-weighted blend)
            self._x = self._x + K @ y
            self._P = (np.eye(4) - K @ H) @ self._P
        return self._emit(t, conf, measured=measured, has_range=has_range)

    def run(self, localizations) -> list[TrackState]:
        """Filter a whole stream of measurements into a smoothed track. O(F)."""
        return [self.update(l) for l in localizations]

    def _process_noise(self, dt):
        """Discrete white-noise-acceleration Q per coordinate (couples position & its velocity)."""
        d4, d3, d2 = dt ** 4 / 4.0, dt ** 3 / 2.0, dt ** 2
        Q = np.zeros((4, 4))
        Q[0, 0], Q[0, 2], Q[2, 0], Q[2, 2] = self._qa * d4, self._qa * d3, self._qa * d3, self._qa * d2
        Q[1, 1], Q[1, 3], Q[3, 1], Q[3, 3] = self._qr * d4, self._qr * d3, self._qr * d3, self._qr * d2
        return Q

    def _emit(self, t, conf, *, measured, has_range):
        angle, rng = float(self._x[0]), float(self._x[1])
        radius = rng if has_range else 1.0
        th = np.deg2rad(angle)
        return TrackState(
            timestamp=t, angle_deg=angle, range_m=(rng if has_range else float("nan")),
            angular_rate=float(self._x[2]), radial_rate=(float(self._x[3]) if has_range else 0.0),
            x_m=radius * float(np.sin(th)), y_m=radius * float(np.cos(th)),
            measured=measured, confidence=conf,
        )


# ----- serialization (mirrors save_dataset / save_calibration) ----------------------------------

def save_localization(room_map: Localization, out_dir) -> Path:
    """Persist the aggregate room map under out_dir: heatmap.npy (R×G joint pseudo-spectrum) +
    angles.npy + ranges.npy (its grids) + meta.json (the peak fix). The PER-FRAME track is published
    through the Publisher wire schema (RecognitionResult), not written here. O(R·G)."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "angles.npy", room_map.angles_deg.astype(np.float32))
    np.save(p / "ranges.npy", room_map.ranges_m.astype(np.float32))
    np.save(p / "heatmap.npy", room_map.heatmap.astype(np.float32))
    with open(p / "meta.json", "w") as f:
        json.dump({
            "peak_angle_deg": room_map.peak_angle_deg,
            "peak_range_m": (None if np.isnan(room_map.peak_range_m) else room_map.peak_range_m),
            "x_m": room_map.x_m, "y_m": room_map.y_m, "confidence": room_map.confidence,
            "num_angles": int(room_map.angles_deg.size), "num_ranges": int(room_map.ranges_m.size),
        }, f, indent=2)
    return p
