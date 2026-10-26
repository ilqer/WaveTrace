"""Synthetic CSI and paired labels, behind `wavetrace.entrypoints.cli`'s `--synthetic` mode. The raw
wire-format helpers (`encode_frame` / `generate_raw_frames`) live in `fixtures/synthetic_csi.py` and are
used only by tests.

  - CSI side  → a CsiFrame stream (`generate_stream`); frame.timestamp = the TRUE world time on the
                CSI host clock. Validates the DSP pipeline only — it cannot fake real posture/weapon
                signatures; recognition models train on real camera-labeled recordings.
  - Camera side (`generate_paired_recording`) → per-frame "observations" at camera_frame_rate_hz on a
                shared timeline. Each observation's CONTENT reflects the true world at its true capture
                time, but its recorded TIMESTAMP is on a *skewed* clock: cam_ts = true_t +
                clock_offset_s + jitter — the make-or-break hazard a constant clock offset silently
                mislabels CSI windows. With discrete camera frames the offset IS observable as the
                systematic component of the matched delta-t, so alignment can measure it.

Labels are BINARY (presence / weapon) carried by the core `Label`, but the raw box/keypoints and the
optional weapon `position` (location-chip path) are preserved so later location/heatmap work needs
no re-run.
"""

import numpy as np

from wavetrace import CsiFrame


def generate_stream(
    *,
    num_antennas: int,
    num_subcarriers: int,
    sample_rate_hz: float,
    num_frames: int,
    perturbation_hz: float,
    perturbation_depth: float,
    cfo_hz: float,
    noise_std: float,
    amplitude_hz: float = 0.0,
    amplitude_depth: float = 0.0,
    seed: int | None = None,
) -> tuple[list[CsiFrame], dict]:
    """Generate a CsiFrame stream plus its ground truth. O(num_frames · num_antennas · num_subcarriers).

    Models: a static multipath channel H0[a][k] = A·e^{jpsi} (distinct per antenna); a periodic
    motion that modulates phase per subcarrier (proxy for the per-subcarrier frequency spread that
    makes the subcarrier-ratio method sensitive to motion); an optional periodic amplitude envelope
    (breathing-like) modulating |H| per subcarrier, with per-subcarrier depth so the cross-subcarrier
    pattern changes over time and survives a GainLock's per-frame mean normalization; a common-mode
    hardware clock offset (CFO) shared by all subcarriers, which must cancel under conjugate-multiply
    / subcarrier-ratio; and complex Gaussian noise."""
    rng = np.random.default_rng(seed)

    amplitude = rng.uniform(0.5, 1.5, size=(num_antennas, num_subcarriers))
    psi = rng.uniform(-np.pi, np.pi, size=(num_antennas, num_subcarriers))
    static_channel = (amplitude * np.exp(1j * psi)).astype(np.complex64)

    # Per-subcarrier motion sensitivity in [-1, 1]; the difference between two subcarriers
    # carries the periodic motion into angle(s_i · conj(s_j)).
    motion_sensitivity = np.linspace(-1.0, 1.0, num_subcarriers) if num_subcarriers > 1 else np.zeros(1)
    # Per-subcarrier amplitude-modulation depth in [0.5, 1.5]; varies per subcarrier so it survives
    # per-frame mean normalization.
    amplitude_scale = np.linspace(0.5, 1.5, num_subcarriers) if num_subcarriers > 1 else np.ones(1)

    times = np.arange(num_frames) / sample_rate_hz
    frames: list[CsiFrame] = []
    for frame_index in range(num_frames):
        timestamp_s = times[frame_index]
        motion_phase = (perturbation_depth * motion_sensitivity
                        * np.sin(2 * np.pi * perturbation_hz * timestamp_s))
        cfo_phase = 2 * np.pi * cfo_hz * timestamp_s  # common-mode, cancels in the subcarrier ratio
        phase_rotor = np.exp(1j * (motion_phase + cfo_phase)).astype(np.complex64)  # (num_subcarriers,)
        # |H| envelope: 1 + depth·sens·sin(2π f t); stays positive for sane depths (amplitude_depth·1.5 < 1)
        amplitude_envelope = (
            1.0 + amplitude_depth * amplitude_scale * np.sin(2 * np.pi * amplitude_hz * timestamp_s)
        ).astype(np.float32)
        noise = (
            rng.normal(0.0, noise_std, (num_antennas, num_subcarriers))
            + 1j * rng.normal(0.0, noise_std, (num_antennas, num_subcarriers))
        ).astype(np.complex64)
        grid = (static_channel * amplitude_envelope[None, :] * phase_rotor[None, :]
                + noise).astype(np.complex64)

        frame = CsiFrame(num_antennas, num_subcarriers)
        frame.timestamp = float(timestamp_s)
        frame.grid[:, :] = grid  # zero-copy write into the native buffer
        frames.append(frame)

    ground_truth = {
        "perturbation_hz": perturbation_hz,
        "amplitude_hz": amplitude_hz,
        "amplitude_depth": amplitude_depth,
        "cfo_hz": cfo_hz,
        "sample_rate_hz": sample_rate_hz,
        "num_frames": num_frames,
        "num_antennas": num_antennas,
        "num_subcarriers": num_subcarriers,
    }
    return frames, ground_truth


def _in_spans(timestamp: float, spans) -> bool:
    """True if timestamp falls in any [start, end) span."""
    return any(start <= timestamp < end for start, end in spans)


def generate_paired_recording(
    *,
    num_antennas: int,
    num_subcarriers: int,
    sample_rate_hz: float,
    duration_s: float,
    camera_frame_rate_hz: float,
    clock_offset_s: float = 0.0,
    jitter_std_s: float = 0.0,
    presence_spans=(),
    weapon_spans=(),
    weapon_position=(0.45, 0.55, 0.10, 0.20),
    perturbation_hz: float = 1.0,
    perturbation_depth: float = 0.5,
    cfo_hz: float = 50.0,
    noise_std: float = 0.01,
    amplitude_hz: float = 0.3,
    amplitude_depth: float = 0.2,
    presence_turbulence_std: float = 0.0,
    weapon_signature_depth: float = 0.0,
    session_id: str = "",
    subject_id: str = "",
    seed: int | None = None,
) -> tuple[list, list[dict], dict]:
    """Paired (CSI frames, camera observations, ground truth) on a shared timeline.

    presence_spans / weapon_spans: iterables of (start, end) in TRUE seconds where a person / weapon is
    present. Camera observations are emitted at camera_frame_rate_hz with timestamps skewed by
    clock_offset_s + N(0, jitter_std_s). O(num_frames·A·S + numCameraFrames).

    presence_turbulence_std: inside a presence span each frame's grid gets a random
    per-(antenna, subcarrier) complex jitter (1+N(0,std))·e^{jN(0,std)} — the physical proxy for the
    dynamic multipath a human body adds, so present windows carry higher amplitude/phase turbulence
    (std/MAD/waveform-length) than absent ones and a presence head becomes learnable on synthetic
    data. The jitter varies per subcarrier, so it survives a GainLock's per-frame mean normalization.
    Drawn from its own rng (seed+2) and only when std > 0, so a seeded stream with the default off
    is unchanged by it. session_id/subject_id are stamped into `truth` for the leave-one-group-out
    folds.

    weapon_signature_depth: inside a weapon span each frame's per-antenna magnitude profile is
    FLATTENED toward its cross-subcarrier mean ((1-d)·|H| + d·mean|H|, phase kept) plus a slight bulk
    attenuation (×(1-0.15d)): the proxy for a coherent metal reflection, which lowers the
    inter-subcarrier σ²[p] the weapon head reads. Deterministic, so a seeded stream is unchanged
    while it is off. A real metal signature depends on geometry and orientation, so this exercises
    the plumbing rather than the physics."""
    num_frames = int(round(duration_s * sample_rate_hz))
    frames, _ = generate_stream(
        num_antennas=num_antennas,
        num_subcarriers=num_subcarriers,
        sample_rate_hz=sample_rate_hz,
        num_frames=num_frames,
        perturbation_hz=perturbation_hz,
        perturbation_depth=perturbation_depth,
        cfo_hz=cfo_hz,
        noise_std=noise_std,
        amplitude_hz=amplitude_hz,
        amplitude_depth=amplitude_depth,
        seed=seed,
    )

    presence = [(float(start), float(end)) for start, end in presence_spans]
    weapon = [(float(start), float(end)) for start, end in weapon_spans]

    # presence -> signal modulation. Own rng (seed+2) so CSI/camera streams stay intact.
    if presence_turbulence_std > 0 and presence:
        turbulence_rng = np.random.default_rng(None if seed is None else seed + 2)
        for frame in frames:
            if _in_spans(frame.timestamp, presence):
                grid = np.asarray(frame.grid)
                amplitude_jitter = turbulence_rng.normal(0.0, presence_turbulence_std, grid.shape)
                phase_jitter = turbulence_rng.normal(0.0, presence_turbulence_std, grid.shape)
                grid *= ((1.0 + amplitude_jitter) * np.exp(1j * phase_jitter)).astype(np.complex64)

    # weapon -> σ²[p] signature (flatten toward the per-antenna mean magnitude).
    if weapon_signature_depth > 0 and weapon:
        depth = float(weapon_signature_depth)
        for frame in frames:
            if _in_spans(frame.timestamp, weapon):
                grid = np.asarray(frame.grid)
                magnitude = np.abs(grid)
                target = ((1.0 - depth) * magnitude + depth * magnitude.mean(axis=1, keepdims=True)) * (
                    1.0 - 0.15 * depth
                )
                # rescale magnitude, keep phase; guard near-zero noise cells
                grid *= (target / np.maximum(magnitude, 1e-9)).astype(np.complex64)
    # +1 keeps the camera clock's jitter stream independent of the CSI noise stream
    rng = np.random.default_rng(None if seed is None else seed + 1)
    num_camera_frames = int(round(duration_s * camera_frame_rate_hz))
    observations: list[dict] = []
    for j in range(num_camera_frames):
        true_timestamp_s = j / camera_frame_rate_hz
        jitter = float(rng.normal(0.0, jitter_std_s)) if jitter_std_s > 0 else 0.0
        camera_timestamp_s = true_timestamp_s + clock_offset_s + jitter
        is_present = _in_spans(true_timestamp_s, presence)
        has_weapon = _in_spans(true_timestamp_s, weapon)
        raw = {
            "present": is_present,
            "weapon": has_weapon,
            # coarse person box (normalized) when present, else None
            "bbox": [0.40, 0.30, 0.20, 0.55] if is_present else None,
            "keypoints": [0.5, 0.2, 0.5, 0.5, 0.5, 0.8] if is_present else [],
            # weapon location ground truth for the location-chip path, else None
            "position": list(weapon_position) if has_weapon else None,
        }
        observations.append({"t": float(camera_timestamp_s), "true_t": float(true_timestamp_s), "raw": raw})

    truth = {
        "clock_offset_s": float(clock_offset_s),
        "jitter_std_s": float(jitter_std_s),
        "camera_fps": float(camera_frame_rate_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "duration_s": float(duration_s),
        "num_frames": num_frames,
        "num_camera_frames": num_camera_frames,
        "presence_spans": presence,
        "weapon_spans": weapon,
        "presence_turbulence_std": float(presence_turbulence_std),
        "weapon_signature_depth": float(weapon_signature_depth),
        "session_id": str(session_id),
        "subject_id": str(subject_id),
    }
    return frames, observations, truth
