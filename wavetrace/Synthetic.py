"""Synthetic CSI + paired-label generation, used at runtime by `wavetrace.Cli`'s `--synthetic`
mode (`Cli._sourceFromArgs` calls `generatePairedRecording` below). `fixtures/SyntheticCsi.py`
holds the raw wire-format helpers (`encodeFrame` / `generateRawFrames`) used only by tests.

  - CSI side  → a CsiFrame stream (`generateStream`); frame.timestamp = the TRUE world time on the
                CSI host clock. Validates the DSP pipeline only — it cannot fake real posture/weapon
                signatures; recognition models train on real camera-labeled recordings.
  - Camera side (`generatePairedRecording`) → per-frame "observations" at cameraFps on a shared
                timeline. Each observation's CONTENT reflects the true world at its true capture
                time, but its recorded TIMESTAMP is on a *skewed* clock: cam_ts = true_t +
                clockOffsetS + jitter — the make-or-break hazard a constant clock offset silently
                mislabels CSI windows. With discrete camera frames the offset IS observable as the
                systematic component of the matched delta-t, so alignment can measure it.

Labels are BINARY (presence / weapon) carried by the core `Label`, but the raw box/keypoints and the
optional weapon `position` (location-chip path) are preserved so later location/heatmap work needs
no re-run.
"""

import numpy as np

from wavetrace import CsiFrame


def generateStream(
    *,
    numAntennas: int,
    numSubcarriers: int,
    sampleRateHz: float,
    numFrames: int,
    perturbationHz: float,
    perturbationDepth: float,
    cfoHz: float,
    noiseStd: float,
    amplitudeHz: float = 0.0,
    amplitudeDepth: float = 0.0,
    seed: int | None = None,
) -> tuple[list[CsiFrame], dict]:
    """Generate a CsiFrame stream plus its ground truth. O(numFrames · numAntennas · numSubcarriers).

    Models: a static multipath channel H0[a][k] = A·e^{jpsi} (distinct per antenna); a periodic
    motion that modulates phase per subcarrier (proxy for the per-subcarrier frequency spread that
    makes the subcarrier-ratio method sensitive to motion); an optional periodic amplitude envelope
    (breathing-like) modulating |H| per subcarrier, with per-subcarrier depth so the cross-subcarrier
    pattern changes over time and survives a GainLock's per-frame mean normalization; a common-mode
    hardware clock offset (CFO) shared by all subcarriers, which must cancel under conjugate-multiply
    / subcarrier-ratio; and complex Gaussian noise."""
    rng = np.random.default_rng(seed)

    amplitude = rng.uniform(0.5, 1.5, size=(numAntennas, numSubcarriers))
    psi = rng.uniform(-np.pi, np.pi, size=(numAntennas, numSubcarriers))
    h0 = (amplitude * np.exp(1j * psi)).astype(np.complex64)

    # Per-subcarrier motion sensitivity in [-1, 1]; the difference between two subcarriers
    # carries the periodic motion into angle(s_i · conj(s_j)).
    scale = np.linspace(-1.0, 1.0, numSubcarriers) if numSubcarriers > 1 else np.zeros(1)
    # Per-subcarrier amplitude-modulation depth in [0.5, 1.5]; varies per subcarrier so it survives per-frame mean normalization.
    amplitude_scale = np.linspace(0.5, 1.5, numSubcarriers) if numSubcarriers > 1 else np.ones(1)

    times = np.arange(numFrames) / sampleRateHz
    frames: list[CsiFrame] = []
    for frame_index in range(numFrames):
        timestamp_s = times[frame_index]
        motion_phase = perturbationDepth * scale * np.sin(2 * np.pi * perturbationHz * timestamp_s)
        cfo_phase = 2 * np.pi * cfoHz * timestamp_s  # common-mode, cancels in the subcarrier ratio
        phase_rotor = np.exp(1j * (motion_phase + cfo_phase)).astype(np.complex64)  # (numSubcarriers,)
        # |H| envelope: 1 + depth·sens·sin(2π f t); stays positive for sane depths (amplitudeDepth·1.5 < 1)
        amplitude_envelope = (
            1.0 + amplitudeDepth * amplitude_scale * np.sin(2 * np.pi * amplitudeHz * timestamp_s)
        ).astype(np.float32)
        noise = (
            rng.normal(0.0, noiseStd, (numAntennas, numSubcarriers))
            + 1j * rng.normal(0.0, noiseStd, (numAntennas, numSubcarriers))
        ).astype(np.complex64)
        grid = (h0 * amplitude_envelope[None, :] * phase_rotor[None, :] + noise).astype(np.complex64)

        frame = CsiFrame(numAntennas, numSubcarriers)
        frame.timestamp = float(timestamp_s)
        frame.grid[:, :] = grid  # zero-copy write into the native buffer
        frames.append(frame)

    ground_truth = {
        "perturbation_hz": perturbationHz,
        "amplitude_hz": amplitudeHz,
        "amplitude_depth": amplitudeDepth,
        "cfo_hz": cfoHz,
        "sample_rate_hz": sampleRateHz,
        "num_frames": numFrames,
        "num_antennas": numAntennas,
        "num_subcarriers": numSubcarriers,
    }
    return frames, ground_truth


def _in_spans(timestamp: float, spans) -> bool:
    """True if timestamp falls in any [start, end) span."""
    return any(start <= timestamp < end for start, end in spans)


def generatePairedRecording(
    *,
    numAntennas: int,
    numSubcarriers: int,
    sampleRateHz: float,
    durationS: float,
    cameraFps: float,
    clockOffsetS: float = 0.0,
    jitterStdS: float = 0.0,
    presenceSpans=(),
    weaponSpans=(),
    weaponPosition=(0.45, 0.55, 0.10, 0.20),
    perturbationHz: float = 1.0,
    perturbationDepth: float = 0.5,
    cfoHz: float = 50.0,
    noiseStd: float = 0.01,
    amplitudeHz: float = 0.3,
    amplitudeDepth: float = 0.2,
    presenceTurbulenceStd: float = 0.0,
    weaponSignatureDepth: float = 0.0,
    sessionId: str = "",
    subjectId: str = "",
    seed: int | None = None,
) -> tuple[list, list[dict], dict]:
    """Paired (CSI frames, camera observations, ground truth) on a shared timeline.

    presenceSpans / weaponSpans: iterables of (start, end) in TRUE seconds where a person / weapon is
    present. Camera observations are emitted at cameraFps with timestamps skewed by clockOffsetS +
    N(0, jitterStdS). O(numFrames·A·S + numCameraFrames).

    presenceTurbulenceStd: inside a presence span each frame's grid gets a random
    per-(antenna, subcarrier) complex jitter (1+N(0,std))·e^{jN(0,std)} — the physical proxy for the
    dynamic multipath a human body adds, so present windows carry higher amplitude/phase turbulence
    (std/MAD/waveform-length) than absent ones and a presence head becomes learnable on synthetic
    data. The jitter varies per subcarrier, so it survives a GainLock's per-frame mean normalization.
    Drawn from its own rng (seed+2) and only when std > 0, so prior seeded streams stay byte-identical
    (default off). sessionId/subjectId are stamped into `truth` — the group ids the leave-one-
    session/subject-out eval gate needs.

    weaponSignatureDepth: inside a weapon span each frame's per-antenna magnitude profile is
    FLATTENED toward its cross-subcarrier mean ((1-d)·|H| + d·mean|H|, phase kept) plus a slight bulk
    attenuation (×(1-0.15d)) — the proxy for a coherent metal reflection, which lowers the
    inter-subcarrier σ²[p] (the weapon discriminator). Deterministic (no rng draws); default off →
    seeded streams stay byte-identical. Even more artificial than the presence turbulence (a real
    metal signature is geometry/orientation-dependent) — plumbing only."""
    numFrames = int(round(durationS * sampleRateHz))
    frames, _ = generateStream(
        numAntennas=numAntennas,
        numSubcarriers=numSubcarriers,
        sampleRateHz=sampleRateHz,
        numFrames=numFrames,
        perturbationHz=perturbationHz,
        perturbationDepth=perturbationDepth,
        cfoHz=cfoHz,
        noiseStd=noiseStd,
        amplitudeHz=amplitudeHz,
        amplitudeDepth=amplitudeDepth,
        seed=seed,
    )

    presence = [(float(start), float(end)) for start, end in presenceSpans]
    weapon = [(float(start), float(end)) for start, end in weaponSpans]

    # presence -> signal modulation. Own rng (seed+2) so CSI/camera streams stay intact.
    if presenceTurbulenceStd > 0 and presence:
        turbulence_rng = np.random.default_rng(None if seed is None else seed + 2)
        for frame in frames:
            if _in_spans(frame.timestamp, presence):
                grid = np.asarray(frame.grid)
                amplitude_jitter = turbulence_rng.normal(0.0, presenceTurbulenceStd, grid.shape)
                phase_jitter = turbulence_rng.normal(0.0, presenceTurbulenceStd, grid.shape)
                grid *= ((1.0 + amplitude_jitter) * np.exp(1j * phase_jitter)).astype(np.complex64)

    # weapon -> σ²[p] signature (flatten toward the per-antenna mean magnitude).
    if weaponSignatureDepth > 0 and weapon:
        depth = float(weaponSignatureDepth)
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
    num_camera_frames = int(round(durationS * cameraFps))
    observations: list[dict] = []
    for j in range(num_camera_frames):
        true_timestamp_s = j / cameraFps
        jitter = float(rng.normal(0.0, jitterStdS)) if jitterStdS > 0 else 0.0
        camera_timestamp_s = true_timestamp_s + clockOffsetS + jitter
        is_present = _in_spans(true_timestamp_s, presence)
        has_weapon = _in_spans(true_timestamp_s, weapon)
        raw = {
            "present": is_present,
            "weapon": has_weapon,
            # coarse person box (normalized) when present, else None
            "bbox": [0.40, 0.30, 0.20, 0.55] if is_present else None,
            "keypoints": [0.5, 0.2, 0.5, 0.5, 0.5, 0.8] if is_present else [],
            # weapon location ground truth for the location-chip path, else None
            "position": list(weaponPosition) if has_weapon else None,
        }
        observations.append({"t": float(camera_timestamp_s), "true_t": float(true_timestamp_s), "raw": raw})

    truth = {
        "clock_offset_s": float(clockOffsetS),
        "jitter_std_s": float(jitterStdS),
        "camera_fps": float(cameraFps),
        "sample_rate_hz": float(sampleRateHz),
        "duration_s": float(durationS),
        "num_frames": numFrames,
        "num_camera_frames": num_camera_frames,
        "presence_spans": presence,
        "weapon_spans": weapon,
        "presence_turbulence_std": float(presenceTurbulenceStd),
        "weapon_signature_depth": float(weaponSignatureDepth),
        "session_id": str(sessionId),
        "subject_id": str(subjectId),
    }
    return frames, observations, truth
