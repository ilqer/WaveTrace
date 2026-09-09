"""Synthetic CSI + paired-label generation. Ships as part of the `wavetrace` package (not a test
fixture) because `wavetrace.Cli`'s `--synthetic` mode depends on it at runtime — see
`generatePairedRecording` below, called from `Cli._sourceFromArgs`. This is the only home for
`generateStream` / `generatePairedRecording`; `fixtures/SyntheticCsi.py` keeps only the raw
wire-format helpers (`encodeFrame` / `generateRawFrames`) that have no runtime caller.

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

    amp = rng.uniform(0.5, 1.5, size=(numAntennas, numSubcarriers))
    psi = rng.uniform(-np.pi, np.pi, size=(numAntennas, numSubcarriers))
    h0 = (amp * np.exp(1j * psi)).astype(np.complex64)

    # Per-subcarrier motion sensitivity in [-1, 1]; the difference between two subcarriers
    # carries the periodic motion into angle(s_i · conj(s_j)).
    scale = np.linspace(-1.0, 1.0, numSubcarriers) if numSubcarriers > 1 else np.zeros(1)
    # Per-subcarrier amplitude-modulation depth in [0.5, 1.5]; varies per subcarrier so it survives per-frame mean normalization.
    ampScale = np.linspace(0.5, 1.5, numSubcarriers) if numSubcarriers > 1 else np.ones(1)

    times = np.arange(numFrames) / sampleRateHz
    frames: list[CsiFrame] = []
    for idx in range(numFrames):
        t = times[idx]
        motionPhase = perturbationDepth * scale * np.sin(2 * np.pi * perturbationHz * t)
        cfoPhase = 2 * np.pi * cfoHz * t  # common-mode, cancels in the subcarrier ratio
        rot = np.exp(1j * (motionPhase + cfoPhase)).astype(np.complex64)  # (numSubcarriers,)
        # |H| envelope: 1 + depth·sens·sin(2π f t); stays positive for sane depths (amplitudeDepth·1.5 < 1)
        ampEnv = (1.0 + amplitudeDepth * ampScale * np.sin(2 * np.pi * amplitudeHz * t)).astype(np.float32)
        noise = (
            rng.normal(0.0, noiseStd, (numAntennas, numSubcarriers))
            + 1j * rng.normal(0.0, noiseStd, (numAntennas, numSubcarriers))
        ).astype(np.complex64)
        grid = (h0 * ampEnv[None, :] * rot[None, :] + noise).astype(np.complex64)

        frame = CsiFrame(numAntennas, numSubcarriers)
        frame.timestamp = float(t)
        frame.grid[:, :] = grid  # zero-copy write into the native buffer
        frames.append(frame)

    groundTruth = {
        "perturbation_hz": perturbationHz,
        "amplitude_hz": amplitudeHz,
        "amplitude_depth": amplitudeDepth,
        "cfo_hz": cfoHz,
        "sample_rate_hz": sampleRateHz,
        "num_frames": numFrames,
        "num_antennas": numAntennas,
        "num_subcarriers": numSubcarriers,
    }
    return frames, groundTruth


def _in_spans(t: float, spans) -> bool:
    """True if t falls in any [start, end) span."""
    return any(s <= t < e for s, e in spans)


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

    presence = [(float(s), float(e)) for s, e in presenceSpans]
    weapon = [(float(s), float(e)) for s, e in weaponSpans]

    # presence -> signal modulation. Own rng (seed+2) so CSI/camera streams stay intact.
    if presenceTurbulenceStd > 0 and presence:
        turbRng = np.random.default_rng(None if seed is None else seed + 2)
        for fr in frames:
            if _in_spans(fr.timestamp, presence):
                g = np.asarray(fr.grid)
                ampJ = turbRng.normal(0.0, presenceTurbulenceStd, g.shape)
                phJ = turbRng.normal(0.0, presenceTurbulenceStd, g.shape)
                g *= ((1.0 + ampJ) * np.exp(1j * phJ)).astype(np.complex64)

    # weapon -> σ²[p] signature (flatten toward the per-antenna mean magnitude).
    if weaponSignatureDepth > 0 and weapon:
        d = float(weaponSignatureDepth)
        for fr in frames:
            if _in_spans(fr.timestamp, weapon):
                g = np.asarray(fr.grid)
                mag = np.abs(g)
                target = ((1.0 - d) * mag + d * mag.mean(axis=1, keepdims=True)) * (1.0 - 0.15 * d)
                # rescale magnitude, keep phase; guard near-zero noise cells
                g *= (target / np.maximum(mag, 1e-9)).astype(np.complex64)
    # +1 keeps the camera clock's jitter stream independent of the CSI noise stream
    rng = np.random.default_rng(None if seed is None else seed + 1)
    numCam = int(round(durationS * cameraFps))
    observations: list[dict] = []
    for j in range(numCam):
        trueT = j / cameraFps
        jitter = float(rng.normal(0.0, jitterStdS)) if jitterStdS > 0 else 0.0
        camTs = trueT + clockOffsetS + jitter
        isPresent = _in_spans(trueT, presence)
        hasWeapon = _in_spans(trueT, weapon)
        raw = {
            "present": isPresent,
            "weapon": hasWeapon,
            # coarse person box (normalized) when present, else None
            "bbox": [0.40, 0.30, 0.20, 0.55] if isPresent else None,
            "keypoints": [0.5, 0.2, 0.5, 0.5, 0.5, 0.8] if isPresent else [],
            # weapon location ground truth for the location-chip path, else None
            "position": list(weaponPosition) if hasWeapon else None,
        }
        observations.append({"t": float(camTs), "true_t": float(trueT), "raw": raw})

    truth = {
        "clock_offset_s": float(clockOffsetS),
        "jitter_std_s": float(jitterStdS),
        "camera_fps": float(cameraFps),
        "sample_rate_hz": float(sampleRateHz),
        "duration_s": float(durationS),
        "num_frames": numFrames,
        "num_camera_frames": numCam,
        "presence_spans": presence,
        "weapon_spans": weapon,
        "presence_turbulence_std": float(presenceTurbulenceStd),
        "weapon_signature_depth": float(weaponSignatureDepth),
        "session_id": str(sessionId),
        "subject_id": str(subjectId),
    }
    return frames, observations, truth
