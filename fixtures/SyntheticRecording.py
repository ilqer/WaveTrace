"""Synthetic PAIRED recording (CSI + camera label stream) for Phase-5 unit tests — no hardware.

Builds two time-aligned streams on a shared timeline so the ground-truth pipeline (CameraLabeler,
Align, DatasetBuilder) and especially the **time-sync error measurement** can be tested with no
camera or capture rig:

  - CSI side  → a CsiFrame stream (reuses SyntheticCsi.generateStream); frame.timestamp = the TRUE
                world time on the CSI host clock.
  - Camera side → per-frame "observations" at cameraFps. Each observation's CONTENT reflects the true
                world at its true capture time, but its recorded TIMESTAMP is on a *skewed* clock:
                cam_ts = true_t + clockOffsetS + jitter.  This is the make-or-break hazard
                (REFERENCE_DIGEST §0B): a constant clock offset silently mislabels CSI windows. With
                discrete camera frames the offset IS observable as the systematic component of the
                matched Δt, so Align can MEASURE it (Phase-5 DoD: sync error measured + bounded).

Labels are BINARY (presence A / weapon E) carried by the core `Label`, but the raw box/keypoints and
the optional weapon `position` (location-chip path) are preserved so the later location/heatmap work
needs no re-run. **This validates the alignment/dataset PLUMBING only — it cannot fake real
posture/weapon CSI signatures (plan §2.2); real recordings are required before any accuracy claim.**
"""

import numpy as np

from fixtures.SyntheticCsi import generateStream


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

    presenceTurbulenceStd (Phase 6a): inside a presence span each frame's grid gets a random
    per-(antenna, subcarrier) complex jitter (1+N(0,std))·e^{jN(0,std)} — the physical proxy for the
    dynamic multipath a human body adds, so present windows carry higher amplitude/phase turbulence
    (std/MAD/waveform-length) than absent ones and a presence head becomes learnable on synthetic
    data. The jitter varies per subcarrier, so it survives a GainLock's per-frame mean normalization.
    Drawn from its own rng (seed+2) and only when std > 0, so prior seeded streams stay byte-identical
    (default off). sessionId/subjectId are stamped into `truth` — the group ids the leave-one-
    session/subject-out eval gate needs (rev-7 #1).

    weaponSignatureDepth (Phase 7p-a): inside a weapon span each frame's per-antenna magnitude
    profile is FLATTENED toward its cross-subcarrier mean ((1-d)·|H| + d·mean|H|, phase kept) plus a
    slight bulk attenuation (×(1-0.15d)) — the proxy for a coherent metal reflection, which lowers
    the inter-subcarrier σ²[p] (the Yousaf/LUMS weapon discriminator). Deterministic (no rng draws);
    default off → seeded streams stay byte-identical. ⚠️ Even more artificial than the presence
    turbulence (a real metal signature is geometry/orientation-dependent) — plumbing only."""
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

    # Phase 6a: presence -> SIGNAL modulation. Own rng (seed+2) keeps the CSI/camera streams intact.
    if presenceTurbulenceStd > 0 and presence:
        turbRng = np.random.default_rng(None if seed is None else seed + 2)
        for fr in frames:
            if _in_spans(fr.timestamp, presence):
                g = np.asarray(fr.grid)
                ampJ = turbRng.normal(0.0, presenceTurbulenceStd, g.shape)
                phJ = turbRng.normal(0.0, presenceTurbulenceStd, g.shape)
                g *= ((1.0 + ampJ) * np.exp(1j * phJ)).astype(np.complex64)

    # Phase 7p-a: weapon -> σ²[p] signature (flatten toward the per-antenna mean magnitude).
    if weaponSignatureDepth > 0 and weapon:
        d = float(weaponSignatureDepth)
        for fr in frames:
            if _in_spans(fr.timestamp, weapon):
                g = np.asarray(fr.grid)
                mag = np.abs(g)
                target = ((1.0 - d) * mag + d * mag.mean(axis=1, keepdims=True)) * (1.0 - 0.15 * d)
                # rescale magnitude, keep phase (guard the near-zero noise cells)
                g *= (target / np.maximum(mag, 1e-9)).astype(np.complex64)
    # +1 so the camera clock's jitter stream is independent of the CSI noise stream.
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
            # a coarse person box (normalized) when present; None otherwise
            "bbox": [0.40, 0.30, 0.20, 0.55] if isPresent else None,
            "keypoints": [0.5, 0.2, 0.5, 0.5, 0.5, 0.8] if isPresent else [],
            # weapon location ground truth (location-chip / segment-train path); None if no weapon
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
