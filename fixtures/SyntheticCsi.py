"""Synthetic CSI generator for DSP unit tests — no hardware required.

Builds a CsiFrame stream with a known ground truth:
  - static multipath channel  H0[a][k] = A·e^{jψ}      (distinct per antenna)
  - a periodic motion that modulates phase per subcarrier (proxy for the per-subcarrier
    frequency spread that makes the subcarrier-ratio method sensitive to motion)
  - an OPTIONAL periodic amplitude envelope (breathing-like) modulating |H| per subcarrier — so
    amplitude/PSD/turbulence features (Phase 4) have meaningful synthetic data. The per-subcarrier
    depth varies, so the cross-subcarrier pattern changes over time and the modulation survives the
    per-frame mean normalization a GainLock would apply. Off by default (amplitudeDepth=0).
  - a common-mode hardware clock offset (CFO) shared by all subcarriers — must cancel under
    conjugate-multiply / subcarrier-ratio (REFERENCE_DIGEST §2.2)
  - complex Gaussian noise

This validates the DSP pipeline only. It cannot fake posture/weapon signatures — recognition
models train on real camera-labeled recordings (plan.md Phase 5). All signal parameters are
explicit (no baked-in dimensions): they depend on the still-undecided sensor hardware (§3).
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
    """Generate a CsiFrame stream plus its ground truth. O(numFrames · numAntennas · numSubcarriers)."""
    rng = np.random.default_rng(seed)

    amp = rng.uniform(0.5, 1.5, size=(numAntennas, numSubcarriers))
    psi = rng.uniform(-np.pi, np.pi, size=(numAntennas, numSubcarriers))
    h0 = (amp * np.exp(1j * psi)).astype(np.complex64)

    # Per-subcarrier motion sensitivity in [-1, 1]; the *difference* between two subcarriers is
    # what carries the periodic motion into angle(s_i · conj(s_j)).
    scale = np.linspace(-1.0, 1.0, numSubcarriers) if numSubcarriers > 1 else np.zeros(1)
    # Per-subcarrier amplitude-modulation depth in [0.5, 1.5] (deterministic, no rng draw so existing
    # seeded streams are unchanged). Distinct per subcarrier -> the cross-subcarrier shape varies in
    # time, so turbulence/amplitude features see real signal even after per-frame mean normalization.
    ampScale = np.linspace(0.5, 1.5, numSubcarriers) if numSubcarriers > 1 else np.ones(1)

    times = np.arange(numFrames) / sampleRateHz
    frames: list[CsiFrame] = []
    for idx in range(numFrames):
        t = times[idx]
        motionPhase = perturbationDepth * scale * np.sin(2 * np.pi * perturbationHz * t)
        cfoPhase = 2 * np.pi * cfoHz * t  # common-mode → cancels in the subcarrier ratio
        rot = np.exp(1j * (motionPhase + cfoPhase)).astype(np.complex64)  # (numSubcarriers,)
        # |H| envelope: 1 + depth·sens·sin(2π f t), kept > 0 for sane depths (amplitudeDepth·1.5 < 1).
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


def encodeFrame(realIQ: np.ndarray, imagIQ: np.ndarray) -> np.ndarray:
    """Pack integer I/Q grids into the ESP32 wire layout that FrameParser decodes (§2.1):
    interleaved bytes [imag, real] per sample, **imaginary first**, as two's-complement int8
    held in uint8. realIQ/imagIQ are int grids in [-128, 127], row-major (antenna x subcarrier).
    The `& 0xFF` reproduces the on-wire unsigned byte, exercising the parser's sign fixup."""
    real = np.asarray(realIQ).astype(np.int16).ravel()
    imag = np.asarray(imagIQ).astype(np.int16).ravel()
    raw = np.empty(real.size * 2, dtype=np.uint8)
    raw[0::2] = (imag & 0xFF).astype(np.uint8)  # imaginary first
    raw[1::2] = (real & 0xFF).astype(np.uint8)
    return raw


def generateRawFrames(
    *, numAntennas: int, numSubcarriers: int, numFrames: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random int8 I/Q frames as (raw_uint8_bytes, expected_complex64_grid) pairs — the exact
    decode FrameParser must reproduce, including negative I/Q. O(numFrames · numAntennas · numSubcarriers)."""
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(numFrames):
        real = rng.integers(-128, 128, size=(numAntennas, numSubcarriers))
        imag = rng.integers(-128, 128, size=(numAntennas, numSubcarriers))
        raw = encodeFrame(real, imag)
        expected = (real + 1j * imag).astype(np.complex64)
        out.append((raw, expected))
    return out
