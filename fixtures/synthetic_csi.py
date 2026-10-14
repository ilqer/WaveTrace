"""Test-only synthetic CSI helpers.

`encode_frame` / `generate_raw_frames` build raw wire-format bytes for exercising `FrameParser`.
`generate_quiet_baseline` / `calibrate_quiet_baseline` build the motionless baseline tests calibrate on.
"""

import numpy as np

from wavetrace import CsiFrame, GainLock
from wavetrace.domain.calibration import Calibration, CalibrationResult
from wavetrace.synthetic import generate_stream

__all__ = ["calibrate_quiet_baseline", "encode_frame", "generate_quiet_baseline", "generate_raw_frames"]


def encode_frame(real_grid: np.ndarray, imaginary_grid: np.ndarray) -> np.ndarray:
    """Pack integer I/Q grids into the ESP32 wire layout that FrameParser decodes:
    interleaved bytes [imag, real] per sample, **imaginary first**, as two's-complement int8
    held in uint8. real_grid/imaginary_grid are int grids in [-128, 127], row-major (antenna x subcarrier).
    The `& 0xFF` reproduces the on-wire unsigned byte, exercising the parser's sign fixup."""
    real = np.asarray(real_grid).astype(np.int16).ravel()
    imaginary = np.asarray(imaginary_grid).astype(np.int16).ravel()
    raw = np.empty(real.size * 2, dtype=np.uint8)
    raw[0::2] = (imaginary & 0xFF).astype(np.uint8)
    raw[1::2] = (real & 0xFF).astype(np.uint8)
    return raw


def generate_raw_frames(
    *, num_antennas: int, num_subcarriers: int, num_frames: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random int8 I/Q frames as (raw_uint8_bytes, expected_complex64_grid) pairs — the exact
    decode FrameParser must reproduce, including negative I/Q. O(num_frames · num_antennas · num_subcarriers)."""
    rng = np.random.default_rng(seed)
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_frames):
        real = rng.integers(-128, 128, size=(num_antennas, num_subcarriers))
        imaginary = rng.integers(-128, 128, size=(num_antennas, num_subcarriers))
        raw = encode_frame(real, imaginary)
        expected = (real + 1j * imaginary).astype(np.complex64)
        frames.append((raw, expected))
    return frames


def generate_quiet_baseline(
    *, num_antennas: int, num_subcarriers: int, sample_rate_hz: float, num_frames: int, seed: int
) -> list[CsiFrame]:
    """Empty-room stream: no motion, no CFO, only 0.005 receiver noise."""
    frames, _ = generate_stream(
        num_antennas=num_antennas, num_subcarriers=num_subcarriers, sample_rate_hz=sample_rate_hz,
        num_frames=num_frames, perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0,
        noise_std=0.005, seed=seed)
    return frames


def calibrate_quiet_baseline(
    *, num_antennas: int, num_subcarriers: int, sample_rate_hz: float, num_frames: int, seed: int,
    baseline_packets: int, nbvi_max: int | None = None,
) -> tuple[CalibrationResult, GainLock]:
    """nbvi_max=None keeps Calibration's own default; raises ValueError if num_frames < baseline_packets."""
    optional_options = {} if nbvi_max is None else {"nbvi_max": nbvi_max}
    calibration = Calibration(baseline_packets=baseline_packets, **optional_options)
    for frame in generate_quiet_baseline(
            num_antennas=num_antennas, num_subcarriers=num_subcarriers, sample_rate_hz=sample_rate_hz,
            num_frames=num_frames, seed=seed):
        calibration.observe(frame)
    return calibration.finalize(), calibration.gain_lock
