"""Test-only synthetic CSI helpers.

`encode_frame` / `generate_raw_frames` build raw wire-format bytes for exercising `FrameParser` and
have no runtime caller, so they stay test-only here. `generate_stream` and `generate_paired_recording`
live in `wavetrace.synthetic` — import them from there.
"""

import numpy as np

__all__ = ["encode_frame", "generate_raw_frames"]


def encode_frame(real_grid: np.ndarray, imaginary_grid: np.ndarray) -> np.ndarray:
    """Pack integer I/Q grids into the ESP32 wire layout that FrameParser decodes:
    interleaved bytes [imag, real] per sample, **imaginary first**, as two's-complement int8
    held in uint8. real_grid/imaginary_grid are int grids in [-128, 127], row-major (antenna x subcarrier).
    The `& 0xFF` reproduces the on-wire unsigned byte, exercising the parser's sign fixup."""
    real = np.asarray(real_grid).astype(np.int16).ravel()
    imag = np.asarray(imaginary_grid).astype(np.int16).ravel()
    raw = np.empty(real.size * 2, dtype=np.uint8)
    raw[0::2] = (imag & 0xFF).astype(np.uint8)  # imaginary first
    raw[1::2] = (real & 0xFF).astype(np.uint8)
    return raw


def generate_raw_frames(
    *, num_antennas: int, num_subcarriers: int, num_frames: int, seed: int | None = None
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random int8 I/Q frames as (raw_uint8_bytes, expected_complex64_grid) pairs — the exact
    decode FrameParser must reproduce, including negative I/Q. O(num_frames · num_antennas · num_subcarriers)."""
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(num_frames):
        real = rng.integers(-128, 128, size=(num_antennas, num_subcarriers))
        imag = rng.integers(-128, 128, size=(num_antennas, num_subcarriers))
        raw = encode_frame(real, imag)
        expected = (real + 1j * imag).astype(np.complex64)
        out.append((raw, expected))
    return out
