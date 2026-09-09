"""Test-only synthetic CSI helpers.

`encodeFrame` / `generateRawFrames` build raw wire-format bytes for exercising `FrameParser` and
have no runtime caller, so they stay test-only here. `generateStream` and `generatePairedRecording`
live in `wavetrace.Synthetic` — import them from there.
"""

import numpy as np

__all__ = ["encodeFrame", "generateRawFrames"]


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
