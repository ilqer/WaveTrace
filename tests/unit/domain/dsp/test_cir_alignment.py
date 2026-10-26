"""UWB tap alignment: first-path re-anchoring and the cross-RX cross-spectrum.

The chip's CIR window start moves between packets. Re-anchoring must put every RX's first path on
one column; the cross-spectrum must not change under any common window shift or common phase.
"""

import numpy as np
import pytest

from wavetrace.domain.dsp.cir_alignment import cross_spectrum, reanchor_taps

NUM_TAPS = 64


def _two_rx_taps(rng):
    """(2, N) taps with paths at taps 20..24, well inside the window."""
    taps = np.zeros((2, NUM_TAPS), dtype=np.complex64)
    taps[:, 20:25] = rng.standard_normal((2, 5)) + 1j * rng.standard_normal((2, 5))
    return taps


def _circular_shift(taps, shift_taps):
    """Delay every row by `shift_taps` (may be fractional) as e^{-j2πkΔ/N} on the FFT bins."""
    phase_ramp = np.exp(-2j * np.pi * np.fft.fftfreq(NUM_TAPS) * shift_taps)
    return np.fft.ifft(np.fft.fft(taps, axis=-1) * phase_ramp, axis=-1).astype(np.complex64)


def test_reanchor_puts_each_first_path_on_anchor_and_zero_fills_edges():
    taps = np.arange(1, 2 * NUM_TAPS + 1, dtype=np.complex64).reshape(2, NUM_TAPS)
    first_path_columns = np.array([10, 40])
    anchor_column = 16
    out = np.full((2, NUM_TAPS), 99.0, dtype=np.complex64)
    result = reanchor_taps(taps, first_path_columns, anchor_column, out=out)
    assert result is out
    for rx, first_path_column in enumerate(first_path_columns):
        assert result[rx, anchor_column] == taps[rx, first_path_column]
    # RX 0 moves right by 6: its first 6 columns are zero; RX 1 moves left by 24: its last 24 are zero
    assert np.all(result[0, :6] == 0) and result[0, 6] == taps[0, 0]
    assert np.all(result[1, -24:] == 0) and result[1, -25] == taps[1, -1]


def test_reanchor_allocates_when_out_is_none():
    taps = np.ones((2, NUM_TAPS), dtype=np.complex64)
    result = reanchor_taps(taps, np.array([5, 5]), 5)
    assert result.shape == taps.shape and result is not taps
    np.testing.assert_array_equal(result, taps)


def test_cross_spectrum_is_invariant_to_common_window_shift():
    rng = np.random.default_rng(0)
    taps = _two_rx_taps(rng)
    reference = cross_spectrum(taps)
    assert reference.shape == (1, NUM_TAPS)
    for shift_taps in (1, 2):  # integer: a plain roll, paths stay inside the window
        np.testing.assert_allclose(cross_spectrum(np.roll(taps, shift_taps, axis=-1)), reference,
                                   rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cross_spectrum(_circular_shift(taps, 2.5)), reference, rtol=1e-4, atol=1e-4)


def test_cross_spectrum_is_invariant_to_common_phase_and_reuses_out():
    rng = np.random.default_rng(1)
    taps = np.vstack([_two_rx_taps(rng), _two_rx_taps(rng)[:1]])  # R = 3
    reference = cross_spectrum(taps)
    out = np.empty((2, NUM_TAPS), dtype=np.complex128)
    rotated = cross_spectrum(taps * np.exp(1j * rng.uniform(0, 2 * np.pi)), out=out)
    assert rotated is out
    np.testing.assert_allclose(rotated, reference, rtol=1e-4, atol=1e-4)


def test_cross_spectrum_needs_two_rx():
    with pytest.raises(ValueError, match="RX"):
        cross_spectrum(np.ones((1, NUM_TAPS), dtype=np.complex64))


def test_reanchor_rejects_out_aliasing_taps_float_columns_and_real_out():
    taps = np.arange(16, dtype=np.complex64).reshape(2, 8)
    with pytest.raises(ValueError):
        reanchor_taps(taps, np.array([1, 1]), 3, out=taps)
    with pytest.raises(TypeError):
        reanchor_taps(taps, np.array([1.0, 2.7]), 3)
    with pytest.raises(TypeError):
        reanchor_taps(taps, np.array([1, 1]), 3, out=np.empty((2, 8), dtype=np.float32))
