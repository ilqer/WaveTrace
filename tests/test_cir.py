"""Delay-domain super-resolution (CIR via ISTA).

A known 2-tap channel must be recovered to within one delay bin: the L1 solve
super-resolves below the 1/BW Nyquist limit and separates a strong reflector from a
weaker one (the body-vs-concealed-object case).
"""

import numpy as np

from wavetrace.recognition.cir import (
    SUBCARRIER_SPACING_HZ,
    cir_features,
    cir_from_csi,
    delay_dictionary,
    estimate_cir_taps,
)

SUBCARRIER_STEP_HZ = SUBCARRIER_SPACING_HZ


def _two_tap_csi(subcarrier_count, delays_s, amplitudes):
    """H[k] = Σ aᵢ·exp(-j2π·k·Δf·τᵢ) over contiguous subcarriers k = 0..subcarrier_count-1."""
    subcarrier_index = np.arange(subcarrier_count)[:, None]
    atoms = np.exp(-2j * np.pi * subcarrier_index * SUBCARRIER_STEP_HZ * np.asarray(delays_s)[None, :])
    return (atoms @ np.asarray(amplitudes, dtype=np.complex128)).astype(np.complex64)


def _peak_delays(cir):
    """Delays (s) of local-maxima taps above a -25 dB floor, matching the module's detector."""
    power = np.abs(cir.taps) ** 2
    floor = power.max() * 10 ** (-2.5)
    is_peak = (power > floor)
    is_peak[1:-1] &= (power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:])
    is_peak[0] = is_peak[-1] = False
    return cir.tap_delays_s[np.flatnonzero(is_peak)]


def test_two_tap_recovery_ht20():
    first_delay_s, second_delay_s = 30e-9, 80e-9
    first_amplitude = 0.8 * np.exp(1j * np.pi / 4)
    second_amplitude = 0.3 * np.exp(1j * 3 * np.pi / 4)
    csi = _two_tap_csi(56, [first_delay_s, second_delay_s], [first_amplitude, second_amplitude])

    cir = cir_from_csi(csi, oversample=3, lam=0.02, max_iterations=200)
    delay_bin_width_s = cir.tap_delays_s[1]

    assert abs(cir.dominant_delay_s - first_delay_s) <= delay_bin_width_s
    # strong path dominates (power ratio 0.64 / 0.73 ≈ 0.88)
    assert cir.dominant_ratio > 0.7
    # exactly two taps survive the -25 dB floor
    assert cir.active_tap_count == 2

    peak_delays_s = _peak_delays(cir)
    assert np.min(np.abs(peak_delays_s - first_delay_s)) <= delay_bin_width_s
    assert np.min(np.abs(peak_delays_s - second_delay_s)) <= delay_bin_width_s


def test_super_resolution_beats_nyquist():
    # Two taps closer than the 1/BW native resolution must still separate.
    subcarrier_count = 56
    native_resolution_s = 1.0 / (subcarrier_count * SUBCARRIER_STEP_HZ)  # ~57 ns
    first_delay_s, second_delay_s = 20e-9, 20e-9 + 0.6 * native_resolution_s
    csi = _two_tap_csi(subcarrier_count, [first_delay_s, second_delay_s], [1.0, 0.7])
    cir = cir_from_csi(csi, oversample=4, lam=0.01, max_iterations=300)
    assert cir.active_tap_count >= 2


def test_dictionary_is_well_conditioned():
    # Φ = dictionary, K = subcarrier_count, G = delay_grid_size: κ(Φ) ≈ 1 since ΦΦᴴ ≈ (G/K)·I
    # for a normalised sub-DFT.
    subcarrier_count, delay_grid_size = 52, 156
    dictionary = delay_dictionary(np.arange(subcarrier_count), delay_grid_size)
    gram = dictionary @ dictionary.conj().T
    diagonal = np.diag(gram).real
    assert np.allclose(diagonal, delay_grid_size / subcarrier_count, rtol=0.1)
    off_diagonal = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off_diagonal)) < 0.5 * (delay_grid_size / subcarrier_count)


def test_features_shape_and_gapped_band():
    # gapped layout (HT40-style central null) must be accepted via subcarrier_indices.
    subcarrier_indices = np.concatenate([np.arange(0, 24), np.arange(28, 52)])
    csi = _two_tap_csi(52, [25e-9, 70e-9], [1.0, 0.4])[subcarrier_indices]
    cir = cir_from_csi(csi, subcarrier_indices=subcarrier_indices, oversample=3, lam=0.02, max_iterations=150)
    features = cir_features(cir)
    assert features.shape == (5,)
    assert np.all(np.isfinite(features))


def test_rejects_bad_shapes():
    dictionary = delay_dictionary(np.arange(10), 30)
    import pytest

    with pytest.raises(Exception):
        estimate_cir_taps(np.zeros(8, dtype=np.complex64), dictionary)  # subcarrier mismatch
    with pytest.raises(Exception):
        cir_from_csi(np.array([], dtype=np.complex64))  # empty band
