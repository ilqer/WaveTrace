"""Offline delay-domain super-resolution (CIR via ISTA) — RuView ADR-134 §2.9 synthetic test plan.

Deterministic, no hardware: a known 2-tap channel must be recovered to within one delay bin, proving
the L1 solve super-resolves below the 1/BW Nyquist limit and isolates a strong reflector from a
weaker one (the body-vs-concealed-object case).
"""

import numpy as np

from wavetrace.recognition.Cir import (
    SUBCARRIER_SPACING_HZ,
    cir_features,
    cir_from_csi,
    delay_dictionary,
    estimate_cir_taps,
)

DF = SUBCARRIER_SPACING_HZ


def _two_tap_csi(k, taus, amps):
    """H[k] = Σ aᵢ·exp(-j2π·k·Δf·τᵢ) over contiguous subcarriers 0..k-1."""
    idx = np.arange(k)[:, None]
    atoms = np.exp(-2j * np.pi * idx * DF * np.asarray(taus)[None, :])  # (k, n_paths)
    return (atoms @ np.asarray(amps, dtype=np.complex128)).astype(np.complex64)


def _peak_delays(cir):
    """Delays (s) of local-maxima taps above a -25 dB floor, matching the module's detector."""
    power = np.abs(cir.taps) ** 2
    floor = power.max() * 10 ** (-2.5)
    is_peak = (power > floor)
    is_peak[1:-1] &= (power[1:-1] >= power[:-2]) & (power[1:-1] >= power[2:])
    is_peak[0] = is_peak[-1] = False
    return cir.tap_delays_s[np.flatnonzero(is_peak)]


def test_two_tap_recovery_ht20():
    # ADR-134 §2.9 Tier 1: direct path 30 ns @0.8∠45°, reflection 80 ns @0.3∠135°, K=56.
    tau1, tau2 = 30e-9, 80e-9
    a1 = 0.8 * np.exp(1j * np.pi / 4)
    a2 = 0.3 * np.exp(1j * 3 * np.pi / 4)
    H = _two_tap_csi(56, [tau1, tau2], [a1, a2])

    cir = cir_from_csi(H, oversample=3, lam=0.02, n_iter=200)
    bin_w = cir.tap_delays_s[1]  # one fine-grid delay bin in seconds

    # dominant tap is the direct path within one bin
    assert abs(cir.dominant_delay_s - tau1) <= bin_w
    # strong path dominates (power ratio 0.64 / 0.73 ≈ 0.88)
    assert cir.dominant_ratio > 0.7
    # exactly two taps survive the -25 dB floor
    assert cir.active_tap_count == 2

    # both physical paths appear as taps within one bin of their true delays
    peak_delays = _peak_delays(cir)
    assert np.min(np.abs(peak_delays - tau1)) <= bin_w
    assert np.min(np.abs(peak_delays - tau2)) <= bin_w


def test_super_resolution_beats_nyquist():
    # two taps closer than the 1/BW native resolution must still separate (3× super-res).
    k = 56
    native_res = 1.0 / (k * DF)  # ~57 ns
    tau1, tau2 = 20e-9, 20e-9 + 0.6 * native_res  # sub-Nyquist separation
    H = _two_tap_csi(k, [tau1, tau2], [1.0, 0.7])
    cir = cir_from_csi(H, oversample=4, lam=0.01, n_iter=300)
    assert cir.active_tap_count >= 2


def test_dictionary_is_well_conditioned():
    # κ(Φ) ≈ 1: ΦΦᴴ ≈ (G/K)·I for a normalised sub-DFT (ADR-134 §2.3).
    k, g = 52, 156
    phi = delay_dictionary(np.arange(k), g)
    gram = phi @ phi.conj().T
    diag = np.diag(gram).real
    assert np.allclose(diag, g / k, rtol=0.1)
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) < 0.5 * (g / k)  # off-diagonals small vs diagonal


def test_features_shape_and_gapped_band():
    # gapped layout (HT40-style central null) must be accepted via freq_idx.
    k = 48
    freq_idx = np.concatenate([np.arange(0, 24), np.arange(28, 52)])  # 4-tone central gap
    H = _two_tap_csi(52, [25e-9, 70e-9], [1.0, 0.4])[freq_idx]
    cir = cir_from_csi(H, freq_idx=freq_idx, oversample=3, lam=0.02, n_iter=150)
    feats = cir_features(cir)
    assert feats.shape == (5,)
    assert np.all(np.isfinite(feats))


def test_rejects_bad_shapes():
    phi = delay_dictionary(np.arange(10), 30)
    import pytest

    with pytest.raises(Exception):
        estimate_cir_taps(np.zeros(8, dtype=np.complex64), phi)  # subcarrier mismatch
    with pytest.raises(Exception):
        cir_from_csi(np.array([], dtype=np.complex64))  # empty band
