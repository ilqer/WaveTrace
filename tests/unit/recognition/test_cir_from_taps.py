"""Delay-domain statistics on a CIR that arrives as taps (UWB), with no ISTA step."""

import numpy as np

from wavetrace.recognition.cir import cir_from_taps

TAP_SPACING_S = 1e-9


def test_two_path_taps_give_dominant_tap_spread_and_active_taps():
    taps = np.zeros(64, dtype=np.complex64)
    taps[10] = 1.0          # strong first path
    taps[30] = 0.5j         # weaker reflection, power 0.25
    cir = cir_from_taps(taps, TAP_SPACING_S)
    assert cir.dominant_bin == 10
    assert cir.dominant_delay_s == 10 * TAP_SPACING_S
    assert cir.active_tap_count == 2
    # power weights 0.8 / 0.2 at 10 ns / 30 ns: mean 14 ns, spread √(0.8·16 + 0.2·256) ns = 8 ns
    np.testing.assert_allclose(cir.rms_delay_spread_s, 8e-9, rtol=1e-6)
    np.testing.assert_allclose(cir.dominant_ratio, 0.8, rtol=1e-6)
    np.testing.assert_allclose(cir.subcarrier_spacing_hz, 1.0 / (64 * TAP_SPACING_S))
