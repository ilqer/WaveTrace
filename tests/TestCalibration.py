"""Phase 3 (step 3c) — Calibration: quiet-baseline gain lock + NBVI session flow."""

import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.Calibration import Calibration, CalibrationResult, reflectionSignature


def _quietBaseline(A, S, F, informative, seed):
    """A still scene: fixed channel + small noise + per-frame AGC. One subcarrier varies for NBVI."""
    rng = np.random.default_rng(seed)
    baseMag = rng.uniform(0.8, 1.2, (A, S))
    phase = rng.uniform(-np.pi, np.pi, (A, S))
    frames = []
    for _ in range(F):
        k = rng.uniform(0.85, 1.15)                      # AGC oscillation (common scale)
        noise = rng.normal(0.0, 0.01, (A, S))
        noise[:, informative] = rng.normal(0.0, 0.3, A)  # this subcarrier varies the most
        grid = ((baseMag + noise) * k * np.exp(1j * phase)).astype(np.complex64)
        f = CsiFrame(A, S)
        f.grid[:, :] = grid
        frames.append(f)
    return frames, baseMag


def test_calibration_locks_gain_and_selects_subcarriers():
    A, S, F, informative = 2, 16, 200, 7
    frames, baseMag = _quietBaseline(A, S, F, informative, seed=3)

    cal = Calibration(baseline_packets=F, nbvi_max=6)
    for fr in frames:
        cal.observe(fr)
    assert cal.ready
    res = cal.finalize()

    assert isinstance(res, CalibrationResult)
    assert res.numBaseline == F
    assert res.reference_scale > 0
    assert informative in res.subcarriers                 # most-variable subcarrier chosen
    assert len(res.subcarriers) <= 6
    assert all(b - a > 1 for a, b in zip(res.subcarriers, res.subcarriers[1:]))  # non-consecutive
    assert all(0 <= s < S for s in res.subcarriers)


def test_calibration_gain_lock_usable_after_finalize():
    frames, _ = _quietBaseline(2, 8, 20, informative=3, seed=4)
    cal = Calibration(baseline_packets=20)
    for fr in frames:
        cal.observe(fr)
    cal.finalize()
    f = frames[0]
    before = np.angle(f.grid).copy()
    cal.gainLock.apply(f)                                # locked -> no raise
    assert np.allclose(np.angle(f.grid), before, atol=1e-5)
    assert np.abs(f.grid).mean() == pytest.approx(cal.gainLock.reference_scale, rel=1e-4)


def test_calibration_ready_flag():
    frames, _ = _quietBaseline(1, 4, 5, informative=1, seed=5)
    cal = Calibration(baseline_packets=3)
    cal.observe(frames[0])
    cal.observe(frames[1])
    assert not cal.ready
    cal.observe(frames[2])
    assert cal.ready


def test_calibration_empty_raises():
    with pytest.raises(ValueError):
        Calibration().finalize()


def test_calibration_ready_guard_rejects_short_baseline():
    # too few frames for baseline_packets -> finalize refuses
    frames, _ = _quietBaseline(2, 8, 5, informative=3, seed=8)
    cal = Calibration(baseline_packets=50)
    for fr in frames:
        cal.observe(fr)
    assert not cal.ready
    with pytest.raises(ValueError):
        cal.finalize()


def test_calibration_without_gain_lock():
    # Without gain lock, NBVI runs but reference_scale is NaN.
    A, S, F = 2, 16, 60
    frames, _ = _quietBaseline(A, S, F, informative=7, seed=9)
    cal = Calibration(baseline_packets=F, use_gain_lock=False)
    for fr in frames:
        cal.observe(fr)
    assert cal.ready
    res = cal.finalize()
    assert np.isnan(res.reference_scale)
    assert 7 in res.subcarriers                  # subcarrier selection unaffected
    with pytest.raises(ValueError):
        _ = cal.gainLock                        # disabled -> no lock to hand out


# Baseline reflection reference (material signature).

def test_reflection_signature_baseline_is_neutral():
    # Baseline vs itself -> mag_ratio ~1, phase_delta ~0.
    A, S, F = 2, 16, 100
    frames, _ = _quietBaseline(A, S, F, informative=7, seed=6)
    cal = Calibration(baseline_packets=F)
    for fr in frames:
        cal.observe(fr)
    res = cal.finalize()
    assert res.baseline_mag.shape == (S,)
    assert res.baseline_diff.shape == (S - 1,)

    # Magnitudes match baseline mean -> ratio 1.
    ref = CsiFrame(A, S)
    ref.grid[:, :] = res.baseline_mag.astype(np.complex64)
    magRatio, _ = reflectionSignature(np.asarray(ref.grid), res)
    assert np.allclose(magRatio, 1.0, atol=1e-4)


def test_reflection_signature_detects_attenuation():
    # Attenuated band -> mag_ratio < 1.
    A, S, F = 1, 16, 100
    frames, baseMag = _quietBaseline(A, S, F, informative=7, seed=7)
    cal = Calibration(baseline_packets=F)
    for fr in frames:
        cal.observe(fr)
    res = cal.finalize()

    subj = CsiFrame(A, S)
    g = np.asarray(frames[0].grid).copy()
    g[:, 4:8] *= 0.4                                          # object attenuates subcarriers 4..7
    subj.grid[:, :] = g
    magRatio, phaseDelta = reflectionSignature(np.asarray(subj.grid), res)
    assert magRatio[4:8].mean() < 0.6                        # clear attenuation dip
    assert magRatio[10:].mean() == pytest.approx(1.0, abs=0.2)  # untouched band stays ~1
    assert phaseDelta.shape == (S - 1,)
