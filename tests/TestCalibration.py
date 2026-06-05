"""Phase 3 (step 3c) — Calibration: quiet-baseline gain lock + NBVI session flow."""

import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.Calibration import Calibration, CalibrationResult


def _quietBaseline(A, S, F, informative, seed):
    """A still scene: fixed channel + small noise + per-frame AGC gain, with one subcarrier given
    extra variation so NBVI has a clear winner."""
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
    assert res.num_baseline == F
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
    cal.gain_lock.apply(f)                                # locked -> no raise
    assert np.allclose(np.angle(f.grid), before, atol=1e-5)
    assert np.abs(f.grid).mean() == pytest.approx(cal.gain_lock.reference_scale, rel=1e-4)


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
