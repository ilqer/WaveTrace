"""Phase 3 (step 3b) — signal/gainlock: AGC stabilization + gain-invariant CV fallback."""

import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, GainLock, coefficient_of_variation


def _frame(grid):
    g = np.asarray(grid, dtype=np.complex64)
    f = CsiFrame(g.shape[0], g.shape[1])
    f.grid[:, :] = g
    return f


# --- Coefficient of Variation: gain-invariant ----------------------------------------------

def test_cv_is_gain_invariant():
    rng = np.random.default_rng(0)
    amp = rng.uniform(0.2, 2.0, 64).astype(np.float32)
    base = coefficient_of_variation(amp)
    # Scaling all amplitudes by any positive k leaves CV unchanged: CV(k*A) == CV(A).
    assert coefficient_of_variation((2.5 * amp).astype(np.float32)) == pytest.approx(base, rel=1e-5)
    assert coefficient_of_variation((0.1 * amp).astype(np.float32)) == pytest.approx(base, rel=1e-5)
    assert coefficient_of_variation(np.ones(32, np.float32)) == pytest.approx(0.0, abs=1e-6)


# --- GainLock: removes per-frame AGC gain, preserves phase ----------------------------------

def test_gainlock_removes_per_frame_gain():
    rng = np.random.default_rng(1)
    A, S = 3, 16
    base = (rng.uniform(0.5, 1.5, (A, S)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S)))).astype(np.complex64)

    gl = GainLock(baseline_packets=200)
    for _ in range(200):
        gl.observe(_frame(base * rng.uniform(0.7, 1.3)))  # AGC oscillation = real per-frame scale
    gl.finalize()
    assert gl.locked

    # Two frames, same underlying signal but very different AGC gains -> identical after lock.
    f1, f2 = _frame(base * 0.7), _frame(base * 1.3)
    gl.apply(f1)
    gl.apply(f2)
    assert np.allclose(np.abs(f1.grid), np.abs(f2.grid), rtol=1e-4)
    # Each frame is rescaled to the reference level.
    assert np.abs(f1.grid).mean() == pytest.approx(gl.reference_scale, rel=1e-4)


def test_gainlock_preserves_phase():
    rng = np.random.default_rng(2)
    base = (rng.uniform(0.5, 1.5, (2, 8)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 8)))).astype(np.complex64)
    gl = GainLock(10)
    for _ in range(10):
        gl.observe(_frame(base * rng.uniform(0.8, 1.2)))
    gl.finalize()

    f = _frame(base * 1.5)
    before = np.angle(f.grid).copy()
    gl.apply(f)
    assert np.allclose(np.angle(f.grid), before, atol=1e-5)  # positive real scale -> phase untouched


# --- GainLock: bookkeeping + error handling --------------------------------------------------

def test_gainlock_ready_and_observed():
    gl = GainLock(3)
    f = _frame(np.ones((1, 2)))
    assert gl.observed == 0 and not gl.ready
    gl.observe(f)
    gl.observe(f)
    assert gl.observed == 2 and not gl.ready
    gl.observe(f)
    assert gl.ready


def test_gainlock_errors():
    gl = GainLock(5)
    f = _frame(np.ones((1, 4)))
    with pytest.raises(FrameError):
        gl.finalize()  # no baseline observed
    with pytest.raises(FrameError):
        gl.apply(f)    # not finalized yet
    gl.observe(f)
    gl.finalize()
    with pytest.raises(FrameError):
        gl.observe(f)  # cannot observe after finalize
