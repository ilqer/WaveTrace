import numpy as np
import pytest

from wavetrace import CsiFrame, FrameError, GainLock, coefficient_of_variation


def _frame(grid):
    complex_grid = np.asarray(grid, dtype=np.complex64)
    frame = CsiFrame(complex_grid.shape[0], complex_grid.shape[1])
    frame.grid[:, :] = complex_grid
    return frame


def test_cv_is_gain_invariant():
    rng = np.random.default_rng(0)
    amplitudes = rng.uniform(0.2, 2.0, 64).astype(np.float32)
    base = coefficient_of_variation(amplitudes)
    # Scaling amplitudes leaves CV unchanged.
    assert coefficient_of_variation((2.5 * amplitudes).astype(np.float32)) == pytest.approx(base, rel=1e-5)
    assert coefficient_of_variation((0.1 * amplitudes).astype(np.float32)) == pytest.approx(base, rel=1e-5)
    assert coefficient_of_variation(np.ones(32, np.float32)) == pytest.approx(0.0, abs=1e-6)


def test_gainlock_removes_per_frame_gain():
    rng = np.random.default_rng(1)
    antenna_count, subcarrier_count = 3, 16
    base = (rng.uniform(0.5, 1.5, (antenna_count, subcarrier_count))
            * np.exp(1j * rng.uniform(-np.pi, np.pi, (antenna_count, subcarrier_count)))
            ).astype(np.complex64)

    gain_lock = GainLock(baseline_packets=200)
    for _ in range(200):
        gain_lock.observe(_frame(base * rng.uniform(0.7, 1.3)))  # AGC oscillation.
    gain_lock.finalize()
    assert gain_lock.locked

    # Same signal with different AGC gains -> identical after lock.
    low_gain_frame, high_gain_frame = _frame(base * 0.7), _frame(base * 1.3)
    gain_lock.apply(low_gain_frame)
    gain_lock.apply(high_gain_frame)
    assert np.allclose(np.abs(low_gain_frame.grid), np.abs(high_gain_frame.grid), rtol=1e-4)
    assert np.abs(low_gain_frame.grid).mean() == pytest.approx(gain_lock.reference_scale, rel=1e-4)


def test_gainlock_preserves_phase():
    rng = np.random.default_rng(2)
    base = (rng.uniform(0.5, 1.5, (2, 8))
            * np.exp(1j * rng.uniform(-np.pi, np.pi, (2, 8)))).astype(np.complex64)
    gain_lock = GainLock(10)
    for _ in range(10):
        gain_lock.observe(_frame(base * rng.uniform(0.8, 1.2)))
    gain_lock.finalize()

    frame = _frame(base * 1.5)
    before = np.angle(frame.grid).copy()
    gain_lock.apply(frame)
    assert np.allclose(np.angle(frame.grid), before, atol=1e-5)  # Positive real scale leaves phase untouched.


def test_gainlock_ready_and_observed():
    gain_lock = GainLock(3)
    frame = _frame(np.ones((1, 2)))
    assert gain_lock.observed == 0 and not gain_lock.ready
    gain_lock.observe(frame)
    gain_lock.observe(frame)
    assert gain_lock.observed == 2 and not gain_lock.ready
    gain_lock.observe(frame)
    assert gain_lock.ready


def test_gainlock_errors():
    gain_lock = GainLock(5)
    frame = _frame(np.ones((1, 4)))
    with pytest.raises(FrameError):
        gain_lock.finalize()  # No baseline observed.
    with pytest.raises(FrameError):
        gain_lock.apply(frame)    # Not finalized yet.
    gain_lock.observe(frame)
    gain_lock.finalize()
    with pytest.raises(FrameError):
        gain_lock.observe(frame)  # Cannot observe after finalize.
