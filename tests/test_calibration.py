import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.calibration import Calibration, CalibrationResult, reflection_signature


def _quiet_baseline(antenna_count, subcarrier_count, frame_count, informative, seed):
    """A still scene: fixed channel + small noise + per-frame AGC. One subcarrier varies for NBVI."""
    rng = np.random.default_rng(seed)
    baseline_magnitude = rng.uniform(0.8, 1.2, (antenna_count, subcarrier_count))
    phase = rng.uniform(-np.pi, np.pi, (antenna_count, subcarrier_count))
    frames = []
    for _ in range(frame_count):
        gain_scale = rng.uniform(0.85, 1.15)                      # AGC oscillation (common scale)
        noise = rng.normal(0.0, 0.01, (antenna_count, subcarrier_count))
        noise[:, informative] = rng.normal(0.0, 0.3, antenna_count)
        grid = ((baseline_magnitude + noise) * gain_scale * np.exp(1j * phase)).astype(np.complex64)
        frame = CsiFrame(antenna_count, subcarrier_count)
        frame.grid[:, :] = grid
        frames.append(frame)
    return frames, baseline_magnitude


def test_calibration_locks_gain_and_selects_subcarriers():
    antenna_count, subcarrier_count, frame_count, informative = 2, 16, 200, 7
    frames, baseline_magnitude = _quiet_baseline(
        antenna_count, subcarrier_count, frame_count, informative, seed=3)

    calibration = Calibration(baseline_packets=frame_count, nbvi_max=6)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    assert calibration.ready
    result = calibration.finalize()

    assert isinstance(result, CalibrationResult)
    assert result.num_baseline == frame_count
    assert result.reference_scale > 0
    assert informative in result.subcarriers
    assert len(result.subcarriers) <= 6
    # non-consecutive
    assert all(later - earlier > 1 for earlier, later in zip(result.subcarriers, result.subcarriers[1:]))
    assert all(0 <= subcarrier < subcarrier_count for subcarrier in result.subcarriers)


def test_calibration_gain_lock_usable_after_finalize():
    frames, _ = _quiet_baseline(2, 8, 20, informative=3, seed=4)
    calibration = Calibration(baseline_packets=20)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    calibration.finalize()
    frame = frames[0]
    before = np.angle(frame.grid).copy()
    calibration.gain_lock.apply(frame)
    assert np.allclose(np.angle(frame.grid), before, atol=1e-5)
    assert np.abs(frame.grid).mean() == pytest.approx(calibration.gain_lock.reference_scale, rel=1e-4)


def test_calibration_ready_flag():
    frames, _ = _quiet_baseline(1, 4, 5, informative=1, seed=5)
    calibration = Calibration(baseline_packets=3)
    calibration.observe(frames[0])
    calibration.observe(frames[1])
    assert not calibration.ready
    calibration.observe(frames[2])
    assert calibration.ready


def test_calibration_empty_raises():
    with pytest.raises(ValueError):
        Calibration().finalize()


def test_calibration_ready_guard_rejects_short_baseline():
    frames, _ = _quiet_baseline(2, 8, 5, informative=3, seed=8)
    calibration = Calibration(baseline_packets=50)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    assert not calibration.ready
    with pytest.raises(ValueError):
        calibration.finalize()


def test_calibration_without_gain_lock():
    antenna_count, subcarrier_count, frame_count = 2, 16, 60
    frames, _ = _quiet_baseline(antenna_count, subcarrier_count, frame_count, informative=7, seed=9)
    calibration = Calibration(baseline_packets=frame_count, use_gain_lock=False)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    assert calibration.ready
    result = calibration.finalize()
    assert np.isnan(result.reference_scale)
    assert 7 in result.subcarriers                  # gain lock does not affect subcarrier selection
    with pytest.raises(ValueError):
        _ = calibration.gain_lock


def test_reflection_signature_baseline_is_neutral():
    # Baseline against itself: mag_ratio ~1, phase_delta ~0.
    antenna_count, subcarrier_count, frame_count = 2, 16, 100
    frames, _ = _quiet_baseline(antenna_count, subcarrier_count, frame_count, informative=7, seed=6)
    calibration = Calibration(baseline_packets=frame_count)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    result = calibration.finalize()
    assert result.baseline_mag.shape == (subcarrier_count,)
    assert result.baseline_diff.shape == (subcarrier_count - 1,)

    # Magnitudes equal to the baseline mean give ratio 1.
    reference_frame = CsiFrame(antenna_count, subcarrier_count)
    reference_frame.grid[:, :] = result.baseline_mag.astype(np.complex64)
    magnitude_ratio, _ = reflection_signature(np.asarray(reference_frame.grid), result)
    assert np.allclose(magnitude_ratio, 1.0, atol=1e-4)


def test_reflection_signature_detects_attenuation():
    antenna_count, subcarrier_count, frame_count = 1, 16, 100
    frames, baseline_magnitude = _quiet_baseline(
        antenna_count, subcarrier_count, frame_count, informative=7, seed=7)
    calibration = Calibration(baseline_packets=frame_count)
    for baseline_frame in frames:
        calibration.observe(baseline_frame)
    result = calibration.finalize()

    subject_frame = CsiFrame(antenna_count, subcarrier_count)
    attenuated_grid = np.asarray(frames[0].grid).copy()
    attenuated_grid[:, 4:8] *= 0.4
    subject_frame.grid[:, :] = attenuated_grid
    magnitude_ratio, phase_delta = reflection_signature(np.asarray(subject_frame.grid), result)
    assert magnitude_ratio[4:8].mean() < 0.6
    assert magnitude_ratio[10:].mean() == pytest.approx(1.0, abs=0.2)  # untouched band stays ~1
    assert phase_delta.shape == (subcarrier_count - 1,)
