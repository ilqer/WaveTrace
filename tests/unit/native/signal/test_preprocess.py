"""Preprocess chain on synthetic data with known ground truth: CFO/SFO cancellation,
spike rejection and phase continuity.
"""

import numpy as np
import pytest

from wavetrace import (
    CsiFrame,
    FrameError,
    Preprocessor,
    combined_channel_difference,
    conjugate_multiply,
    hampel,
    unwrap_step,
)
from wavetrace.synthetic import generate_stream


def test_combined_channel_difference_subtracts_antenna_zero():
    rng = np.random.default_rng(7)
    A, S = 3, 6
    H = (rng.uniform(0.5, 1.5, (A, S)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S)))).astype(np.complex64)
    input_frame = CsiFrame(A, S)
    input_frame.grid[:, :] = H
    out = CsiFrame(1, 1)
    combined_channel_difference(input_frame, out)
    assert out.grid.shape == (A - 1, S)  # antenna-difference -> (A-1) x S
    expected = (H[1:, :] - H[0, :]).astype(np.complex64)
    assert np.allclose(out.grid, expected, atol=1e-5)


def test_combined_channel_difference_requires_two_antennas():
    with pytest.raises(FrameError):
        combined_channel_difference(CsiFrame(1, 8), CsiFrame(1, 1))


def test_conjugate_multiply_cross_antenna_cancels_common_offset():
    rng = np.random.default_rng(0)
    A, S = 3, 5
    true_channel = (rng.uniform(0.5, 1.5, (A, S))
                    * np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S)))).astype(np.complex64)
    theta = 1.234  # one common-mode hardware/CFO phase added to every antenna+subcarrier
    measured = (true_channel * np.exp(1j * theta)).astype(np.complex64)

    input_frame = CsiFrame(A, S)
    input_frame.grid[:, :] = measured
    out = CsiFrame(1, 1)
    conjugate_multiply(input_frame, out)

    assert out.grid.shape == (A - 1, S)  # cross-antenna -> (A-1) x S
    expected = (true_channel[1:, :] * np.conj(true_channel[0, :])).astype(np.complex64)  # theta cancels
    assert np.allclose(out.grid, expected, atol=1e-4)


def test_conjugate_multiply_cross_subcarrier_cancels_common_offset():
    rng = np.random.default_rng(1)
    S = 8
    true_channel = (rng.uniform(0.5, 1.5, S)
                    * np.exp(1j * rng.uniform(-np.pi, np.pi, S))).astype(np.complex64)
    measured = (true_channel * np.exp(1j * 0.77)).astype(np.complex64)  # common across subcarriers

    input_frame = CsiFrame(1, S)
    input_frame.grid[:, :] = measured
    out = CsiFrame(1, 1)
    conjugate_multiply(input_frame, out)

    assert out.grid.shape == (1, S - 1)  # cross-subcarrier fallback
    expected = (true_channel[1:] * np.conj(true_channel[:-1])).astype(np.complex64)
    assert np.allclose(out.grid[0], expected, atol=1e-4)


def test_conjugate_multiply_single_subcarrier_raises():
    input_frame = CsiFrame(1, 1)  # 1 antenna, 1 subcarrier -> nothing to pair
    with pytest.raises(FrameError):
        conjugate_multiply(input_frame, CsiFrame(1, 1))


def test_hampel_passes_normal_replaces_spike():
    window = np.array([1.0, 1.1, 0.9, 1.05, 50.0], dtype=np.float32)
    assert hampel(window, current=50.0, k=5.0) == pytest.approx(1.05)  # outlier -> median
    normal = np.array([1.0, 1.1, 0.9, 1.05, 1.0], dtype=np.float32)
    assert hampel(normal, current=1.0, k=5.0) == pytest.approx(1.0)    # inlier -> unchanged


def test_unwrap_step_continuity_and_wrap():
    # Small step: accumulates.
    assert unwrap_step(0.2, 0.1, 10.0) == pytest.approx(10.1)
    # Apparent jump 3.0 -> -3.0 is physically +0.2832 rad (crossed +pi).
    assert unwrap_step(-3.0, 3.0, 0.0) == pytest.approx(-6.0 + 2 * np.pi, abs=1e-5)


def test_preprocessor_output_geometry():
    assert (Preprocessor(3, 64).out_rows, Preprocessor(3, 64).out_cols) == (2, 64)  # cross-antenna
    assert (Preprocessor(1, 64).out_rows, Preprocessor(1, 64).out_cols) == (1, 63)  # cross-subcarrier


def _reference_chain(frames, alpha):
    """Numpy reference for A=1 cross-subcarrier chain (Hampel disabled).
    conj-mult -> temporal unwrap -> EMA-detrend normalize."""
    H = np.stack([frame.grid[0] for frame in frames])
    D = H[:, 1:] * np.conj(H[:, :-1])
    unwrapped = np.unwrap(np.angle(D), axis=0)
    trend = np.empty_like(unwrapped)
    trend[0] = unwrapped[0]
    for i in range(1, len(unwrapped)):
        trend[i] = alpha * unwrapped[i] + (1 - alpha) * trend[i - 1]
    return (unwrapped - trend).astype(np.float32)


def test_preprocessor_chain_matches_numpy_reference():
    # Wrapping motion tests unwrap. Hampel disabled (k huge) to isolate other steps.
    rng = np.random.default_rng(5)
    T, S = 200, 6
    psi = rng.uniform(-np.pi, np.pi, S)
    frame_indices = np.arange(T)
    frames = []
    for frame_index in frame_indices:
        # phase per subcarrier drifts enough to wrap across frames
        phase = psi + 0.15 * frame_index * np.linspace(0.5, 1.5, S)
        frame = CsiFrame(1, S)
        frame.grid[:, :] = np.exp(1j * phase).astype(np.complex64)[None, :]
        frames.append(frame)

    alpha = 0.1
    preprocessor = Preprocessor(1, S, hampel_k=1e9, normalize_alpha=alpha)
    got = np.stack([preprocessor.process(input_frame).copy()[0] for input_frame in frames])
    ref = _reference_chain(frames, alpha)
    assert np.allclose(got, ref, atol=1e-3)


def test_preprocessor_rejects_magnitude_spike():
    # Hampel(magnitude) must hold phase steady during a magnitude spike.
    rng = np.random.default_rng(7)
    A, S, T, spike_index = 2, 4, 30, 15
    base_grid = np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S))).astype(np.complex64)
    frames = []
    for frame_index in range(T):
        grid = (base_grid * rng.uniform(0.9, 1.1)).astype(np.complex64)  # Jitter amplitude, keep phase
        if frame_index == spike_index:
            grid[1, 0] = 50.0 * np.exp(1j * (np.angle(base_grid[1, 0]) + 2.0))  # |.|=50 spike + 2 rad twist
        frame = CsiFrame(A, S)
        frame.grid[:, :] = grid
        frames.append(frame)

    reject = Preprocessor(A, S, hampel_window=7, hampel_k=5.0, normalize_alpha=0.1)
    rejecting_output = np.stack([reject.process(input_frame).copy() for input_frame in frames])

    control = Preprocessor(A, S, hampel_window=7, hampel_k=1e9, normalize_alpha=0.1)  # no rejection
    control_output = np.stack([control.process(input_frame).copy() for input_frame in frames])

    cell = rejecting_output[spike_index, 0, 0]          # affected cell (antenna1 vs antenna0, subcarrier0)
    assert abs(cell) < 0.05                                  # spike held -> output stays ~0
    assert abs(control_output[spike_index, 0, 0]) > 10 * abs(cell) + 0.1  # without rejection it glitches


def test_preprocessor_recovers_motion_frequency():
    sample_rate_hz, true_frequency_hz = 100.0, 0.3
    frames, ground_truth = generate_stream(
        num_antennas=1, num_subcarriers=64, sample_rate_hz=sample_rate_hz, num_frames=1024,
        perturbation_hz=true_frequency_hz, perturbation_depth=1.0, cfo_hz=4.0, noise_std=0.002, seed=9,
    )
    preprocessor = Preprocessor(1, 64, normalize_alpha=0.01)  # gentle high-pass so 0.3 Hz passes
    out = np.stack([preprocessor.process(frame).copy()[0] for frame in frames])

    # Average power spectrum across cells to lift SNR. Band-argmax to find true_frequency_hz.
    frame_count = out.shape[0]
    x = (out - out.mean(0)) * np.hanning(frame_count)[:, None]
    fft_size = 1 << (max(4 * frame_count, 64) - 1).bit_length()
    power = (np.abs(np.fft.rfft(x, fft_size, axis=0)) ** 2).mean(1)
    frequencies_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate_hz)
    band = (frequencies_hz >= 0.1) & (frequencies_hz <= 2.0)
    recovered = frequencies_hz[band][np.argmax(power[band])]
    assert recovered == pytest.approx(ground_truth["perturbation_hz"], abs=0.05)
