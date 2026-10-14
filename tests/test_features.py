import numpy as np
import pytest

from wavetrace import (
    FeatureExtractor,
    InterCarrierExtractor,
    Preprocessor,
    WaveTraceError,
    block_average_decimate,
    doppler_features,
    fft,
    inter_carrier_phase_stats,
    inter_carrier_stats,
    nine_features,
    power_spectrum,
    reconstruct_complex_csi,
    reflection_null,
)


def test_reconstruct_complex_csi_removes_linear_ramp_keeps_material():
    subcarrier_count = 30
    subcarrier_indices = np.arange(subcarrier_count)
    magnitude = (2.0 + 0.1 * np.sin(subcarrier_indices)).astype(np.float64)
    ramp = 0.5 * subcarrier_indices - 1.2                  # Linear STO ramp.
    material = 0.3 * np.cos(0.7 * subcarrier_indices)      # Nonlinear material signature.
    phase = ramp + material
    channel = (magnitude * np.exp(1j * phase)).astype(np.complex64)

    out = reconstruct_complex_csi(channel)

    assert np.allclose(np.abs(out), magnitude, atol=1e-4)
    # Linear component removed. Residual is phase minus LS line.
    line_coefficients = np.polyfit(subcarrier_indices, phase, 1)
    expected_residual_phase = phase - np.polyval(line_coefficients, subcarrier_indices)
    assert np.allclose(np.angle(out), expected_residual_phase, atol=1e-3)


def test_reconstruct_complex_csi_pure_ramp_becomes_real():
    subcarrier_count = 16
    subcarrier_indices = np.arange(subcarrier_count)
    # pure linear ramp, no material
    channel = (1.5 * np.exp(1j * (0.3 * subcarrier_indices + 0.7))).astype(np.complex64)
    out = reconstruct_complex_csi(channel)
    assert np.allclose(np.angle(out), 0.0, atol=1e-4)  # nothing left once the line is removed
    assert np.allclose(np.abs(out), 1.5, atol=1e-4)


def test_reflection_null_cancels_empty_room():
    subcarrier_count = 16
    rng = np.random.default_rng(3)
    baseline_h1 = (rng.standard_normal(subcarrier_count)
                   + 1j * rng.standard_normal(subcarrier_count)).astype(np.complex64)
    baseline_h2 = (rng.standard_normal(subcarrier_count)
                   + 1j * rng.standard_normal(subcarrier_count)).astype(np.complex64)
    # Empty room (h == baseline) -> 0.
    empty = reflection_null(baseline_h1, baseline_h2, baseline_h1, baseline_h2)
    assert np.allclose(empty, 0.0, atol=1e-4)
    # Object perturbs path -> nonzero residual.
    perturbed = reflection_null(baseline_h1 + (0.5 + 0.5j), baseline_h2, baseline_h1, baseline_h2)
    assert np.abs(perturbed).mean() > 1e-2


def test_reflection_null_length_mismatch_raises():
    unit_channel = np.ones(8, dtype=np.complex64)
    with pytest.raises(WaveTraceError):
        reflection_null(unit_channel, unit_channel, unit_channel, np.ones(7, dtype=np.complex64))


def test_block_average_decimate_block_means():
    x = np.arange(100, dtype=np.float32)
    out = block_average_decimate(x, 20)
    assert out.shape == (5,)
    assert np.allclose(out, [9.5, 29.5, 49.5, 69.5, 89.5])


def test_block_average_decimate_drops_remainder():
    out = block_average_decimate(np.arange(105, dtype=np.float32), 20)
    assert out.shape == (5,)  # the trailing 5 samples are dropped
from wavetrace.synthetic import generate_stream


def test_fft_matches_numpy():
    rng = np.random.default_rng(0)
    for length in (8, 64, 128, 1024):
        x = (rng.standard_normal(length) + 1j * rng.standard_normal(length)).astype(np.complex64)
        got = fft(x)
        expected = np.fft.fft(x.astype(np.complex128))
        # float32 vs float64 numpy.
        assert np.allclose(got, expected, rtol=1e-3, atol=1e-2)


def test_fft_non_power_of_two_raises():
    with pytest.raises(WaveTraceError):
        fft(np.ones(10, dtype=np.complex64))


def _reference_nine_features(x):
    x = x.astype(np.float64)
    deviation = x - x.mean()
    std = x.std()  # population (ddof=0), matches the C++ two-pass var/n
    skew = (deviation ** 3).mean() / std ** 3 if std > 0 else 0.0
    lag1_autocorrelation = (
        float(np.sum(deviation[1:] * deviation[:-1]) / np.sum(deviation * deviation))
        if np.sum(deviation * deviation) > 0 else 0.0
    )
    median_absolute_deviation = np.median(np.abs(x - np.median(x)))
    return np.array([
        x.mean(), std, x.max(), x.min(),
        np.percentile(x, 75) - np.percentile(x, 25),  # IQR (numpy default linear interp)
        skew, lag1_autocorrelation, median_absolute_deviation,
        np.sum(np.abs(np.diff(x))),   # waveform-length
    ])


def test_nine_features_match_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(128).astype(np.float32)
    got = np.asarray(nine_features(x), dtype=np.float64)
    assert np.allclose(got, _reference_nine_features(x), rtol=1e-3, atol=1e-4)


def test_nine_features_constant_window():
    out = nine_features(np.full(32, 3.0, dtype=np.float32))
    (mean, std, maximum, minimum, interquartile_range, skew, lag1_autocorrelation,
     median_absolute_deviation, waveform_length) = out
    assert mean == pytest.approx(3.0) and maximum == pytest.approx(3.0) and minimum == pytest.approx(3.0)
    assert std == pytest.approx(0.0) and interquartile_range == pytest.approx(0.0)
    assert (median_absolute_deviation == pytest.approx(0.0) and waveform_length == pytest.approx(0.0)
            and skew == pytest.approx(0.0))


def test_inter_carrier_stats_matches_numpy():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(52).astype(np.float32)  # 52 = ESP32 HT20 data subcarriers
    mean, variance = inter_carrier_stats(x)
    assert mean == pytest.approx(float(np.mean(x.astype(np.float64))), rel=1e-5)
    assert variance == pytest.approx(float(np.var(x.astype(np.float64), ddof=1)), rel=1e-4)  # sample (M-1)


def test_inter_carrier_stats_metal_lower_variance():
    # Flat metal -> low variance. Diffuse body -> high variance.
    rng = np.random.default_rng(8)
    flat = (np.full(52, 5.0) + rng.standard_normal(52) * 0.05).astype(np.float32)
    diffuse = (5.0 + rng.standard_normal(52) * 2.0).astype(np.float32)
    _, flat_variance = inter_carrier_stats(flat)
    _, diffuse_variance = inter_carrier_stats(diffuse)
    assert flat_variance < diffuse_variance


def test_inter_carrier_stats_edge_cases():
    mean, variance = inter_carrier_stats(np.full(30, 2.5, dtype=np.float32))
    assert mean == pytest.approx(2.5) and variance == pytest.approx(0.0)
    # single -> no (M-1) blowup
    single_mean, single_variance = inter_carrier_stats(np.array([4.0], dtype=np.float32))
    assert single_mean == pytest.approx(4.0) and single_variance == pytest.approx(0.0)


def test_inter_carrier_phase_stats_recovers_slope():
    # Linear phase ramp (pure group delay) -> fit recovers slope, residual ~0.
    subcarrier_count = 52
    true_slope = 0.2  # rad/subcarrier
    phase = (true_slope * np.arange(subcarrier_count) + 1.3).astype(np.float32)
    # Wrap to [-pi, pi] to exercise unwrap path.
    wrapped = np.angle(np.exp(1j * phase)).astype(np.float32)
    slope, residual = inter_carrier_phase_stats(wrapped)
    assert slope == pytest.approx(true_slope, abs=1e-3)
    assert residual == pytest.approx(0.0, abs=1e-3)


def test_inter_carrier_phase_stats_coherent_vs_diffuse():
    # Coherent reflector -> low residual. Diffuse body -> high residual.
    rng = np.random.default_rng(11)
    subcarrier_count = 52
    ramp = 0.15 * np.arange(subcarrier_count)
    coherent = np.angle(np.exp(1j * (ramp + rng.standard_normal(subcarrier_count) * 0.02))).astype(np.float32)
    diffuse = np.angle(np.exp(1j * (ramp + rng.standard_normal(subcarrier_count) * 1.5))).astype(np.float32)
    _, coherent_residual = inter_carrier_phase_stats(coherent)
    _, diffuse_residual = inter_carrier_phase_stats(diffuse)
    assert coherent_residual < diffuse_residual


def test_inter_carrier_phase_stats_edge_cases():
    # fewer than 2 subcarriers -> zeros
    slope, residual = inter_carrier_phase_stats(np.array([0.5], dtype=np.float32))
    assert slope == pytest.approx(0.0) and residual == pytest.approx(0.0)


def test_inter_carrier_extractor_cadence_and_shape():
    W, H = 8, 2
    extractor = InterCarrierExtractor(W, H)
    assert extractor.output_size == 27  # 3 series (mu|sigma2|cv) x 9 features
    rng = np.random.default_rng(12)
    emits = [i for i in range(20)
             if extractor.push((5.0 + rng.standard_normal(52) * 0.5).astype(np.float32))]
    assert emits[0] == W - 1
    assert all((emit - emits[0]) % H == 0 for emit in emits)
    assert extractor.features.shape == (27,)


def test_inter_carrier_extractor_matches_nine_features():
    rng = np.random.default_rng(13)
    W, H, K = 16, 4, 52
    frames = (5.0 + rng.standard_normal((W, K)) * 0.5).astype(np.float32)
    mean_series, variance_series, coefficient_of_variation_series = [], [], []
    extractor = InterCarrierExtractor(W, H)
    emitted = False
    for row in frames:
        emitted = extractor.push(np.ascontiguousarray(row))
        row_mean, row_variance = inter_carrier_stats(np.ascontiguousarray(row))
        mean_series.append(row_mean)
        variance_series.append(row_variance)
        coefficient_of_variation_series.append(np.sqrt(row_variance) / row_mean if row_mean > 1e-12 else 0.0)
    assert emitted
    got = extractor.features.reshape(3, 9)
    for series_index, series in enumerate((mean_series, variance_series, coefficient_of_variation_series)):
        reference = nine_features(np.asarray(series, dtype=np.float32))
        assert np.allclose(got[series_index], reference, rtol=1e-4, atol=1e-5)


def _phase_series(seed=9):
    sample_rate_hz, true_frequency_hz = 100.0, 0.3
    frames, ground_truth = generate_stream(
        num_antennas=1, num_subcarriers=64, sample_rate_hz=sample_rate_hz, num_frames=1024,
        perturbation_hz=true_frequency_hz, perturbation_depth=1.0, cfo_hz=4.0, noise_std=0.002, seed=seed,
    )
    preprocessor = Preprocessor(1, 64, normalize_alpha=0.01)  # gentle high-pass so 0.3 Hz passes
    out = np.stack([preprocessor.process(frame).copy()[0] for frame in frames])
    return out.mean(axis=1).astype(np.float32), sample_rate_hz, ground_truth["perturbation_hz"]


def test_power_spectrum_recovers_motion_frequency():
    series, sample_rate_hz, true_frequency_hz = _phase_series()
    fft_size = 4096  # zero-pad for fine bin spacing (sample_rate_hz/fft_size ≈ 0.024 Hz)
    power = power_spectrum(series, nfft=fft_size)
    frequencies_hz = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate_hz)
    assert power.shape == frequencies_hz.shape
    band = (frequencies_hz >= 0.1) & (frequencies_hz <= 2.0)
    recovered = frequencies_hz[band][np.argmax(power[band])]
    assert recovered == pytest.approx(true_frequency_hz, abs=0.05)


def test_doppler_recovers_motion_frequency():
    series, sample_rate_hz, true_frequency_hz = _phase_series()
    max_shift_hz, spread = doppler_features(series, fs=sample_rate_hz, f_hi=2.0, nfft=4096)
    assert max_shift_hz == pytest.approx(true_frequency_hz, abs=0.05)  # peak Doppler = injected motion freq
    assert spread >= 0.0


def test_feature_extractor_cadence_and_shape():
    C, W, H = 3, 8, 2
    extractor = FeatureExtractor(C, W, H)
    assert extractor.output_size == 9 * C
    emits = [i for i in range(20) if extractor.push(np.full(C, float(i), dtype=np.float32))]
    # Emit when window fills, then every hop.
    assert emits[0] == W - 1
    assert all((emit - emits[0]) % H == 0 for emit in emits)
    assert extractor.features.shape == (9 * C,)


def test_feature_extractor_matches_nine_features():
    # The window must be read back in chronological order.
    rng = np.random.default_rng(3)
    C, W, H = 2, 16, 4
    data = rng.standard_normal((W, C)).astype(np.float32)  # exactly one full window
    extractor = FeatureExtractor(C, W, H)
    emitted = False
    for row in data:
        emitted = extractor.push(np.ascontiguousarray(row))
    assert emitted  # frame W-1 fills the window and W-1 is a multiple of H
    got = extractor.features.reshape(C, 9)
    for channel in range(C):
        assert np.allclose(got[channel], nine_features(np.ascontiguousarray(data[:, channel])),
                           rtol=1e-4, atol=1e-5)


def test_feature_extractor_push_wrong_length_raises():
    extractor = FeatureExtractor(4, 8, 2)
    with pytest.raises(WaveTraceError):
        extractor.push(np.ones(3, dtype=np.float32))


def test_amplitude_features_detect_modulation():
    sample_rate_hz = 100.0
    stream_options = dict(num_antennas=1, num_subcarriers=16, sample_rate_hz=sample_rate_hz,
                          num_frames=256,
                          perturbation_hz=0.0, perturbation_depth=0.0, cfo_hz=0.0, noise_std=0.001,
                          seed=4)
    modulated_frames, _ = generate_stream(**stream_options, amplitude_hz=0.25, amplitude_depth=0.5)
    flat_frames, _ = generate_stream(**stream_options, amplitude_hz=0.25, amplitude_depth=0.0)

    subcarrier_index = 15  # subcarrier with the largest amplitude sensitivity (amplitude_scale = 1.5)
    modulated_amplitude = np.abs(
        np.stack([frame.grid[0, subcarrier_index] for frame in modulated_frames])).astype(np.float32)
    flat_amplitude = np.abs(
        np.stack([frame.grid[0, subcarrier_index] for frame in flat_frames])).astype(np.float32)

    modulated = nine_features(modulated_amplitude)
    flat = nine_features(flat_amplitude)
    # Slow envelope lifts spread features (std/IQR/MAD).
    for feature_index in (1, 4, 7):
        assert modulated[feature_index] > 20 * flat[feature_index]
    assert modulated[8] > flat[8]  # waveform-length still increases, just modestly
