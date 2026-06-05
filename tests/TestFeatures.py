"""Phase 4 (step 4a/4b) — util/Fft + signal/Features.

Unit tests per public function: the radix-2 FFT vs numpy; the §2.9 nine features vs independent
numpy definitions; PSD + Doppler recovering the injected motion frequency (the DoD motion-recovery
check, routed through Features); and the streaming FeatureExtractor's cadence, shape, and content.
"""

import numpy as np
import pytest

from wavetrace import (
    CsiFrame,
    FeatureExtractor,
    Preprocessor,
    WaveTraceError,
    doppler_features,
    fft,
    nine_features,
    power_spectrum,
)
from fixtures.SyntheticCsi import generateStream


# --- FFT: matches numpy ----------------------------------------------------------------------

def test_fft_matches_numpy():
    rng = np.random.default_rng(0)
    for n in (8, 64, 128, 1024):
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
        got = fft(x)
        expected = np.fft.fft(x.astype(np.complex128))
        # float32 radix-2 vs float64 numpy: tolerance scaled to the spectrum magnitude.
        assert np.allclose(got, expected, rtol=1e-3, atol=1e-2)


def test_fft_non_power_of_two_raises():
    with pytest.raises(WaveTraceError):
        fft(np.ones(10, dtype=np.complex64))


# --- §2.9 nine features: match independent numpy definitions ---------------------------------

def _refNineFeatures(x):
    x = x.astype(np.float64)
    d = x - x.mean()
    std = x.std()  # population (ddof=0), matches the C++ two-pass var/n
    skew = (d ** 3).mean() / std ** 3 if std > 0 else 0.0
    lag1 = float(np.sum(d[1:] * d[:-1]) / np.sum(d * d)) if np.sum(d * d) > 0 else 0.0
    mad = np.median(np.abs(x - np.median(x)))
    return np.array([
        x.mean(), std, x.max(), x.min(),
        np.percentile(x, 75) - np.percentile(x, 25),  # IQR (numpy default linear interp)
        skew, lag1, mad, np.sum(np.abs(np.diff(x))),   # MAD, waveform-length
    ])


def test_nine_features_match_numpy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(128).astype(np.float32)
    got = np.asarray(nine_features(x), dtype=np.float64)
    assert np.allclose(got, _refNineFeatures(x), rtol=1e-3, atol=1e-4)


def test_nine_features_constant_window():
    # Constant series: spread features vanish; mean/max/min equal the constant.
    out = nine_features(np.full(32, 3.0, dtype=np.float32))
    mean, std, mx, mn, iqr, skew, lag1, mad, wl = out
    assert mean == pytest.approx(3.0) and mx == pytest.approx(3.0) and mn == pytest.approx(3.0)
    assert std == pytest.approx(0.0) and iqr == pytest.approx(0.0)
    assert mad == pytest.approx(0.0) and wl == pytest.approx(0.0) and skew == pytest.approx(0.0)


# --- PSD + Doppler recover the injected motion frequency (phase path) -------------------------

def _phaseSeries(seed=9):
    """Differential-phase scalar series (mean across cells) from the synthetic motion stream."""
    fs, fTrue = 100.0, 0.3
    frames, gt = generateStream(
        numAntennas=1, numSubcarriers=64, sampleRateHz=fs, numFrames=1024,
        perturbationHz=fTrue, perturbationDepth=1.0, cfoHz=4.0, noiseStd=0.002, seed=seed,
    )
    pre = Preprocessor(1, 64, normalize_alpha=0.01)  # gentle high-pass so 0.3 Hz passes
    out = np.stack([pre.process(f).copy()[0] for f in frames])  # (T, S-1)
    return out.mean(axis=1).astype(np.float32), fs, gt["perturbation_hz"]


def test_power_spectrum_recovers_motion_frequency():
    series, fs, fTrue = _phaseSeries()
    nfft = 4096  # zero-pad for fine bin spacing (fs/nfft ≈ 0.024 Hz)
    power = power_spectrum(series, nfft=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    assert power.shape == freqs.shape
    band = (freqs >= 0.1) & (freqs <= 2.0)
    recovered = freqs[band][np.argmax(power[band])]
    assert recovered == pytest.approx(fTrue, abs=0.05)


def test_doppler_recovers_motion_frequency():
    series, fs, fTrue = _phaseSeries()
    max_shift, spread = doppler_features(series, fs=fs, f_hi=2.0, nfft=4096)
    assert max_shift == pytest.approx(fTrue, abs=0.05)  # peak Doppler = injected motion freq
    assert spread >= 0.0


# --- Streaming FeatureExtractor: cadence, shape, content -------------------------------------

def test_feature_extractor_cadence_and_shape():
    C, W, H = 3, 8, 2
    fe = FeatureExtractor(C, W, H)
    assert fe.output_size == 9 * C
    emits = [i for i in range(20) if fe.push(np.full(C, float(i), dtype=np.float32))]
    # First emit when the window fills (frame W-1), then every hop.
    assert emits[0] == W - 1
    assert all((e - emits[0]) % H == 0 for e in emits)
    assert fe.features.shape == (9 * C,)


def test_feature_extractor_matches_nine_features():
    # The streaming extractor must equal nine_features over the chronological window per series.
    rng = np.random.default_rng(3)
    C, W, H = 2, 16, 4
    data = rng.standard_normal((W, C)).astype(np.float32)  # exactly one full window
    fe = FeatureExtractor(C, W, H)
    emitted = False
    for row in data:
        emitted = fe.push(np.ascontiguousarray(row))
    assert emitted  # frame W-1 fills the window and W-1 is a multiple of H
    got = fe.features.reshape(C, 9)
    for c in range(C):
        assert np.allclose(got[c], nine_features(np.ascontiguousarray(data[:, c])), rtol=1e-4, atol=1e-5)


def test_feature_extractor_push_wrong_length_raises():
    fe = FeatureExtractor(4, 8, 2)
    with pytest.raises(WaveTraceError):
        fe.push(np.ones(3, dtype=np.float32))


# --- Amplitude features are meaningful on the extended fixture (Q7) --------------------------

def test_amplitude_features_detect_modulation():
    # Breathing-like amplitude envelope (Q7 fixture knob) must show up in the §2.9 features.
    fs = 100.0
    kw = dict(numAntennas=1, numSubcarriers=16, sampleRateHz=fs, numFrames=256,
              perturbationHz=0.0, perturbationDepth=0.0, cfoHz=0.0, noiseStd=0.001, seed=4)
    modFrames, _ = generateStream(**kw, amplitudeHz=0.25, amplitudeDepth=0.5)
    flatFrames, _ = generateStream(**kw, amplitudeHz=0.25, amplitudeDepth=0.0)

    k = 15  # subcarrier with the largest amplitude sensitivity (ampScale = 1.5)
    modAmp = np.abs(np.stack([f.grid[0, k] for f in modFrames])).astype(np.float32)
    flatAmp = np.abs(np.stack([f.grid[0, k] for f in flatFrames])).astype(np.float32)

    mod = nine_features(modAmp)
    flat = nine_features(flatAmp)
    # A slow (0.25 Hz) envelope lifts the SPREAD features (std=1, IQR=4, MAD=7) far above the noise
    # floor; it barely touches waveform-length (sensitive to fast step-to-step change, ~noise-bound).
    for idx in (1, 4, 7):
        assert mod[idx] > 20 * flat[idx]
    assert mod[8] > flat[8]  # waveform-length still increases, just modestly
