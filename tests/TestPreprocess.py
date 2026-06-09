"""Phase 3 (step 3a) — signal/preprocess: conj-mult -> Hampel -> unwrap -> normalize.

Per-transform unit tests + streaming Preprocessor integration tests. Validates the physics
(CFO/SFO cancellation, spike rejection, phase continuity) on synthetic data with known truth.
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
from fixtures.SyntheticCsi import generateStream


# --- combined_channel_difference: nulls the common environment (in-baggage material feature) ---

def test_combined_channel_difference_subtracts_antenna_zero():
    rng = np.random.default_rng(7)
    A, S = 3, 6
    H = (rng.uniform(0.5, 1.5, (A, S)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S)))).astype(np.complex64)
    in_frame = CsiFrame(A, S)
    in_frame.grid[:, :] = H
    out = CsiFrame(1, 1)
    combined_channel_difference(in_frame, out)
    assert out.grid.shape == (A - 1, S)  # antenna-difference -> (A-1) x S
    expected = (H[1:, :] - H[0, :]).astype(np.complex64)
    assert np.allclose(out.grid, expected, atol=1e-5)


def test_combined_channel_difference_requires_two_antennas():
    with pytest.raises(FrameError):
        combined_channel_difference(CsiFrame(1, 8), CsiFrame(1, 1))


# --- conjugate_multiply: cancels common-mode clock drift ------------------------------------

def test_conjugate_multiply_cross_antenna_cancels_common_offset():
    rng = np.random.default_rng(0)
    A, S = 3, 5
    trueH = (rng.uniform(0.5, 1.5, (A, S)) * np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S)))).astype(np.complex64)
    theta = 1.234  # one common-mode hardware/CFO phase added to every antenna+subcarrier
    measured = (trueH * np.exp(1j * theta)).astype(np.complex64)

    in_frame = CsiFrame(A, S)
    in_frame.grid[:, :] = measured
    out = CsiFrame(1, 1)
    conjugate_multiply(in_frame, out)

    assert out.grid.shape == (A - 1, S)  # cross-antenna -> (A-1) x S
    expected = (trueH[1:, :] * np.conj(trueH[0, :])).astype(np.complex64)  # theta cancels
    assert np.allclose(out.grid, expected, atol=1e-4)


def test_conjugate_multiply_cross_subcarrier_cancels_common_offset():
    rng = np.random.default_rng(1)
    S = 8
    trueH = (rng.uniform(0.5, 1.5, S) * np.exp(1j * rng.uniform(-np.pi, np.pi, S))).astype(np.complex64)
    measured = (trueH * np.exp(1j * 0.77)).astype(np.complex64)  # common across subcarriers

    in_frame = CsiFrame(1, S)
    in_frame.grid[:, :] = measured
    out = CsiFrame(1, 1)
    conjugate_multiply(in_frame, out)

    assert out.grid.shape == (1, S - 1)  # cross-subcarrier fallback
    expected = (trueH[1:] * np.conj(trueH[:-1])).astype(np.complex64)
    assert np.allclose(out.grid[0], expected, atol=1e-4)


def test_conjugate_multiply_single_subcarrier_raises():
    in_frame = CsiFrame(1, 1)  # 1 antenna, 1 subcarrier -> nothing to pair
    with pytest.raises(FrameError):
        conjugate_multiply(in_frame, CsiFrame(1, 1))


# --- Hampel ----------------------------------------------------------------------------------

def test_hampel_passes_normal_replaces_spike():
    window = np.array([1.0, 1.1, 0.9, 1.05, 50.0], dtype=np.float32)
    assert hampel(window, current=50.0, k=5.0) == pytest.approx(1.05)  # outlier -> median
    normal = np.array([1.0, 1.1, 0.9, 1.05, 1.0], dtype=np.float32)
    assert hampel(normal, current=1.0, k=5.0) == pytest.approx(1.0)    # inlier -> unchanged


# --- unwrap_step -----------------------------------------------------------------------------

def test_unwrap_step_continuity_and_wrap():
    # Small step: just accumulates.
    assert unwrap_step(0.2, 0.1, 10.0) == pytest.approx(10.1)
    # Apparent jump 3.0 -> -3.0 is physically +0.2832 rad (crossed +pi), not -6.0.
    assert unwrap_step(-3.0, 3.0, 0.0) == pytest.approx(-6.0 + 2 * np.pi, abs=1e-5)


# --- Preprocessor: geometry ------------------------------------------------------------------

def test_preprocessor_output_geometry():
    assert (Preprocessor(3, 64).out_rows, Preprocessor(3, 64).out_cols) == (2, 64)  # cross-antenna
    assert (Preprocessor(1, 64).out_rows, Preprocessor(1, 64).out_cols) == (1, 63)  # cross-subcarrier


# --- Preprocessor: matches an independent numpy implementation of the chain ------------------

def _referenceChain(frames, alpha):
    """numpy reference for the A=1 cross-subcarrier chain (Hampel disabled): conj-mult adjacent
    subcarriers -> temporal unwrap -> EMA-detrend normalize."""
    H = np.stack([f.grid[0] for f in frames])          # (T, S)
    D = H[:, 1:] * np.conj(H[:, :-1])                  # (T, S-1)
    u = np.unwrap(np.angle(D), axis=0)                 # temporal unwrap
    ema = np.empty_like(u)
    ema[0] = u[0]
    for t in range(1, len(u)):
        ema[t] = alpha * u[t] + (1 - alpha) * ema[t - 1]
    return (u - ema).astype(np.float32)


def test_preprocessor_chain_matches_numpy_reference():
    # Unit-magnitude H + wrapping motion so unwrap is exercised; Hampel disabled (k huge) to isolate
    # conj-mult + unwrap + normalize as an exact, deterministic comparison.
    rng = np.random.default_rng(5)
    T, S = 200, 6
    psi = rng.uniform(-np.pi, np.pi, S)
    t = np.arange(T)
    frames = []
    for ti in t:
        # phase per subcarrier drifts enough to wrap across frames
        phase = psi + 0.15 * ti * np.linspace(0.5, 1.5, S)
        frame = CsiFrame(1, S)
        frame.grid[:, :] = np.exp(1j * phase).astype(np.complex64)[None, :]
        frames.append(frame)

    alpha = 0.1
    pre = Preprocessor(1, S, hampel_k=1e9, normalize_alpha=alpha)
    got = np.stack([pre.process(f).copy()[0] for f in frames])  # (T, S-1)
    ref = _referenceChain(frames, alpha)
    assert np.allclose(got, ref, atol=1e-3)


# --- Preprocessor: spike rejection holds phase ----------------------------------------------

def test_preprocessor_rejects_magnitude_spike():
    # Phase-static channel with small real amplitude jitter (so the Hampel window has non-zero MAD
    # — a constant-magnitude window can't measure deviation). Differential phase stays constant ->
    # preprocessed output settles to ~0. Inject one frame with a magnitude spike AND a phase jump on
    # one cell; Hampel(magnitude) must hold the phase so the output stays ~0.
    rng = np.random.default_rng(7)
    A, S, T, f = 2, 4, 30, 15
    h0 = np.exp(1j * rng.uniform(-np.pi, np.pi, (A, S))).astype(np.complex64)
    frames = []
    for ti in range(T):
        g = (h0 * rng.uniform(0.9, 1.1)).astype(np.complex64)  # real scale -> jitter |.|, keep phase
        if ti == f:
            g[1, 0] = 50.0 * np.exp(1j * (np.angle(h0[1, 0]) + 2.0))  # |.|=50 spike + 2 rad twist
        frame = CsiFrame(A, S)
        frame.grid[:, :] = g
        frames.append(frame)

    reject = Preprocessor(A, S, hampel_window=7, hampel_k=5.0, normalize_alpha=0.1)
    out_reject = np.stack([reject.process(fr).copy() for fr in frames])

    control = Preprocessor(A, S, hampel_window=7, hampel_k=1e9, normalize_alpha=0.1)  # no rejection
    out_control = np.stack([control.process(fr).copy() for fr in frames])

    cell = out_reject[f, 0, 0]          # affected cell (antenna1 vs antenna0, subcarrier0)
    assert abs(cell) < 0.05                                  # spike held -> output stays ~0
    assert abs(out_control[f, 0, 0]) > 10 * abs(cell) + 0.1  # without rejection it glitches


# --- Preprocessor: recovers injected motion frequency (cross-subcarrier path) ----------------

def test_preprocessor_recovers_motion_frequency():
    fs, fTrue = 100.0, 0.3
    frames, gt = generateStream(
        numAntennas=1, numSubcarriers=64, sampleRateHz=fs, numFrames=1024,
        perturbationHz=fTrue, perturbationDepth=1.0, cfoHz=4.0, noiseStd=0.002, seed=9,
    )
    pre = Preprocessor(1, 64, normalize_alpha=0.01)  # gentle high-pass so 0.3 Hz passes
    out = np.stack([pre.process(f).copy()[0] for f in frames])  # (T, S-1)

    # Average power spectrum across cells (all oscillate at fTrue) to lift SNR, then band-argmax.
    n = out.shape[0]
    x = (out - out.mean(0)) * np.hanning(n)[:, None]
    nfft = 1 << (max(4 * n, 64) - 1).bit_length()
    power = (np.abs(np.fft.rfft(x, nfft, axis=0)) ** 2).mean(1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= 0.1) & (freqs <= 2.0)
    recovered = freqs[band][np.argmax(power[band])]
    assert recovered == pytest.approx(gt["perturbation_hz"], abs=0.05)
