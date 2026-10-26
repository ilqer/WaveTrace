"""NBVI scoring and non-consecutive subcarrier selection."""

import numpy as np

from wavetrace import nbvi_scores, select_subcarriers_nbvi


def _alternating_matrix(sigma):
    """Frames x subcarriers amplitude matrix with mean 1 and exact per-subcarrier std = sigma[s]
    (values alternate 1 +/- sigma each frame), so NBVI score == sigma[s] (mu=1 -> mu^2=1)."""
    sigma = np.asarray(sigma, dtype=np.float32)
    F = 200
    sign = np.where(np.arange(F) % 2 == 0, 1.0, -1.0)[:, None]
    return (1.0 + sign * sigma[None, :]).astype(np.float32)


def test_nbvi_scores_match_relative_variation():
    sigma = [0.0, 0.0, 0.5, 0.45, 0.0, 0.3, 0.0, 0.1]
    scores = np.array(nbvi_scores(_alternating_matrix(sigma), alpha=0.75))
    assert np.allclose(scores, sigma, atol=1e-4)  # mu=1 -> NBVI == sigma
    assert int(np.argmax(scores)) == 2


def test_select_is_non_consecutive_and_picks_top():
    sigma = [0.0, 0.0, 0.5, 0.45, 0.0, 0.3, 0.0, 0.1]
    selected = select_subcarriers_nbvi(_alternating_matrix(sigma), max_subcarriers=12)
    assert 2 in selected          # highest score
    assert 3 not in selected      # adjacent to 2 -> blocked for spectral diversity
    assert 5 in selected          # next non-adjacent high scorer
    assert all(right - left > 1
               for left, right in zip(sorted(selected), sorted(selected)[1:]))  # no two consecutive


def test_select_respects_max():
    sigma = np.linspace(0.01, 0.5, 32).astype(np.float32)  # all distinct, all gated-in
    selected = select_subcarriers_nbvi(_alternating_matrix(sigma), max_subcarriers=6)
    assert len(selected) <= 6
    assert all(right - left > 1 for left, right in zip(selected, selected[1:]))


def test_noise_gate_drops_low_amplitude_subcarrier():
    # Subcarrier 5 would top NBVI on sigma/mu^2 alone; the amplitude noise gate must drop it anyway.
    S = 10
    sigma = np.full(S, 0.05, dtype=np.float32)
    amplitudes = _alternating_matrix(sigma)  # mean ~1 everywhere
    # mean ~0.001, high CV
    amplitudes[:, 5] = np.where(np.arange(amplitudes.shape[0]) % 2 == 0, 0.0015, 0.0005)
    raw_scores = np.array(nbvi_scores(amplitudes))
    assert int(np.argmax(raw_scores)) == 5             # it WOULD win on score
    selected = select_subcarriers_nbvi(amplitudes, noise_gate_percentile=0.15)
    assert 5 not in selected                                 # ...but the gate drops it


def test_select_is_deterministic():
    sigma = np.linspace(0.01, 0.5, 20).astype(np.float32)
    amplitudes = _alternating_matrix(sigma)
    assert select_subcarriers_nbvi(amplitudes) == select_subcarriers_nbvi(amplitudes)  # stable set
