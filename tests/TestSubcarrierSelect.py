"""Phase 3 (step 3c) — signal/subcarrier-select: NBVI scoring + non-consecutive selection."""

import numpy as np
import pytest

from wavetrace import nbvi_scores, select_subcarriers_nbvi


def _altMatrix(sigma):
    """Frames x subcarriers amplitude matrix with mean 1 and exact per-subcarrier std = sigma[s]
    (values alternate 1 +/- sigma each frame), so NBVI score == sigma[s] (mu=1 -> mu^2=1)."""
    sigma = np.asarray(sigma, dtype=np.float32)
    F, S = 200, sigma.size
    sign = np.where(np.arange(F) % 2 == 0, 1.0, -1.0)[:, None]  # (F,1)
    return (1.0 + sign * sigma[None, :]).astype(np.float32)


def test_nbvi_scores_match_relative_variation():
    sigma = [0.0, 0.0, 0.5, 0.45, 0.0, 0.3, 0.0, 0.1]
    scores = np.array(nbvi_scores(_altMatrix(sigma), alpha=0.75))
    assert np.allclose(scores, sigma, atol=1e-4)  # mu=1 -> NBVI == sigma
    assert int(np.argmax(scores)) == 2


def test_select_is_non_consecutive_and_picks_top():
    sigma = [0.0, 0.0, 0.5, 0.45, 0.0, 0.3, 0.0, 0.1]
    sel = select_subcarriers_nbvi(_altMatrix(sigma), max_subcarriers=12)
    assert 2 in sel          # highest score
    assert 3 not in sel      # adjacent to 2 -> blocked for spectral diversity
    assert 5 in sel          # next non-adjacent high scorer
    assert all(b - a > 1 for a, b in zip(sorted(sel), sorted(sel)[1:]))  # no two consecutive


def test_select_respects_max():
    sigma = np.linspace(0.01, 0.5, 32).astype(np.float32)  # all distinct, all gated-in
    sel = select_subcarriers_nbvi(_altMatrix(sigma), max_subcarriers=6)
    assert len(sel) <= 6
    assert all(b - a > 1 for a, b in zip(sel, sel[1:]))


def test_noise_gate_drops_low_amplitude_subcarrier():
    # Subcarrier 5 has a tiny mean amplitude but a huge sigma/mu^2 (would top NBVI) — the amplitude
    # noise gate must drop it anyway (this is how DC/guard bands get excluded without hardcoding).
    S = 10
    sigma = np.full(S, 0.05, dtype=np.float32)
    amp = _altMatrix(sigma)            # mean ~1 everywhere
    amp[:, 5] = np.where(np.arange(amp.shape[0]) % 2 == 0, 0.0015, 0.0005)  # mean ~0.001, high CV
    raw_scores = np.array(nbvi_scores(amp))
    assert int(np.argmax(raw_scores)) == 5             # it WOULD win on score
    sel = select_subcarriers_nbvi(amp, noise_gate_percentile=0.15)
    assert 5 not in sel                                 # ...but the gate drops it


def test_select_is_deterministic():
    sigma = np.linspace(0.01, 0.5, 20).astype(np.float32)
    amp = _altMatrix(sigma)
    assert select_subcarriers_nbvi(amp) == select_subcarriers_nbvi(amp)  # stable set
