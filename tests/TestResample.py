"""resample_uniform: jittery CSI stream -> uniform grid (fixes the front-end fs_ok drops)."""

import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.Source import resample_uniform


def _frame(t, vals):
    """One CsiFrame (1, S) at time t with complex subcarrier values `vals`."""
    fr = CsiFrame(1, len(vals))
    fr.grid[0, :] = np.asarray(vals, dtype=np.complex64)
    fr.timestamp = float(t)
    fr.node_id = 7
    return fr


def _jittery_stream(fs_true=120.0, dur=1.0, seed=0):
    """Frames at jittery intervals carrying a known per-subcarrier complex sinusoid."""
    rng = np.random.default_rng(seed)
    n = int(fs_true * dur)
    # non-uniform timestamps: base grid + jitter, kept monotonic
    t = np.cumsum(rng.uniform(0.3, 1.7, size=n)) / fs_true
    freqs = np.array([3.0, 7.0])  # Hz, one per subcarrier
    frames = [_frame(ti, np.exp(2j * np.pi * freqs * ti)) for ti in t]
    return frames, freqs


def test_uniform_grid_spacing_and_span():
    frames, _ = _jittery_stream()
    fs = 100.0
    out = resample_uniform(frames, fs)
    ts = np.array([f.timestamp for f in out])
    dt = np.diff(ts)
    assert np.allclose(dt, 1.0 / fs, rtol=1e-6)          # exactly uniform
    assert ts[0] == pytest.approx(frames[0].timestamp)    # starts at first sample
    assert ts[-1] <= frames[-1].timestamp + 1e-9          # stays within the input span
    assert all(f.node_id == 7 for f in out)               # metadata preserved


def test_values_track_the_signal():
    frames, freqs = _jittery_stream(fs_true=400.0)        # dense input -> small interp error
    fs = 100.0
    out = resample_uniform(frames, fs)
    err = 0.0
    for fr in out:
        truth = np.exp(2j * np.pi * freqs * fr.timestamp)
        err = max(err, float(np.max(np.abs(fr.grid[0, :] - truth))))
    assert err < 0.05                                     # linear interp of a dense sinusoid is tight


def test_edge_cases():
    assert resample_uniform([], 100.0) == []
    one = [_frame(0.0, [1 + 1j])]
    assert len(resample_uniform(one, 100.0)) == 1         # <2 frames: returned as-is
    with pytest.raises(ValueError):
        resample_uniform(one, 0.0)
