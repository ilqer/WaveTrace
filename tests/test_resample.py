"""resample_uniform: jittery CSI stream -> uniform grid (fixes the front-end is_sample_rate_ok drops)."""

import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.domain.dsp.resample import resample_uniform


def _frame(timestamp_s, values):
    """One CsiFrame (1, S) at the given time with complex subcarrier `values`."""
    frame = CsiFrame(1, len(values))
    frame.grid[0, :] = np.asarray(values, dtype=np.complex64)
    frame.timestamp = float(timestamp_s)
    frame.node_id = 7
    return frame


def _jittery_stream(true_sample_rate_hz=120.0, duration_s=1.0, seed=0):
    """Frames at jittery intervals carrying a known per-subcarrier complex sinusoid."""
    rng = np.random.default_rng(seed)
    frame_count = int(true_sample_rate_hz * duration_s)
    # non-uniform timestamps: base grid + jitter, kept monotonic
    timestamps_s = np.cumsum(rng.uniform(0.3, 1.7, size=frame_count)) / true_sample_rate_hz
    frequencies_hz = np.array([3.0, 7.0])  # one per subcarrier
    frames = [_frame(timestamp_s, np.exp(2j * np.pi * frequencies_hz * timestamp_s))
              for timestamp_s in timestamps_s]
    return frames, frequencies_hz


def test_uniform_grid_spacing_and_span():
    frames, _ = _jittery_stream()
    sample_rate_hz = 100.0
    out = resample_uniform(frames, sample_rate_hz)
    timestamps_s = np.array([frame.timestamp for frame in out])
    intervals_s = np.diff(timestamps_s)
    assert np.allclose(intervals_s, 1.0 / sample_rate_hz, rtol=1e-6)
    assert timestamps_s[0] == pytest.approx(frames[0].timestamp)
    assert timestamps_s[-1] <= frames[-1].timestamp + 1e-9
    assert all(frame.node_id == 7 for frame in out)


def test_values_track_the_signal():
    frames, frequencies_hz = _jittery_stream(true_sample_rate_hz=400.0)  # dense input -> small interp error
    sample_rate_hz = 100.0
    out = resample_uniform(frames, sample_rate_hz)
    max_error = 0.0
    for frame in out:
        truth = np.exp(2j * np.pi * frequencies_hz * frame.timestamp)
        max_error = max(max_error, float(np.max(np.abs(frame.grid[0, :] - truth))))
    assert max_error < 0.05                               # linear interp of a dense sinusoid is tight


def test_edge_cases():
    assert resample_uniform([], 100.0) == []
    one = [_frame(0.0, [1 + 1j])]
    assert len(resample_uniform(one, 100.0)) == 1         # <2 frames: returned as-is
    with pytest.raises(ValueError):
        resample_uniform(one, 0.0)
