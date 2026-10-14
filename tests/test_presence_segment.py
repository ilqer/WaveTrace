"""PresenceSegmenter: a streaming variance gate that flags active segments.

The fixture is a high-modulation stretch between two quiet stretches.
"""

import numpy as np
import pytest

from wavetrace import PresenceSegmenter, WaveTraceError
from wavetrace.synthetic import generate_stream


def _energy(frame):
    # Antenna-collapsed per-subcarrier magnitude vector.
    return np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)


def _quiet_active_quiet(frames_per_stretch=80, seed=5):
    common = dict(num_antennas=1, num_subcarriers=32, sample_rate_hz=100.0, cfo_hz=0.0,
                  noise_std=0.001, seed=seed)
    still = dict(perturbation_hz=0.0, perturbation_depth=0.0, amplitude_hz=0.0, amplitude_depth=0.0)
    first_quiet, _ = generate_stream(**common, num_frames=frames_per_stretch, **still)
    active, _ = generate_stream(**common, num_frames=frames_per_stretch, perturbation_hz=0.0,
                                perturbation_depth=0.0, amplitude_hz=2.0,
                                amplitude_depth=0.8)  # fluctuating -> high temporal CV
    second_quiet, _ = generate_stream(**common, num_frames=frames_per_stretch, **still)
    # active region = [frames_per_stretch, 2 * frames_per_stretch)
    return first_quiet + active + second_quiet, frames_per_stretch


def test_presence_segmenter_flags_active_region():
    frames, frames_per_stretch = _quiet_active_quiet()
    segmenter = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    states = np.array([segmenter.push(_energy(frame)) for frame in frames])
    assert states[frames_per_stretch + 40]          # deep inside the active region -> active
    assert not states[40]          # deep inside the first quiet region -> inactive
    assert not states[2 * frames_per_stretch + 40]  # deep inside the last quiet region -> inactive


def test_presence_segmenter_reports_segment_bounds():
    frames, frames_per_stretch = _quiet_active_quiet()
    segmenter = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    start = end = None
    for i, frame in enumerate(frames):
        segmenter.push(_energy(frame))
        if segmenter.segment_closed:
            start, end = segmenter.last_segment_start, segmenter.last_segment_end
    assert start is not None and end is not None
    # Window lags true bounds by < window length. Bounds must bracket the active region.
    assert frames_per_stretch <= start <= frames_per_stretch + 20
    assert 2 * frames_per_stretch <= end <= 2 * frames_per_stretch + 20
    assert end > start


def test_presence_segmenter_hysteresis_no_chatter():
    # Hysteresis: prevents re-triggering on ripples after an active burst.
    frames, frames_per_stretch = _quiet_active_quiet()
    segmenter = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    transitions = 0
    was_active = False
    for frame in frames:
        is_active = segmenter.push(_energy(frame))
        transitions += int(is_active != was_active)
        was_active = is_active
    assert transitions == 2  # exactly one rise + one fall for a single active burst


def test_presence_segmenter_bad_construction_raises():
    with pytest.raises(WaveTraceError):
        PresenceSegmenter(window=0, enter_cv=0.05, exit_cv=0.02)
    with pytest.raises(WaveTraceError):
        PresenceSegmenter(window=20, enter_cv=0.02, exit_cv=0.05)  # enter must be >= exit
