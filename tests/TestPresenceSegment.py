"""PresenceSegmenter — streaming variance-gate active-segment detector (Option-A LOF stand-in).

Validates: it flags the active (fluctuating) region and not the quiet ones, reports a plausible
[start, end) for the closed segment, applies enter/exit hysteresis, and rejects bad construction.
The active region is a high-amplitude-modulation stretch sandwiched between two still stretches.
"""

import numpy as np
import pytest

from wavetrace import PresenceSegmenter, WaveTraceError
from fixtures.SyntheticCsi import generateStream


def _energy(frame):
    # antenna-collapsed per-subcarrier magnitude vector (what the segmenter ingests)
    return np.abs(np.asarray(frame.grid)).mean(axis=0).astype(np.float32)


def _quiet_active_quiet(n=80, seed=5):
    common = dict(numAntennas=1, numSubcarriers=32, sampleRateHz=100.0, cfoHz=0.0,
                  noiseStd=0.001, seed=seed)
    still = dict(perturbationHz=0.0, perturbationDepth=0.0, amplitudeHz=0.0, amplitudeDepth=0.0)
    q1, _ = generateStream(**common, numFrames=n, **still)
    ac, _ = generateStream(**common, numFrames=n, perturbationHz=0.0, perturbationDepth=0.0,
                           amplitudeHz=2.0, amplitudeDepth=0.8)  # fluctuating -> high temporal CV
    q2, _ = generateStream(**common, numFrames=n, **still)
    return q1 + ac + q2, n  # active region = [n, 2n)


def test_presence_segmenter_flags_active_region():
    frames, n = _quiet_active_quiet()
    seg = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    states = np.array([seg.push(_energy(f)) for f in frames])
    assert states[n + 40]          # deep inside the active region -> active
    assert not states[40]          # deep inside the first quiet region -> inactive
    assert not states[2 * n + 40]  # deep inside the last quiet region -> inactive


def test_presence_segmenter_reports_segment_bounds():
    frames, n = _quiet_active_quiet()
    seg = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    start = end = None
    for i, f in enumerate(frames):
        seg.push(_energy(f))
        if seg.segment_closed:
            start, end = seg.last_segment_start, seg.last_segment_end
    assert start is not None and end is not None
    # The window lags onset/offset by < the window length; bounds should bracket the active region.
    assert n <= start <= n + 20
    assert 2 * n <= end <= 2 * n + 20
    assert end > start


def test_presence_segmenter_hysteresis_no_chatter():
    # A signal hovering between exit and enter must NOT flip state once below enter (hysteresis): a
    # quiet tail after one active burst stays inactive rather than re-triggering on small ripples.
    frames, n = _quiet_active_quiet()
    seg = PresenceSegmenter(window=20, enter_cv=0.05, exit_cv=0.02)
    transitions = 0
    prev = False
    for f in frames:
        cur = seg.push(_energy(f))
        transitions += int(cur != prev)
        prev = cur
    assert transitions == 2  # exactly one rise + one fall for a single active burst


def test_presence_segmenter_bad_construction_raises():
    with pytest.raises(WaveTraceError):
        PresenceSegmenter(window=0, enter_cv=0.05, exit_cv=0.02)
    with pytest.raises(WaveTraceError):
        PresenceSegmenter(window=20, enter_cv=0.02, exit_cv=0.05)  # enter must be >= exit
