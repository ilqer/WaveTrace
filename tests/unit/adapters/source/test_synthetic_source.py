from types import SimpleNamespace

import numpy as np

from wavetrace.adapters.source import build_csi_source, build_synthetic_source


def test_typed_synthetic_source_yields_the_frames_of_the_options_bag():
    options = SimpleNamespace(recording=None, synthetic=True, antennas=2, subcarriers=16, fs=50.0, duration=3.0,
                              presence="0.5:1.5", weapon="1:2", weapon_depth=0.5, seed=3)
    typed = build_synthetic_source(num_antennas=2, num_subcarriers=16, sample_rate_hz=50.0, duration_s=3.0,
                                   presence_spans=[(0.5, 1.5)], weapon_spans=[(1.0, 2.0)],
                                   weapon_signature_depth=0.5, seed=3)
    expected_frames, frames = list(build_csi_source(options).frames()), list(typed.frames())
    assert len(frames) == len(expected_frames) == 150
    assert [frame.timestamp for frame in frames] == [frame.timestamp for frame in expected_frames]
    assert all(np.array_equal(frame.grid, expected.grid) for frame, expected in zip(frames, expected_frames))


def test_no_presence_spans_means_present_throughout_as_before():
    options = SimpleNamespace(recording=None, synthetic=True, antennas=1, subcarriers=8, fs=50.0, duration=1.0,
                              presence="", weapon="", weapon_depth=0.0, seed=1)
    typed = build_synthetic_source(num_antennas=1, num_subcarriers=8, sample_rate_hz=50.0, duration_s=1.0,
                                   presence_spans=[], weapon_spans=[], weapon_signature_depth=0.0, seed=1)
    assert all(np.array_equal(frame.grid, expected.grid)
               for frame, expected in zip(typed.frames(), build_csi_source(options).frames()))
