"""Per-link serving math: calibration width and the last-window prediction."""

from types import SimpleNamespace

import numpy as np

from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.recognition.serving import last_window_proba, required_subcarrier_width
from wavetrace.domain.dsp.resample import resample_uniform
from wavetrace.synthetic import generate_stream

SAMPLE_RATE_HZ = 100.0
RESULT = SimpleNamespace(subcarriers=[1, 2, 3, 4], image_subcarriers=[1, 2, 3, 4, 9])
CONFIG = SimpleNamespace(window=32, hop=16)


class _RecordingSession:
    def __init__(self):
        self.inputs = []

    def predict_window_proba(self, window_input):
        self.inputs.append(np.array(window_input))
        return np.array([float(np.sum(window_input))])


def _frames(num_frames):
    frames, _ = generate_stream(
        num_antennas=1, num_subcarriers=16, sample_rate_hz=SAMPLE_RATE_HZ, num_frames=num_frames,
        perturbation_hz=1.0, perturbation_depth=0.5, cfo_hz=0.0, noise_std=0.01, seed=3,
    )
    return frames


def _features(features, image, intercarrier):
    return features


def test_required_subcarrier_width_covers_feature_and_image_indices():
    assert required_subcarrier_width(RESULT) == 10


def test_last_window_proba_predicts_only_the_last_window():
    frames = _frames(200)
    session = _RecordingSession()
    proba = last_window_proba(frames, SAMPLE_RATE_HZ, RESULT, None, CONFIG, False, _features, session)

    windows = [features.copy() for _, features, _, _ in iter_windows(
        resample_uniform(frames, SAMPLE_RATE_HZ), RESULT.subcarriers, None,
        window=CONFIG.window, hop=CONFIG.hop, image_subcarriers=RESULT.image_subcarriers)]
    assert len(windows) > 1
    assert len(session.inputs) == 1
    np.testing.assert_array_equal(session.inputs[0], windows[-1])
    assert proba[0] == float(np.sum(windows[-1]))


def test_last_window_proba_is_none_without_a_full_window():
    session = _RecordingSession()
    assert last_window_proba(
        _frames(10), SAMPLE_RATE_HZ, RESULT, None, CONFIG, False, _features, session) is None
    assert session.inputs == []
