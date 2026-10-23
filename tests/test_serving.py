"""Per-link serving math: calibration width and the last-window prediction."""

from types import SimpleNamespace

import numpy as np

from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.recognition import serving, weapon_serving
from wavetrace.recognition.serving import last_window_proba, required_subcarrier_width
from wavetrace.recognition.weapon_serving import dwell_proba
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


def _windows(frames, intercarrier=False):
    """Every window's (features, image, intercarrier features), copied out of the reused buffers."""
    return [(features.copy(), image.copy(), None if intercarrier_features is None else intercarrier_features.copy())
            for _, features, image, intercarrier_features in iter_windows(
                resample_uniform(frames, SAMPLE_RATE_HZ), RESULT.subcarriers, None, window=CONFIG.window,
                hop=CONFIG.hop, intercarrier=intercarrier, image_subcarriers=RESULT.image_subcarriers)]


def _count_copies(monkeypatch, module):
    copied_windows = []
    copy_window_once = serving.copy_window

    def copy_window(*window):
        copied_windows.append(window)
        return copy_window_once(*window)

    monkeypatch.setattr(module, "copy_window", copy_window)
    return copied_windows


def _assert_has_frames_after_the_last_window(frames):
    """So the views must survive iter_windows pushing frames past the last yield."""
    assert (len(resample_uniform(frames, SAMPLE_RATE_HZ)) - CONFIG.window) % CONFIG.hop != 0


def _assert_is_window(window_view, window):
    features, image, intercarrier_features = window
    np.testing.assert_array_equal(window_view.features, features)
    np.testing.assert_array_equal(window_view.image, image)
    if intercarrier_features is None:
        assert window_view.intercarrier_features is None
    else:
        np.testing.assert_array_equal(window_view.intercarrier_features, intercarrier_features)


def test_last_window_proba_predicts_only_the_last_window(monkeypatch):
    frames = _frames(200)
    _assert_has_frames_after_the_last_window(frames)
    copied_windows = _count_copies(monkeypatch, serving)
    session = _RecordingSession()
    proba, window_view = last_window_proba(frames, SAMPLE_RATE_HZ, RESULT, None, CONFIG, False, _features,
                                           session)

    windows = _windows(frames)
    assert len(windows) > 1
    assert len(session.inputs) == 1
    np.testing.assert_array_equal(session.inputs[0], windows[-1][0])
    assert proba[0] == float(np.sum(windows[-1][0]))
    _assert_is_window(window_view, windows[-1])
    assert len(copied_windows) == 1


def test_last_window_proba_is_none_without_a_full_window():
    session = _RecordingSession()
    assert last_window_proba(
        _frames(10), SAMPLE_RATE_HZ, RESULT, None, CONFIG, False, _features, session) == (None, None)
    assert session.inputs == []


def test_dwell_proba_averages_every_window_and_views_the_last(monkeypatch):
    frames = _frames(200)
    _assert_has_frames_after_the_last_window(frames)
    copied_windows = _count_copies(monkeypatch, weapon_serving)
    session = _RecordingSession()
    entry = SimpleNamespace(config=CONFIG, calibration=RESULT, gain_lock=None, should_use_intercarrier=True,
                            intercarrier_baseline=None, pick_input=_features, session=session)
    proba, window_view = dwell_proba(frames, SAMPLE_RATE_HZ, entry)

    windows = _windows(frames, intercarrier=True)
    np.testing.assert_allclose(proba[0], np.mean([float(np.sum(features)) for features, _, _ in windows]))
    _assert_is_window(window_view, windows[-1])
    assert len(copied_windows) == 1
