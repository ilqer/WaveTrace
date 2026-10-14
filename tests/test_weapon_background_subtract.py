"""IC-path background subtraction in iter_windows (the weapon variance feature)."""
import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.domain.dsp.frontend import iter_windows
from wavetrace.domain.recognition import VARIANCE_FEATURE

SUBCARRIER_COUNT = 32
WINDOW, HOP = 32, 16


def _frames(magnitudes):
    """Build single-antenna CsiFrames whose antenna-collapsed magnitude == each row of `magnitudes`."""
    out = []
    for i, magnitude in enumerate(magnitudes):
        frame = CsiFrame(1, SUBCARRIER_COUNT)
        frame.timestamp = float(i) * 0.01
        frame.grid[0, :] = magnitude.astype(np.complex64)
        out.append(frame)
    return out


def _ic_blocks(frames, intercarrier_baseline):
    subcarrier_indices = np.arange(SUBCARRIER_COUNT, dtype=np.intp)
    return np.stack([intercarrier_block.copy()
                     for _timestamp, _features, _image, intercarrier_block in iter_windows(
                         frames, subcarrier_indices, None, window=WINDOW, hop=HOP, intercarrier=True,
                         intercarrier_baseline=intercarrier_baseline)])


def test_ic_baseline_none_is_byte_identical():
    rng = np.random.default_rng(0)
    magnitudes = np.abs(rng.normal(5.0, 1.0, (80, SUBCARRIER_COUNT)))
    frames = _frames(magnitudes)
    np.testing.assert_array_equal(_ic_blocks(frames, None), _ic_blocks(frames, None))
    # Explicit None equals default omitted path.
    subcarrier_indices = np.arange(SUBCARRIER_COUNT, dtype=np.intp)
    default = np.stack([intercarrier_block.copy()
                        for _timestamp, _features, _image, intercarrier_block in iter_windows(
                            frames, subcarrier_indices, None, window=WINDOW, hop=HOP, intercarrier=True)])
    np.testing.assert_array_equal(_ic_blocks(frames, None), default)


def test_ic_baseline_nulls_the_static_room_variance():
    """Subtracting room baseline drops window-mean variance by orders of magnitude."""
    rng = np.random.default_rng(1)
    room = np.linspace(2.0, 12.0, SUBCARRIER_COUNT)
    noise = rng.normal(0.0, 0.05, (80, SUBCARRIER_COUNT))
    frames = _frames(room[None, :] + noise)

    without = _ic_blocks(frames, None)[:, VARIANCE_FEATURE].mean()
    with_baseline = _ic_blocks(frames, room.astype(np.float32))[:, VARIANCE_FEATURE].mean()
    assert with_baseline < without * 0.05


def test_ic_baseline_width_mismatch_raises():
    frames = _frames(np.abs(np.random.default_rng(2).normal(5.0, 1.0, (40, SUBCARRIER_COUNT))))
    with pytest.raises(ValueError, match="width"):
        _ic_blocks(frames, np.ones(SUBCARRIER_COUNT + 4, dtype=np.float32))
