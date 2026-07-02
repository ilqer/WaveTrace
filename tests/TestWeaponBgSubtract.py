"""Tests for IC-path background subtraction in iter_windows (weapon variance feature)."""
import numpy as np
import pytest

from wavetrace import CsiFrame
from wavetrace.Frontend import iter_windows
from wavetrace.recognition.Weapon import VARIANCE_FEATURE  # column 9 = σ²-series window mean

S = 32
WINDOW, HOP = 32, 16


def _frames(mags):
    """Build single-antenna CsiFrames whose antenna-collapsed magnitude == each row of `mags`."""
    out = []
    for i, m in enumerate(mags):
        fr = CsiFrame(1, S)
        fr.timestamp = float(i) * 0.01
        fr.grid[0, :] = m.astype(np.complex64)
        out.append(fr)
    return out


def _ic_blocks(frames, ic_baseline):
    subc = np.arange(S, dtype=np.intp)
    return np.stack([ic.copy() for _t, _f, _i, ic in iter_windows(
        frames, subc, None, window=WINDOW, hop=HOP, intercarrier=True, ic_baseline=ic_baseline)])


def test_ic_baseline_none_is_byte_identical():
    rng = np.random.default_rng(0)
    mags = np.abs(rng.normal(5.0, 1.0, (80, S)))
    frames = _frames(mags)
    np.testing.assert_array_equal(_ic_blocks(frames, None), _ic_blocks(frames, None))
    # Explicit None equals default omitted path.
    subc = np.arange(S, dtype=np.intp)
    default = np.stack([ic.copy() for _t, _f, _i, ic in iter_windows(
        frames, subc, None, window=WINDOW, hop=HOP, intercarrier=True)])
    np.testing.assert_array_equal(_ic_blocks(frames, None), default)


def test_ic_baseline_nulls_the_static_room_variance():
    """Subtracting room baseline drops window-mean variance by orders of magnitude."""
    rng = np.random.default_rng(1)
    room = np.linspace(2.0, 12.0, S)          # Static room shape.
    noise = rng.normal(0.0, 0.05, (80, S))     # Tiny perturbation.
    frames = _frames(room[None, :] + noise)

    without = _ic_blocks(frames, None)[:, VARIANCE_FEATURE].mean()
    withbg = _ic_blocks(frames, room.astype(np.float32))[:, VARIANCE_FEATURE].mean()
    assert withbg < without * 0.05  # Residual variance is a tiny fraction.


def test_ic_baseline_width_mismatch_raises():
    frames = _frames(np.abs(np.random.default_rng(2).normal(5.0, 1.0, (40, S))))
    with pytest.raises(ValueError, match="width"):
        _ic_blocks(frames, np.ones(S + 4, dtype=np.float32))
