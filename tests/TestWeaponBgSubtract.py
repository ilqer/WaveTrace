"""Item 10 / CAUSE 2B — IC-path background subtraction in iter_windows (weapon σ²[p])."""
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
        fr.timestamp = float(i) * 0.01  # 100 Hz
        fr.grid[0, :] = m.astype(np.complex64)  # real, non-negative -> |grid| == m
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
    # explicit None must equal the default (omitted) path
    subc = np.arange(S, dtype=np.intp)
    default = np.stack([ic.copy() for _t, _f, _i, ic in iter_windows(
        frames, subc, None, window=WINDOW, hop=HOP, intercarrier=True)])
    np.testing.assert_array_equal(_ic_blocks(frames, None), default)


def test_ic_baseline_nulls_the_static_room_variance():
    """A strong per-subcarrier room shape + small motion noise: subtracting the room baseline drops
    the σ²[p] window-mean (column 9) by orders of magnitude — the whole point of CAUSE 2B."""
    rng = np.random.default_rng(1)
    room = np.linspace(2.0, 12.0, S)          # high inter-subcarrier variance "room" shape
    noise = rng.normal(0.0, 0.05, (80, S))     # tiny perturbation
    frames = _frames(room[None, :] + noise)

    without = _ic_blocks(frames, None)[:, VARIANCE_FEATURE].mean()
    withbg = _ic_blocks(frames, room.astype(np.float32))[:, VARIANCE_FEATURE].mean()
    assert withbg < without * 0.05  # residual variance is a tiny fraction of the full-channel variance


def test_ic_baseline_width_mismatch_raises():
    frames = _frames(np.abs(np.random.default_rng(2).normal(5.0, 1.0, (40, S))))
    with pytest.raises(ValueError, match="width"):
        _ic_blocks(frames, np.ones(S + 4, dtype=np.float32))
