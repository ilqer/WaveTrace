"""Spatial localization (AoA) tests — plant a known azimuth into a ULA and assert MUSIC/Bartlett
recover it, plus the range/heatmap/aggregate/serialize plumbing. No hardware: the steering is built
with the same convention the Localizer uses, so a clean rank-1 source must peak at the planted angle.
"""

import io
import json

import numpy as np
import pytest

from wavetrace.Localize import (
    Localizer,
    Localization,
    Tracker,
    save_localization,
    SPEED_OF_LIGHT,
)


def _meas(angle, conf, t, rng=5.0):
    """A bare Localization measurement (only the fields the Tracker reads are meaningful)."""
    e = np.empty(0)
    return Localization(timestamp=t, angles_deg=e, angle_spectrum=e, ranges_m=e, range_profile=e,
                        heatmap=e, peak_angle_deg=angle, peak_range_m=rng, x_m=0.0, y_m=0.0,
                        confidence=conf)

A = 2          # 2-antenna ESP32 / Pi
S = 64
SPACING = 0.5  # lambda/2


def _planted_grid(angle_deg, *, num_ant=A, num_sub=S, spacing=SPACING, noise=0.01, seed=0):
    """(A, S) CSI of ONE plane wave from `angle_deg` (broadside=0): a(theta) x a per-subcarrier gain."""
    rng = np.random.default_rng(seed)
    m = np.arange(num_ant)
    steer = np.exp(1j * 2 * np.pi * spacing * m * np.sin(np.deg2rad(angle_deg)))  # (A,)
    gain = rng.standard_normal(num_sub) + 1j * rng.standard_normal(num_sub)        # (S,)
    H = np.outer(steer, gain)
    H += noise * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    return H.astype(np.complex64)


class _Frame:
    """Minimal CsiFrame stand-in (grid + timestamp) for the stream/aggregate paths."""
    def __init__(self, grid, t):
        self.grid = grid
        self.timestamp = t


@pytest.mark.parametrize("angle", [0.0, 20.0, -30.0, 45.0])
def test_music_recovers_planted_angle(angle):
    loc = Localizer(A, spacing=SPACING, method="music")
    est = loc.locate(_planted_grid(angle, seed=1)).peak_angle_deg
    assert abs(est - angle) <= 3.0  # 1-deg grid + clean rank-1 source -> within a few degrees


@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
def test_bartlett_recovers_planted_angle(angle):
    loc = Localizer(A, spacing=SPACING, method="bartlett")
    est = loc.locate(_planted_grid(angle, seed=2)).peak_angle_deg
    assert abs(est - angle) <= 6.0  # beamformer is lower-resolution than MUSIC


def test_spectrum_normalized_and_confident():
    loc = Localizer(A, method="music")
    out = loc.locate(_planted_grid(15.0, seed=3))
    assert out.angle_spectrum.min() >= 0.0 and out.angle_spectrum.max() == pytest.approx(1.0)
    assert 0.0 < out.confidence <= 1.0  # a single sharp arrival -> peaked spectrum


def test_locate_per_frame_is_azimuth_spectrum():
    # a single frame can't resolve a room map: the per-frame heatmap is the (1, G) azimuth spectrum
    loc = Localizer(A, method="music", num_angles=181, max_range_m=15.0)
    out = loc.locate(_planted_grid(20.0, seed=4))
    assert out.heatmap.shape == (1, out.angles_deg.size)
    assert out.ranges_m.size > 0  # a coarse 1-D range profile still places (x, y)
    th = np.deg2rad(out.peak_angle_deg)
    assert out.x_m == pytest.approx(out.peak_range_m * np.sin(th), rel=1e-5, abs=1e-9)
    assert out.y_m == pytest.approx(out.peak_range_m * np.cos(th), rel=1e-5, abs=1e-9)


def test_aggregate_is_joint_2d_room_map():
    # range enabled + >1 frame -> the aggregate heatmap is the joint 2-D (range × angle) MUSIC map
    loc = Localizer(A, method="music", num_ranges=48, max_range_m=12.0)
    frames = [_Frame(_planted_grid(25.0, seed=i), t=i) for i in range(12)]
    agg = loc.aggregate(frames)
    assert agg.heatmap.shape == (48, 181)
    assert abs(agg.peak_angle_deg - 25.0) <= 4.0
    assert 0.0 <= agg.peak_range_m <= 12.0


def test_joint_2d_resolves_planted_range():
    # inflate the bandwidth so the delay phase ramp is observable (real WiFi BW is too small for
    # room-scale range — that is the documented caveat, not an algorithm limit)
    df, S = 5e6, 32

    def grid(angle, range_m, seed):
        rng = np.random.default_rng(seed)
        steer = np.exp(1j * 2 * np.pi * SPACING * np.arange(A) * np.sin(np.deg2rad(angle)))  # (A,)
        sub = np.exp(-1j * 2 * np.pi * df * np.arange(S) * (range_m / SPEED_OF_LIGHT))         # (S,)
        g = rng.standard_normal() + 1j * rng.standard_normal()  # per-frame gain (decorrelates snaps)
        H = g * np.outer(steer, sub)
        H += 0.01 * (rng.standard_normal((A, S)) + 1j * rng.standard_normal((A, S)))
        return H.astype(np.complex64)

    loc = Localizer(A, method="music", subcarrier_spacing_hz=df, max_range_m=20.0, num_ranges=80)
    frames = [_Frame(grid(15.0, 6.0, i), t=i) for i in range(16)]
    agg = loc.aggregate(frames)
    assert abs(agg.peak_angle_deg - 15.0) <= 5.0
    assert abs(agg.peak_range_m - 6.0) <= 3.0


def test_no_range_mode_is_azimuth_only():
    loc = Localizer(A, method="music", range_enabled=False)
    out = loc.locate(_planted_grid(10.0, seed=5))
    assert out.ranges_m.size == 0 and np.isnan(out.peak_range_m)
    assert out.heatmap.shape == (1, out.angles_deg.size)


def test_aggregate_azimuth_fallback_is_steady():
    # range disabled -> aggregate averages the per-frame 1-D AoA spectra; robust to per-subcarrier
    # gain + heavy noise (the 1-D antenna covariance is a clean single-source problem)
    loc = Localizer(A, method="music", range_enabled=False)
    frames = [_Frame(_planted_grid(30.0, seed=10 + i, noise=0.3), t=i * 0.01) for i in range(20)]
    agg = loc.aggregate(frames)
    assert agg.ranges_m.size == 0 and abs(agg.peak_angle_deg - 30.0) <= 4.0
    assert agg.timestamp == pytest.approx(0.19)  # last frame's timestamp


def test_locate_stream_yields_per_frame():
    loc = Localizer(A, method="music")
    frames = [_Frame(_planted_grid(0.0, seed=i), t=i) for i in range(5)]
    outs = list(loc.locate_stream(frames))
    assert len(outs) == 5 and all(isinstance(o, Localization) for o in outs)
    assert [o.timestamp for o in outs] == [0, 1, 2, 3, 4]


def test_save_localization_roundtrip(tmp_path):
    loc = Localizer(A, method="music", num_ranges=40)
    frames = [_Frame(_planted_grid(20.0, seed=i), t=i * 0.1) for i in range(8)]
    agg = loc.aggregate(frames)
    out = save_localization(agg, tmp_path / "loc")
    angles = np.load(out / "angles.npy")
    ranges = np.load(out / "ranges.npy")
    heatmap = np.load(out / "heatmap.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert angles.size == 181 and ranges.size == agg.ranges_m.size
    assert heatmap.shape == agg.heatmap.shape == (40, 181)
    assert meta["num_angles"] == 181 and meta["peak_angle_deg"] == agg.peak_angle_deg


def test_localize_source_publishes_track_and_saves_map(tmp_path):
    from fixtures.SyntheticCsi import generateStream
    from wavetrace.Source import SyntheticSource
    from wavetrace.output import JsonlPublisher
    from wavetrace.Cli import localize_source

    frames, _ = generateStream(numAntennas=2, numSubcarriers=32, sampleRateHz=100.0, numFrames=20,
                               perturbationHz=1.0, perturbationDepth=0.3, cfoHz=10.0,
                               noiseStd=0.01, seed=3)
    sink = io.StringIO()
    pub = JsonlPublisher(sink, mode="localize")
    path, agg = localize_source(SyntheticSource(frames), tmp_path / "loc", num_antennas=2,
                                publisher=pub)
    lines = sink.getvalue().strip().splitlines()
    assert len(lines) == 20  # one RecognitionResult per frame, through the wire schema
    rec = json.loads(lines[0])
    assert rec["mode"] == "localize" and len(rec["keypoints"]) == 2 and rec["bbox"] is not None
    assert (path / "heatmap.npy").exists()


def test_tracker_follows_linear_motion():
    # a target sweeping 2 deg/frame at dt=0.1 -> 20 deg/s; the filter should track angle + rate
    tr = Tracker()
    states = tr.run([_meas(2.0 * i, 0.9, i * 0.1) for i in range(15)])
    last = states[-1]
    assert abs(last.angle_deg - 28.0) <= 2.0          # tracks the measured azimuth
    assert abs(last.angular_rate - 20.0) <= 6.0       # recovers the motion (deg/s)
    assert all(s.measured for s in states)


def test_tracker_gates_teleport():
    # steady at 0, then one frame "teleports" to 80 deg -> gated out, track stays put (anti-teleport)
    tr = Tracker()
    seq = [0.0, 0.0, 0.0, 0.0, 80.0, 0.0]
    states = tr.run([_meas(a, 0.9, i * 0.1) for i, a in enumerate(seq)])
    assert states[4].measured is False        # the impossible jump was rejected
    assert abs(states[4].angle_deg) < 20.0     # coasted on the motion model, did not jump to 80
    assert states[5].measured is True          # recovers once the measurement is plausible again


def test_tracker_confidence_sets_the_gain():
    # same 5-deg offset, different confidence: higher confidence -> larger Kalman gain -> moves more
    def step(conf):
        tr = Tracker(range_enabled=False)
        tr.update(_meas(0.0, 0.9, 0.0, rng=float("nan")))    # init at 0
        return tr.update(_meas(5.0, conf, 0.1, rng=float("nan"))).angle_deg
    high, low = step(0.95), step(0.05)
    assert 0.0 < low < high < 5.0              # low conf barely moves; high conf moves toward 5


def test_tracker_handles_missing_range():
    tr = Tracker(range_enabled=False)
    st = tr.run([_meas(10.0, 0.8, i * 0.1, rng=float("nan")) for i in range(5)])[-1]
    assert np.isnan(st.range_m) and st.radial_rate == 0.0 and abs(st.angle_deg - 10.0) <= 2.0


def test_single_antenna_rejected():
    with pytest.raises(ValueError):
        Localizer(1)


def test_music_needs_room_for_noise_subspace():
    with pytest.raises(ValueError):
        Localizer(2, method="music", num_sources=2)  # A - num_sources = 0, no noise subspace
