"""Plants a known azimuth into a ULA and checks that MUSIC/Bartlett recover it.

Uses the same steering convention as Localizer: a clean rank-1 source must peak at the
planted angle.
"""

import io
import json

import numpy as np
import pytest

from wavetrace.domain.localize import Localizer, Localization, Tracker, SPEED_OF_LIGHT
from wavetrace.adapters.store.localization import save_localization


def _measurement(angle_deg, confidence, timestamp_s, range_m=5.0):
    """A mock Localization measurement. Only populates fields used by Tracker."""
    empty = np.empty(0)
    return Localization(timestamp=timestamp_s, angles_deg=empty, angle_spectrum=empty, ranges_m=empty,
                        range_profile=empty, heatmap=empty, peak_angle_deg=angle_deg, peak_range_m=range_m,
                        x_m=0.0, y_m=0.0, confidence=confidence)

ANTENNA_COUNT = 2          # 2-antenna ESP32 / Pi
SUBCARRIER_COUNT = 64
SPACING = 0.5  # lambda/2


def _planted_grid(angle_deg, *, antenna_count=ANTENNA_COUNT, subcarrier_count=SUBCARRIER_COUNT,
                  spacing=SPACING, noise=0.01, seed=0):
    """(antenna_count, subcarrier_count) CSI for a single plane wave from `angle_deg` (0 = broadside).

    Generates a(theta) * per-subcarrier gain."""
    rng = np.random.default_rng(seed)
    antenna_index = np.arange(antenna_count)
    steering_vector = np.exp(1j * 2 * np.pi * spacing * antenna_index * np.sin(np.deg2rad(angle_deg)))
    gain = rng.standard_normal(subcarrier_count) + 1j * rng.standard_normal(subcarrier_count)
    csi = np.outer(steering_vector, gain)
    csi += noise * (rng.standard_normal(csi.shape) + 1j * rng.standard_normal(csi.shape))
    return csi.astype(np.complex64)


class _Frame:
    """Mock CsiFrame providing only grid and timestamp for stream/aggregate tests."""
    def __init__(self, grid, timestamp_s):
        self.grid = grid
        self.timestamp = timestamp_s


@pytest.mark.parametrize("angle", [0.0, 20.0, -30.0, 45.0])
def test_music_recovers_planted_angle(angle):
    localizer = Localizer(ANTENNA_COUNT, spacing=SPACING, method="music")
    estimated_angle_deg = localizer.locate(_planted_grid(angle, seed=1)).peak_angle_deg
    assert abs(estimated_angle_deg - angle) <= 3.0  # 1-deg grid + clean rank-1 source -> within a few degrees


@pytest.mark.parametrize("angle", [0.0, 25.0, -40.0])
def test_bartlett_recovers_planted_angle(angle):
    localizer = Localizer(ANTENNA_COUNT, spacing=SPACING, method="bartlett")
    estimated_angle_deg = localizer.locate(_planted_grid(angle, seed=2)).peak_angle_deg
    assert abs(estimated_angle_deg - angle) <= 6.0  # beamformer is lower-resolution than MUSIC


def test_spectrum_normalized_and_confident():
    localizer = Localizer(ANTENNA_COUNT, method="music")
    out = localizer.locate(_planted_grid(15.0, seed=3))
    assert out.angle_spectrum.min() >= 0.0 and out.angle_spectrum.max() == pytest.approx(1.0)
    assert 0.0 < out.confidence <= 1.0  # a single sharp arrival -> peaked spectrum


def test_locate_per_frame_is_azimuth_spectrum():
    # A single frame cannot resolve 2D range. Heatmap becomes a (1, G) azimuth spectrum.
    localizer = Localizer(ANTENNA_COUNT, method="music", num_angles=181, max_range_m=15.0)
    out = localizer.locate(_planted_grid(20.0, seed=4))
    assert out.heatmap.shape == (1, out.angles_deg.size)
    assert out.ranges_m.size > 0  # a coarse 1-D range profile still places (x, y)
    peak_angle_rad = np.deg2rad(out.peak_angle_deg)
    assert out.x_m == pytest.approx(out.peak_range_m * np.sin(peak_angle_rad), rel=1e-5, abs=1e-9)
    assert out.y_m == pytest.approx(out.peak_range_m * np.cos(peak_angle_rad), rel=1e-5, abs=1e-9)


def test_aggregate_is_joint_2d_room_map():
    # With range enabled and multiple frames, the aggregate heatmap is a joint 2D MUSIC map (range × angle).
    localizer = Localizer(ANTENNA_COUNT, method="music", num_ranges=48, max_range_m=12.0)
    frames = [_Frame(_planted_grid(25.0, seed=i), timestamp_s=i) for i in range(12)]
    aggregate = localizer.aggregate(frames)
    assert aggregate.heatmap.shape == (48, 181)
    assert abs(aggregate.peak_angle_deg - 25.0) <= 4.0
    assert 0.0 <= aggregate.peak_range_m <= 12.0


def test_joint_2d_resolves_planted_range():
    # Inflated bandwidth to make delay phase ramp observable.
    # Real WiFi BW is too narrow for room-scale range resolution.
    subcarrier_spacing_hz, subcarrier_count = 5e6, 32

    def grid(angle, range_m, seed):
        rng = np.random.default_rng(seed)
        steering_vector = np.exp(
            1j * 2 * np.pi * SPACING * np.arange(ANTENNA_COUNT) * np.sin(np.deg2rad(angle)))
        delay_ramp = np.exp(-1j * 2 * np.pi * subcarrier_spacing_hz * np.arange(subcarrier_count)
                            * (range_m / SPEED_OF_LIGHT))
        frame_gain = rng.standard_normal() + 1j * rng.standard_normal()  # per-frame gain, decorrelates snaps
        csi = frame_gain * np.outer(steering_vector, delay_ramp)
        csi += 0.01 * (rng.standard_normal((ANTENNA_COUNT, subcarrier_count))
                       + 1j * rng.standard_normal((ANTENNA_COUNT, subcarrier_count)))
        return csi.astype(np.complex64)

    localizer = Localizer(ANTENNA_COUNT, method="music", subcarrier_spacing_hz=subcarrier_spacing_hz,
                          max_range_m=20.0, num_ranges=80)
    frames = [_Frame(grid(15.0, 6.0, i), timestamp_s=i) for i in range(16)]
    aggregate = localizer.aggregate(frames)
    assert abs(aggregate.peak_angle_deg - 15.0) <= 5.0
    assert abs(aggregate.peak_range_m - 6.0) <= 3.0


def test_no_range_mode_is_azimuth_only():
    localizer = Localizer(ANTENNA_COUNT, method="music", range_enabled=False)
    out = localizer.locate(_planted_grid(10.0, seed=5))
    assert out.ranges_m.size == 0 and np.isnan(out.peak_range_m)
    assert out.heatmap.shape == (1, out.angles_deg.size)


def test_aggregate_azimuth_fallback_is_steady():
    # With range disabled, aggregate averages per-frame 1D AoA spectra. Robust to noise and gain variations.
    localizer = Localizer(ANTENNA_COUNT, method="music", range_enabled=False)
    frames = [_Frame(_planted_grid(30.0, seed=10 + i, noise=0.3), timestamp_s=i * 0.01) for i in range(20)]
    aggregate = localizer.aggregate(frames)
    assert aggregate.ranges_m.size == 0 and abs(aggregate.peak_angle_deg - 30.0) <= 4.0
    assert aggregate.timestamp == pytest.approx(0.19)  # last frame's timestamp


def test_locate_stream_yields_per_frame():
    localizer = Localizer(ANTENNA_COUNT, method="music")
    frames = [_Frame(_planted_grid(0.0, seed=i), timestamp_s=i) for i in range(5)]
    localizations = list(localizer.locate_stream(frames))
    assert len(localizations) == 5 and all(isinstance(item, Localization) for item in localizations)
    assert [item.timestamp for item in localizations] == [0, 1, 2, 3, 4]


def test_save_localization_roundtrip(tmp_path):
    localizer = Localizer(ANTENNA_COUNT, method="music", num_ranges=40)
    frames = [_Frame(_planted_grid(20.0, seed=i), timestamp_s=i * 0.1) for i in range(8)]
    aggregate = localizer.aggregate(frames)
    out = save_localization(aggregate, tmp_path / "loc")
    angles = np.load(out / "angles.npy")
    ranges = np.load(out / "ranges.npy")
    heatmap = np.load(out / "heatmap.npy")
    meta = json.loads((out / "meta.json").read_text())
    assert angles.size == 181 and ranges.size == aggregate.ranges_m.size
    assert heatmap.shape == aggregate.heatmap.shape == (40, 181)
    assert meta["num_angles"] == 181 and meta["peak_angle_deg"] == aggregate.peak_angle_deg


def test_localize_source_publishes_track_and_saves_map(tmp_path):
    from wavetrace.synthetic import generate_stream
    from wavetrace.adapters.source import SyntheticSource
    from wavetrace.output import JsonlPublisher
    from wavetrace.pipeline import localize_source

    frames, _ = generate_stream(num_antennas=2, num_subcarriers=32, sample_rate_hz=100.0, num_frames=20,
                               perturbation_hz=1.0, perturbation_depth=0.3, cfo_hz=10.0,
                               noise_std=0.01, seed=3)
    sink = io.StringIO()
    publisher = JsonlPublisher(sink, mode="localize")
    path, aggregate = localize_source(SyntheticSource(frames), tmp_path / "loc", num_antennas=2,
                                 publisher=publisher)
    lines = sink.getvalue().strip().splitlines()
    assert len(lines) == 20  # one RecognitionResult per frame, through the wire schema
    record = json.loads(lines[0])
    assert record["mode"] == "localize" and len(record["keypoints"]) == 2 and record["bbox"] is not None
    assert (path / "heatmap.npy").exists()


def test_tracker_follows_linear_motion():
    # Target sweeping 2 deg/frame at dt=0.1s (20 deg/s). Filter must track both angle and angular rate.
    tracker = Tracker()
    states = tracker.run([_measurement(2.0 * i, 0.9, i * 0.1) for i in range(15)])
    last = states[-1]
    assert abs(last.angle_deg - 28.0) <= 2.0
    assert abs(last.angular_rate - 20.0) <= 6.0       # recovers the motion (deg/s)
    assert all(state.measured for state in states)


def test_tracker_gates_teleport():
    # Target steady at 0, then teleports to 80 deg in one frame.
    # Tracker should gate out the outlier and coast.
    tracker = Tracker()
    angle_sequence_deg = [0.0, 0.0, 0.0, 0.0, 80.0, 0.0]
    states = tracker.run(
        [_measurement(angle_deg, 0.9, i * 0.1) for i, angle_deg in enumerate(angle_sequence_deg)])
    assert states[4].measured is False
    assert abs(states[4].angle_deg) < 20.0     # coasted on the motion model, did not jump to 80
    assert states[5].measured is True          # recovers once the measurement is plausible again


def test_tracker_confidence_sets_the_gain():
    # Higher confidence raises the Kalman gain, so the tracker moves further on the same offset.
    def step(confidence):
        tracker = Tracker(range_enabled=False)
        tracker.update(_measurement(0.0, 0.9, 0.0, range_m=float("nan")))
        return tracker.update(_measurement(5.0, confidence, 0.1, range_m=float("nan"))).angle_deg
    high, low = step(0.95), step(0.05)
    assert 0.0 < low < high < 5.0              # low confidence barely moves; high confidence moves toward 5


def test_tracker_handles_missing_range():
    tracker = Tracker(range_enabled=False)
    state = tracker.run([_measurement(10.0, 0.8, i * 0.1, range_m=float("nan")) for i in range(5)])[-1]
    assert np.isnan(state.range_m) and state.radial_rate == 0.0 and abs(state.angle_deg - 10.0) <= 2.0


def test_single_antenna_rejected():
    with pytest.raises(ValueError):
        Localizer(1)


def test_music_needs_room_for_noise_subspace():
    with pytest.raises(ValueError):
        Localizer(2, method="music", num_sources=2)  # ANTENNA_COUNT - num_sources = 0, no noise subspace
