import numpy as np
import pytest

from fixtures.synthetic_csi import calibrate_quiet_baseline, generate_quiet_baseline
from wavetrace.calibration import CalibrationResult
from wavetrace.diagnostics import NodeHealthMeter

NUM_ANTENNAS = 2
NUM_SUBCARRIERS = 64
NUM_FRAMES = 200
# the dashboard turns a node amber when |gain_drift - 1| > 0.2 (web/ui/src/components/NodeHealth.tsx)
UI_GAIN_DRIFT_WARNING = 0.2


# the live stream replays the calibrated room: a different seed is a different synthetic channel
ROOM_SEED = 0


def _calibrate() -> CalibrationResult:
    result, _ = calibrate_quiet_baseline(
        num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=100.0,
        num_frames=NUM_FRAMES, seed=ROOM_SEED, baseline_packets=NUM_FRAMES)
    return result


def _feed_quiet_room(meter: NodeHealthMeter, *, node_id: int, gain: float = 1.0) -> None:
    for frame in generate_quiet_baseline(
            num_antennas=NUM_ANTENNAS, num_subcarriers=NUM_SUBCARRIERS, sample_rate_hz=100.0,
            num_frames=NUM_FRAMES, seed=ROOM_SEED):
        frame.node_id = node_id
        frame.grid[:] *= gain
        meter.observe(frame)


def _health_of(meter: NodeHealthMeter, node_id: int) -> dict:
    return next(health for health in meter.snapshot() if health["node_id"] == node_id)


def test_unchanged_room_reports_gain_drift_near_one_and_no_ui_warning():
    result = _calibrate()
    meter = NodeHealthMeter(calibration_by_node={3: result})
    _feed_quiet_room(meter, node_id=3)

    health = _health_of(meter, 3)
    assert health["gain_drift"] == pytest.approx(1.0, abs=0.002)
    assert abs(health["gain_drift"] - 1.0) <= UI_GAIN_DRIFT_WARNING


def test_gain_raised_by_thirty_percent_reports_drift_of_1_3_and_trips_the_ui_warning():
    result = _calibrate()
    meter = NodeHealthMeter(calibration_by_node={3: result})
    _feed_quiet_room(meter, node_id=3, gain=1.3)

    health = _health_of(meter, 3)
    assert health["gain_drift"] == pytest.approx(1.3, abs=0.002)
    assert abs(health["gain_drift"] - 1.0) > UI_GAIN_DRIFT_WARNING


def test_calibrated_node_reports_its_nbvi_subcarriers():
    result = _calibrate()
    meter = NodeHealthMeter(calibration_by_node={3: result})
    _feed_quiet_room(meter, node_id=3)

    subcarriers = _health_of(meter, 3)["subcarriers"]
    assert subcarriers == result.subcarriers
    assert len(subcarriers) > 0
    assert all(isinstance(subcarrier, int) for subcarrier in subcarriers)


def test_node_without_calibration_reports_no_drift_and_no_subcarriers():
    meter = NodeHealthMeter(calibration_by_node={3: _calibrate()})
    _feed_quiet_room(meter, node_id=7)

    health = _health_of(meter, 7)
    assert health["gain_drift"] is None
    assert health["subcarriers"] == []


def test_default_calibration_covers_nodes_without_their_own():
    meter = NodeHealthMeter(default_calibration=_calibrate())
    _feed_quiet_room(meter, node_id=5, gain=1.3)

    assert _health_of(meter, 5)["gain_drift"] == pytest.approx(1.3, abs=0.002)


def test_own_calibration_takes_precedence_over_the_default():
    own = _calibrate()
    doubled_default = _calibrate()
    doubled_default.baseline_mag = doubled_default.baseline_mag * 2.0
    meter = NodeHealthMeter(calibration_by_node={3: own}, default_calibration=doubled_default)
    _feed_quiet_room(meter, node_id=3)

    assert _health_of(meter, 3)["gain_drift"] == pytest.approx(1.0, abs=0.002)


def test_zero_reference_amplitude_reports_no_drift():
    zero_reference = CalibrationResult(
        reference_scale=float("nan"), subcarriers=[4, 9], num_baseline=NUM_FRAMES,
        baseline_mag=np.zeros(NUM_SUBCARRIERS, dtype=np.float32),
        baseline_diff=np.zeros(NUM_SUBCARRIERS - 1, dtype=np.complex64))
    meter = NodeHealthMeter(calibration_by_node={3: zero_reference})
    _feed_quiet_room(meter, node_id=3)

    health = _health_of(meter, 3)
    assert health["gain_drift"] is None
    assert health["subcarriers"] == [4, 9]
