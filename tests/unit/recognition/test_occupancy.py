r"""Pins OccupancyGrid/_GridKalman/HeatmapTrack's behaviour and their options-object validation:
wavetrace/recognition/__init__.py doesn't export any of the three, and grep found no caller anywhere
in the codebase (confirmed: `grep -rn "OccupancyGrid(\|HeatmapTrack(\|_GridKalman(" --include='*.py'
. | grep -v .venv` returns hits only inside Occupancy.py itself). This module is otherwise dead code
-- see REFACTOR_PLAN.md §5 backlog item D5 ("wire it into the heatmap serving path or delete it");
these tests exist so that decision can be made safely, without also having to characterize
undocumented behaviour first."""

import numpy as np
import pytest

from wavetrace.recognition.Occupancy import (
    HeatmapTrack,
    KalmanOptions,
    OccupancyGrid,
    OccupancyOptions,
    _GridKalman,
)


# ----- OccupancyGrid --------------------------------------------------------------------------------

def test_occupancy_grid_defaults_match_the_previous_constructor_literals():
    grid = OccupancyGrid()
    assert grid.g == 16 and grid.decay == 0.1 and grid.blur == 0.5 and grid.floor == 0.2
    assert grid.state.shape == (16, 16)


def test_occupancy_grid_honors_custom_options():
    grid = OccupancyGrid(OccupancyOptions(grid=4, decay=0.0, blur=0.0, measurement_weight_floor=1.0))
    assert grid.state.shape == (4, 4)
    measurement = np.zeros((4, 4), dtype=np.float32)
    measurement[1, 2] = 1.0
    fused = grid.update(measurement, confidence=1.0)
    assert fused[1, 2] == pytest.approx(1.0)  # floor=1.0, decay=0, blur=0 -> pure measurement passthrough
    assert grid.peak() == (1, 2, pytest.approx(1.0))


def test_occupancy_options_rejects_invalid_ranges():
    with pytest.raises(ValueError, match="grid"):
        OccupancyOptions(grid=0)
    with pytest.raises(ValueError, match="decay"):
        OccupancyOptions(decay=1.5)
    with pytest.raises(ValueError, match="blur"):
        OccupancyOptions(blur=-0.1)
    with pytest.raises(ValueError, match="measurement_weight_floor"):
        OccupancyOptions(measurement_weight_floor=1.5)


# ----- _GridKalman -----------------------------------------------------------------------------------

def test_grid_kalman_defaults_match_the_previous_constructor_literals():
    kf = _GridKalman()
    assert kf._qa == pytest.approx(2.0 ** 2)
    assert kf._rs == pytest.approx(1.5 ** 2)
    assert kf._gate == pytest.approx(9.0)


def test_grid_kalman_honors_custom_options():
    kf = _GridKalman(KalmanOptions(acceleration_std=5.0, measurement_std=0.25, gate=50.0))
    assert kf._qa == pytest.approx(5.0 ** 2)
    assert kf._rs == pytest.approx(0.25 ** 2)
    assert kf._gate == pytest.approx(50.0)


def test_grid_kalman_options_rejects_non_positive_fields():
    with pytest.raises(ValueError, match="acceleration_std"):
        KalmanOptions(acceleration_std=0)
    with pytest.raises(ValueError, match="measurement_std"):
        KalmanOptions(measurement_std=0)
    with pytest.raises(ValueError, match="gate"):
        KalmanOptions(gate=0)


def test_grid_kalman_first_update_passes_through_then_smooths():
    kf = _GridKalman()
    r0, c0, measured0 = kf.update(1.0, 2.0, t=0.0)
    assert (r0, c0, measured0) == (1.0, 2.0, True)  # first sample seeds state directly
    r1, c1, measured1 = kf.update(1.0, 2.0, t=1.0, confidence=1.0)
    assert measured1 is True
    assert r1 == pytest.approx(1.0, abs=0.5) and c1 == pytest.approx(2.0, abs=0.5)


# ----- HeatmapTrack ------------------------------------------------------------------------------------

def test_heatmap_track_defaults_and_update_shape():
    track = HeatmapTrack()
    measurement = np.zeros((16, 16), dtype=np.float32)
    measurement[8, 8] = 1.0
    out = track.update(measurement, confidence=1.0, t=0.0)
    assert set(out.keys()) == {"grid", "peak", "track"}
    assert len(out["grid"]) == 16 * 16
    assert set(out["track"].keys()) == {"x", "y", "measured"}


def test_heatmap_track_reset_honors_its_own_kalman_options_not_defaults():
    """Guards against a real regression risk: reset() must reconstruct `_GridKalman` from the
    options this instance was actually built with, not from `KalmanOptions()` defaults — otherwise
    a caller's custom tuning silently reverts to the default on every reset()."""
    custom_kalman = KalmanOptions(acceleration_std=7.0, measurement_std=0.5, gate=42.0)
    track = HeatmapTrack(occupancy=OccupancyOptions(grid=4), kalman=custom_kalman)
    assert track._kalman._qa == pytest.approx(7.0 ** 2)

    measurement = np.zeros((4, 4), dtype=np.float32)
    measurement[0, 0] = 1.0
    track.update(measurement, confidence=1.0, t=0.0)
    track.update(measurement, confidence=1.0, t=1.0)  # perturb the Kalman's internal state

    track.reset()

    assert track.occ.state.sum() == 0.0                    # occupancy grid cleared
    assert track._kalman._x is None                        # kalman state cleared (fresh instance)
    # the custom tuning survives the reset -- reconstructed from the stored options, not defaults
    assert track._kalman._qa == pytest.approx(7.0 ** 2)
    assert track._kalman._rs == pytest.approx(0.5 ** 2)
    assert track._kalman._gate == pytest.approx(42.0)
