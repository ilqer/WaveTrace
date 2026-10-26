import json

import numpy as np

from wavetrace.adapters.store.calibration import load_calibration, save_calibration
from wavetrace.domain.calibration import CalibrationResult


def _result(subcarrier_width):
    return CalibrationResult(reference_scale=1.5, subcarriers=[1, 5], num_baseline=200,
                             baseline_mag=np.ones(8, dtype=np.float32),
                             baseline_diff=np.ones(7, dtype=np.complex64), image_subcarriers=[1, 2, 5],
                             subcarrier_width=subcarrier_width)


def test_the_capture_width_round_trips_through_meta_json(tmp_path):
    save_calibration(_result(subcarrier_width=128), tmp_path)
    assert json.loads((tmp_path / "meta.json").read_text())["subcarrier_width"] == 128
    assert load_calibration(tmp_path)[0].subcarrier_width == 128


def test_a_calibration_without_a_recorded_width_writes_no_width_key_and_loads_as_none(tmp_path):
    save_calibration(_result(subcarrier_width=None), tmp_path)
    assert "subcarrier_width" not in json.loads((tmp_path / "meta.json").read_text())
    assert load_calibration(tmp_path)[0].subcarrier_width is None


def test_a_meta_json_saved_before_the_width_existed_still_loads(tmp_path):
    save_calibration(_result(subcarrier_width=None), tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta.pop("image_subcarriers")
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    result, gain_lock = load_calibration(tmp_path)
    assert (result.subcarrier_width, result.subcarriers, result.image_subcarriers) == (None, [1, 5], [1, 5])
    assert gain_lock is not None
