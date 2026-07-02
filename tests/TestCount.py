"""Independent people-count pipeline tests."""

import numpy as np
import pytest

from collect_count import countName
from run_count import _expandProba
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler


def test_count_name_bins_top_as_open_ended():
    """0..N-1 exact, top level N renders as 'N+'."""
    assert [countName(c, 3) for c in range(4)] == ["0", "1", "2", "3+"]
    assert countName(0, 1) == "0"
    assert countName(1, 1) == "1+"


def test_constant_count_labeler_labels_whole_segment():
    """Lambda labeler tags every timestamp with fixed count."""
    lab = ScriptedLabeler([(0.0, 10.0, True)], label_fn=lambda raw, t: (2, "2"))
    assert lab(0.0).class_id == 2
    assert lab(9.9).class_id == 2
    assert lab(5.0).name == "2"


def test_expand_proba_maps_into_global_classes():
    """Head seeing [0,2] lands proba in cols 0 and 2. Sum kept."""
    g = _expandProba(np.array([0.7, 0.3]), col_map=[0, 2], k=4)
    assert np.allclose(g, [0.7, 0.0, 0.3, 0.0])
    assert g.sum() == pytest.approx(1.0)


def test_expand_proba_full_classes_identity():
    """Expansion is copy if head saw all classes."""
    g = _expandProba(np.array([0.1, 0.2, 0.3, 0.4]), col_map=[0, 1, 2, 3], k=4)
    assert np.allclose(g, [0.1, 0.2, 0.3, 0.4])
