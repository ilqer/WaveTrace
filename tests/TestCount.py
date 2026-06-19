"""Independent people-count pipeline: count-label formatting, constant-count labeler, proba expansion."""

import numpy as np
import pytest

from collect_count import count_name
from run_count import _expand_proba
from wavetrace.groundtruth.CameraLabeler import ScriptedLabeler


def test_count_name_bins_top_as_open_ended():
    """0..N-1 are exact; the top level N renders as 'N+'."""
    assert [count_name(c, 3) for c in range(4)] == ["0", "1", "2", "3+"]
    assert count_name(0, 1) == "0"
    assert count_name(1, 1) == "1+"


def test_constant_count_labeler_labels_whole_segment():
    """The lambda labeler used in collect_count tags every timestamp with the fixed count."""
    lab = ScriptedLabeler([(0.0, 10.0, True)], label_fn=lambda raw, t: (2, "2"))
    assert lab(0.0).class_id == 2
    assert lab(9.9).class_id == 2
    assert lab(5.0).name == "2"


def test_expand_proba_maps_into_global_classes():
    """A head that saw classes [0,2] (global space [0,1,2,3]) lands its proba in cols 0 and 2; sum kept."""
    g = _expand_proba(np.array([0.7, 0.3]), col_map=[0, 2], k=4)
    assert np.allclose(g, [0.7, 0.0, 0.3, 0.0])
    assert g.sum() == pytest.approx(1.0)


def test_expand_proba_full_classes_identity():
    """When the head saw every global class in order, expansion is a copy."""
    g = _expand_proba(np.array([0.1, 0.2, 0.3, 0.4]), col_map=[0, 1, 2, 3], k=4)
    assert np.allclose(g, [0.1, 0.2, 0.3, 0.4])
