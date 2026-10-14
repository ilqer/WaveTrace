import numpy as np
import pytest

from collect_count import count_name
from run_count import _expand_proba
from wavetrace.domain.groundtruth.labeler import ScriptedLabeler


def test_count_name_bins_top_as_open_ended():
    """0..N-1 exact, top level N renders as 'N+'."""
    assert [count_name(count, 3) for count in range(4)] == ["0", "1", "2", "3+"]
    assert count_name(0, 1) == "0"
    assert count_name(1, 1) == "1+"


def test_constant_count_labeler_labels_whole_segment():
    labeler = ScriptedLabeler([(0.0, 10.0, True)], label_function=lambda raw, timestamp_s: (2, "2"))
    assert labeler(0.0).class_id == 2
    assert labeler(9.9).class_id == 2
    assert labeler(5.0).name == "2"


def test_expand_proba_maps_into_global_classes():
    global_probabilities = _expand_proba(np.array([0.7, 0.3]), col_map=[0, 2], k=4)
    assert np.allclose(global_probabilities, [0.7, 0.0, 0.3, 0.0])
    assert global_probabilities.sum() == pytest.approx(1.0)


def test_expand_proba_full_classes_identity():
    global_probabilities = _expand_proba(np.array([0.1, 0.2, 0.3, 0.4]), col_map=[0, 1, 2, 3], k=4)
    assert np.allclose(global_probabilities, [0.1, 0.2, 0.3, 0.4])
