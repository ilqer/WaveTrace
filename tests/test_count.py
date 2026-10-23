from wavetrace.domain.groundtruth.labeler import ScriptedLabeler
from wavetrace.recognition import count_name


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

