from types import SimpleNamespace

import pytest

from wavetrace.recognition.serving import NodeServer


def make_node_server():
    return NodeServer(calibration=None, gain_lock=None, should_use_intercarrier=False,
                      pick_input=lambda features, image, intercarrier: features,
                      session=SimpleNamespace(), config=SimpleNamespace())


def test_defaults_keep_a_node_voting_with_no_class_info():
    node = make_node_server()
    assert (node.weight, node.accuracy) == (1.0, 1.0)
    assert node.positive_class_index == -1 and node.intercarrier_baseline is None


def test_lists_are_not_shared_between_nodes():
    first, second = make_node_server(), make_node_server()
    first.classes.append(1)
    assert second.classes == []


def test_weight_is_reassignable_but_a_misspelt_field_is_rejected():
    node = make_node_server()
    node.weight = 0.25
    assert node.weight == 0.25
    with pytest.raises(AttributeError):
        node.wieght = 0.5


def test_trained_width_is_the_capture_width_in_the_head_contract():
    node = make_node_server()
    node.session = SimpleNamespace(head=SimpleNamespace(contract=SimpleNamespace(subcarrier_width=192)))
    assert node.trained_subcarrier_width == 192


def test_calibrated_width_and_target_rate_come_from_the_calibration_and_the_head_contract():
    node = make_node_server()
    node.calibration = SimpleNamespace(subcarrier_width=128)
    node.session = SimpleNamespace(head=SimpleNamespace(contract=SimpleNamespace(target_sample_rate_hz=50.0)))
    assert node.calibrated_subcarrier_width == 128
    assert node.target_sample_rate_hz == 50.0
