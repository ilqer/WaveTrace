"""`YoloLabelerOptions.weights_path` is a plain `str | None`, separate from the `model=` constructor
parameter that takes a pre-built network. `model=` is what these tests use to bypass the
ultralytics import entirely."""

import numpy as np
import pytest

from wavetrace.groundtruth import YoloLabeler, YoloLabelerOptions, YoloSegLabeler, presenceLabelFn


class _StubYoloResult:
    """Mimics ultralytics' Results enough for _yoloToDetections to find nothing (empty boxes)."""

    boxes = None


def _stub_net(calls):
    def net(image, **kwargs):
        calls.append((image, kwargs))
        return [_StubYoloResult()]
    return net


def test_yolo_labeler_model_bypasses_weights_loading():
    """Passing model= a prebuilt callable never touches options.weights_path or ultralytics."""
    calls = []
    lab = YoloLabeler(model=_stub_net(calls), options=YoloLabelerOptions(weights_path="should-not-load"),
                       label_fn=presenceLabelFn)
    label = lab.label(np.zeros((4, 4, 3), dtype=np.uint8), timestamp=1.0)
    assert label.class_id == 0 and label.name == "absent"  # no boxes -> no person detected
    assert len(calls) == 1  # the stub net was invoked, not a real ultralytics load


def test_yolo_seg_labeler_model_bypasses_weights_loading():
    calls = []
    lab = YoloSegLabeler(model=_stub_net(calls), label_fn=presenceLabelFn)
    label = lab.label(np.zeros((4, 4, 3), dtype=np.uint8), timestamp=1.0)
    assert label.class_id == 0
    assert len(calls) == 1


def test_default_weights_are_visible_class_attributes():
    assert YoloLabeler.DEFAULT_WEIGHTS == "yolov8n.pt"
    assert YoloSegLabeler.DEFAULT_WEIGHTS == "yolov8n-seg.pt"


def test_yolo_labeler_options_weights_path_is_a_plain_string_field():
    options = YoloLabelerOptions(weights_path="custom.pt")
    assert options.weights_path == "custom.pt"
    assert not hasattr(YoloLabelerOptions(), "model")


def test_yolo_labeler_options_rejects_negative_person_class():
    with pytest.raises(ValueError, match="person_class"):
        YoloLabelerOptions(person_class=-1)
