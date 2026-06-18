"""VisionLabeler tests — the camera-supervised labeling seam, now wired. A STUB detector exercises
the detection->Label policy (present/bbox/keypoints/weapon) with no model dependency; the real
YoloLabeler is the same policy over an ultralytics model (not exercised here — optional dep)."""

import numpy as np
import pytest

from wavetrace.groundtruth import (
    Detection,
    VisionLabeler,
    presence_label_fn,
    weapon_label_fn,
)

IMG = np.zeros((8, 8, 3), dtype=np.uint8)  # a dummy frame; the stub detector ignores its content


def _person_detector(image):
    return [Detection(cls=0, conf=0.9, bbox_xywhn=(0.4, 0.3, 0.2, 0.5), keypoints=[0.5, 0.2, 0.5, 0.5])]


def _empty_detector(image):
    return []


def test_person_detection_labels_present_with_bbox():
    lab = VisionLabeler(_person_detector, label_fn=presence_label_fn)
    l = lab.label(IMG, 1.0)
    assert l.class_id == 1 and l.name == "present"
    assert list(l.bbox) == pytest.approx([0.4, 0.3, 0.2, 0.5])  # native Label stores float32
    assert list(l.keypoints) == pytest.approx([0.5, 0.2, 0.5, 0.5])


def test_no_detection_labels_absent():
    lab = VisionLabeler(_empty_detector, label_fn=presence_label_fn)
    l = lab.label(IMG, 2.0)
    assert l.class_id == 0 and l.name == "absent" and l.bbox is None


def test_low_confidence_is_filtered_out():
    detector = lambda img: [Detection(0, 0.10, (0.1, 0.1, 0.1, 0.1))]
    lab = VisionLabeler(detector, conf=0.35, label_fn=presence_label_fn)
    assert lab.label(IMG, 0.0).class_id == 0  # below threshold -> not present


def test_weapon_class_flags_weapon():
    # a person (0) and a knife (43) in frame; weapon_classes marks 43 as a weapon
    detector = lambda img: [
        Detection(0, 0.8, (0.4, 0.3, 0.2, 0.5)),
        Detection(43, 0.7, (0.45, 0.5, 0.05, 0.1)),
    ]
    lab = VisionLabeler(detector, weapon_classes=(43,), label_fn=weapon_label_fn)
    l = lab.label(IMG, 3.0)
    assert l.class_id == 1 and l.name == "weapon"


def test_best_person_chosen_by_confidence():
    detector = lambda img: [
        Detection(0, 0.5, (0.1, 0.1, 0.1, 0.1)),
        Detection(0, 0.95, (0.4, 0.3, 0.2, 0.5)),
    ]
    lab = VisionLabeler(detector, label_fn=presence_label_fn)
    assert list(lab.label(IMG, 0.0).bbox) == pytest.approx([0.4, 0.3, 0.2, 0.5])  # higher-conf box


def test_label_stream_sorted_by_time():
    lab = VisionLabeler(_person_detector, label_fn=presence_label_fn)
    obs = [{"t": 2.0}, {"t": 0.5}, {"t": 1.0}]
    out = lab.label_stream(obs)
    assert [l.timestamp for l in out] == [0.5, 1.0, 2.0]


def test_non_callable_detector_rejected():
    with pytest.raises(ValueError):
        VisionLabeler(detector=object())
