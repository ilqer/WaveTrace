"""VisionLabeler tests: camera-supervised labeling.

A STUB detector exercises the detection-to-label policy (presence, bbox, keypoints, weapon)
without model dependency.
"""

import numpy as np
import pytest

from wavetrace.groundtruth import (
    Detection,
    LabelerOptions,
    VisionLabeler,
    presence_label_function,
    weapon_label_function,
)

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)  # Dummy frame. Stub detector ignores content.


def _person_detector(image):
    return [Detection(cls=0, confidence=0.9, bbox_xywhn=(0.4, 0.3, 0.2, 0.5), keypoints=[0.5, 0.2, 0.5, 0.5])]


def _empty_detector(image):
    return []


def test_person_detection_labels_present_with_bbox():
    labeler = VisionLabeler(_person_detector, label_function=presence_label_function)
    label = labeler.label(IMAGE, 1.0)
    assert label.class_id == 1 and label.name == "present"
    assert list(label.bbox) == pytest.approx([0.4, 0.3, 0.2, 0.5])  # Label stores float32.
    assert list(label.keypoints) == pytest.approx([0.5, 0.2, 0.5, 0.5])


def test_no_detection_labels_absent():
    labeler = VisionLabeler(_empty_detector, label_function=presence_label_function)
    label = labeler.label(IMAGE, 2.0)
    assert label.class_id == 0 and label.name == "absent" and label.bbox is None


def test_low_confidence_is_filtered_out():
    detector = lambda image: [Detection(0, 0.10, (0.1, 0.1, 0.1, 0.1))]
    labeler = VisionLabeler(detector, confidence_threshold=0.35, label_function=presence_label_function)
    assert labeler.label(IMAGE, 0.0).class_id == 0


def test_weapon_class_flags_weapon():
    # Person (0) and knife (43). weapon_classes flags 43.
    detector = lambda image: [
        Detection(0, 0.8, (0.4, 0.3, 0.2, 0.5)),
        Detection(43, 0.7, (0.45, 0.5, 0.05, 0.1)),
    ]
    labeler = VisionLabeler(detector, options=LabelerOptions(weapon_classes=(43,)),
                            label_function=weapon_label_function)
    label = labeler.label(IMAGE, 3.0)
    assert label.class_id == 1 and label.name == "weapon"


def test_best_person_chosen_by_confidence():
    detector = lambda image: [
        Detection(0, 0.5, (0.1, 0.1, 0.1, 0.1)),
        Detection(0, 0.95, (0.4, 0.3, 0.2, 0.5)),
    ]
    labeler = VisionLabeler(detector, label_function=presence_label_function)
    assert list(labeler.label(IMAGE, 0.0).bbox) == pytest.approx([0.4, 0.3, 0.2, 0.5])


def test_label_stream_sorted_by_time():
    labeler = VisionLabeler(_person_detector, label_function=presence_label_function)
    observations = [{"t": 2.0}, {"t": 0.5}, {"t": 1.0}]
    out = labeler.label_stream(observations)
    assert [label.timestamp for label in out] == [0.5, 1.0, 2.0]


def test_non_callable_detector_rejected():
    with pytest.raises(ValueError):
        VisionLabeler(detector=object())
