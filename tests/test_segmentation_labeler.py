"""A STUB segmenter supplies the pixel masks, so the policy (presence/weapon, mask-overlap
gate, grid occupancy) is tested without model dependencies. YoloSegLabeler uses the same
policy.
"""

import numpy as np
import pytest

from wavetrace.groundtruth import (
    LabelerOptions,
    Segment,
    SegmentationLabeler,
    presence_label_function,
    weapon_label_function,
)

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)  # Dummy frame. Stub segmenter ignores content.
MASK_HEIGHT = MASK_WIDTH = 16


def _rect_mask(left, top, right, bottom):
    mask = np.zeros((MASK_HEIGHT, MASK_WIDTH), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


# Person fills left half. Small weapon mask inside, one outside.
PERSON = _rect_mask(0, 0, 8, 16)
WEAPON_IN = _rect_mask(2, 6, 5, 9)
WEAPON_OUT = _rect_mask(12, 6, 15, 9)


def _segmenter(*segments):
    return lambda image: list(segments)


def test_person_sets_present_bbox_and_mask_grid():
    labeler = SegmentationLabeler(_segmenter(Segment(0, 0.9, PERSON)), grid=8,
                                  label_function=presence_label_function)
    label = labeler.label(IMAGE, 1.0)
    assert label.class_id == 1 and label.name == "present"
    assert label.mask_grid == 8 and len(label.mask) == 64
    # Person fills left half: left columns saturate to 1, right columns are 0.
    occupancy_grid = np.asarray(label.mask).reshape(8, 8)
    assert occupancy_grid[:, :4] == pytest.approx(np.ones((8, 4)))
    assert occupancy_grid[:, 4:] == pytest.approx(np.zeros((8, 4)))
    # Tight bbox from mask: left half yields centre x=0.25, full height.
    assert label.bbox[0] == pytest.approx(0.25) and label.bbox[3] == pytest.approx(1.0)


def test_no_segment_is_absent_with_no_mask():
    labeler = SegmentationLabeler(_segmenter(), label_function=presence_label_function)
    label = labeler.label(IMAGE, 0.0)
    assert label.class_id == 0 and label.name == "absent"
    assert label.bbox is None and list(label.mask) == [] and label.mask_grid == 0


def test_weapon_inside_person_passes_gate():
    segmenter = _segmenter(Segment(0, 0.9, PERSON), Segment(43, 0.8, WEAPON_IN))
    labeler = SegmentationLabeler(segmenter, options=LabelerOptions(weapon_classes=(43,)), overlap_min=0.5,
                                  label_function=weapon_label_function)
    label = labeler.label(IMAGE, 2.0)
    assert label.class_id == 1 and label.name == "weapon"
    # Supervised on weapon mask, not person mask. Grid mass is localized.
    assert 0.0 < np.asarray(label.mask).sum() < np.asarray(_grid_sum(PERSON, labeler))


def test_weapon_outside_person_rejected_by_gate():
    segmenter = _segmenter(Segment(0, 0.9, PERSON), Segment(43, 0.95, WEAPON_OUT))
    labeler = SegmentationLabeler(segmenter, options=LabelerOptions(weapon_classes=(43,)), overlap_min=0.5,
                                  label_function=weapon_label_function)
    label = labeler.label(IMAGE, 3.0)
    assert label.class_id == 0 and label.name == "no_weapon"  # High confidence, but fails mask-overlap gate.


def test_weapon_without_person_rejected():
    labeler = SegmentationLabeler(_segmenter(Segment(43, 0.99, WEAPON_OUT)),
                                  options=LabelerOptions(weapon_classes=(43,)),
                                  label_function=weapon_label_function)
    assert labeler.label(IMAGE, 0.0).class_id == 0  # No person to gate against.


def test_low_confidence_segment_filtered():
    labeler = SegmentationLabeler(_segmenter(Segment(0, 0.10, PERSON)), confidence_threshold=0.5,
                                  label_function=presence_label_function)
    assert labeler.label(IMAGE, 0.0).class_id == 0


def test_best_person_chosen_by_confidence():
    small = _rect_mask(0, 0, 2, 2)
    segmenter = _segmenter(Segment(0, 0.5, small), Segment(0, 0.95, PERSON))
    labeler = SegmentationLabeler(segmenter, grid=8, label_function=presence_label_function)
    label = labeler.label(IMAGE, 0.0)
    assert label.bbox[3] == pytest.approx(1.0)  # Selects full-height higher-confidence person.


def test_non_callable_segmenter_rejected():
    with pytest.raises(ValueError):
        SegmentationLabeler(segmenter=object())


def _grid_sum(mask, labeler):
    """Occupancy-grid mass of a mask at the labeler's grid resolution."""
    from wavetrace.groundtruth.camera_labeler import _mask_to_grid
    return sum(_mask_to_grid(mask, labeler._grid))
