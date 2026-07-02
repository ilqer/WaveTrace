"""SegmentationLabeler tests: pretrained-segmentation teacher for the CSI heatmap head.

A STUB segmenter provides pixel masks to test policy (presence/weapon, mask-overlap gate, grid occupancy)
without model dependencies. YoloSegLabeler uses identical policy.
"""

import numpy as np
import pytest

from wavetrace.groundtruth import (
    Segment,
    SegmentationLabeler,
    presenceLabelFn,
    weaponLabelFn,
)

IMG = np.zeros((8, 8, 3), dtype=np.uint8)  # Dummy frame. Stub segmenter ignores content.
H = W = 16


def _rect_mask(x0, y0, x1, y1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# Person fills left half. Small weapon mask inside, one outside.
PERSON = _rect_mask(0, 0, 8, 16)
WEAPON_IN = _rect_mask(2, 6, 5, 9)
WEAPON_OUT = _rect_mask(12, 6, 15, 9)


def _segmenter(*segments):
    return lambda image: list(segments)


def test_person_sets_present_bbox_and_mask_grid():
    lab = SegmentationLabeler(_segmenter(Segment(0, 0.9, PERSON)), grid=8, label_fn=presenceLabelFn)
    l = lab.label(IMG, 1.0)
    assert l.class_id == 1 and l.name == "present"
    assert l.mask_grid == 8 and len(l.mask) == 64
    # Person fills left half: left columns saturate to 1, right columns are 0.
    g = np.asarray(l.mask).reshape(8, 8)
    assert g[:, :4] == pytest.approx(np.ones((8, 4)))
    assert g[:, 4:] == pytest.approx(np.zeros((8, 4)))
    # Tight bbox from mask: left half yields centre x=0.25, full height.
    assert l.bbox[0] == pytest.approx(0.25) and l.bbox[3] == pytest.approx(1.0)


def test_no_segment_is_absent_with_no_mask():
    lab = SegmentationLabeler(_segmenter(), label_fn=presenceLabelFn)
    l = lab.label(IMG, 0.0)
    assert l.class_id == 0 and l.name == "absent"
    assert l.bbox is None and list(l.mask) == [] and l.mask_grid == 0


def test_weapon_inside_person_passes_gate():
    seg = _segmenter(Segment(0, 0.9, PERSON), Segment(43, 0.8, WEAPON_IN))
    lab = SegmentationLabeler(seg, weapon_classes=(43,), overlap_min=0.5, label_fn=weaponLabelFn)
    l = lab.label(IMG, 2.0)
    assert l.class_id == 1 and l.name == "weapon"
    # Supervised on weapon mask, not person mask. Grid mass is localized.
    assert 0.0 < np.asarray(l.mask).sum() < np.asarray(_grid_sum(PERSON, lab))


def test_weapon_outside_person_rejected_by_gate():
    seg = _segmenter(Segment(0, 0.9, PERSON), Segment(43, 0.95, WEAPON_OUT))
    lab = SegmentationLabeler(seg, weapon_classes=(43,), overlap_min=0.5, label_fn=weaponLabelFn)
    l = lab.label(IMG, 3.0)
    assert l.class_id == 0 and l.name == "no_weapon"  # High conf, but fails mask-overlap gate.


def test_weapon_without_person_rejected():
    lab = SegmentationLabeler(_segmenter(Segment(43, 0.99, WEAPON_OUT)), weapon_classes=(43,),
                              label_fn=weaponLabelFn)
    assert lab.label(IMG, 0.0).class_id == 0  # No person to gate against.


def test_low_confidence_segment_filtered():
    lab = SegmentationLabeler(_segmenter(Segment(0, 0.10, PERSON)), conf=0.5, label_fn=presenceLabelFn)
    assert lab.label(IMG, 0.0).class_id == 0


def test_best_person_chosen_by_confidence():
    small = _rect_mask(0, 0, 2, 2)
    seg = _segmenter(Segment(0, 0.5, small), Segment(0, 0.95, PERSON))
    lab = SegmentationLabeler(seg, grid=8, label_fn=presenceLabelFn)
    l = lab.label(IMG, 0.0)
    assert l.bbox[3] == pytest.approx(1.0)  # Selects full-height higher-conf person.


def test_non_callable_segmenter_rejected():
    with pytest.raises(ValueError):
        SegmentationLabeler(segmenter=object())


def _grid_sum(mask, lab):
    """Occupancy-grid mass of a mask at the labeler's grid resolution (test helper)."""
    from wavetrace.groundtruth.CameraLabeler import _maskToGrid
    return sum(_maskToGrid(mask, lab._grid))
