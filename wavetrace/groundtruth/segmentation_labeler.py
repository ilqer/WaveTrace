from dataclasses import dataclass

import numpy as np

from wavetrace.domain.groundtruth.labeler import FrameDetection, Labeler, presence_label_function
from wavetrace.groundtruth.vision_labeler import LabelerOptions


@dataclass
class Segment:
    """One segmentation hit. `mask` is an (H, W) array (bool or [0,1]) of the object's pixels;
    `bbox_xywhn` is optional — derived tightly from the mask when omitted."""

    cls: int
    confidence: float
    mask: np.ndarray            # (H, W); truthy where the object is
    bbox_xywhn: tuple = None    # (x, y, w, h) normalized; None -> computed from the mask


def _mask_bbox_xywhn(mask) -> tuple:
    """Tight normalized (x, y, w, h) enclosing the mask. O(H*W). (0,0,0,0) if empty."""
    is_object_pixel = np.asarray(mask) > 0
    rows = np.any(is_object_pixel, axis=1)
    cols = np.any(is_object_pixel, axis=0)
    if not rows.any():
        return (0.0, 0.0, 0.0, 0.0)
    height_px, width_px = is_object_pixel.shape
    top_px, bottom_px = np.argmax(rows), height_px - np.argmax(rows[::-1])
    left_px, right_px = np.argmax(cols), width_px - np.argmax(cols[::-1])
    return ((left_px + right_px) / (2 * width_px), (top_px + bottom_px) / (2 * height_px),
            (right_px - left_px) / width_px, (bottom_px - top_px) / height_px)


def _mask_overlap(weapon, person) -> float:
    """Fraction of the weapon mask that lies inside the person mask (intersection / weapon area).
    Gates false positives: a real concealed/carried weapon sits ON the body. O(H*W)."""
    is_weapon_pixel = np.asarray(weapon) > 0
    weapon_area_px = int(is_weapon_pixel.sum())
    if weapon_area_px == 0:
        return 0.0
    return float((is_weapon_pixel & (np.asarray(person) > 0)).sum()) / weapon_area_px


def _mask_to_grid(mask, grid: int) -> list:
    """Average-pool an (H, W) mask into a flattened grid*grid occupancy heatmap in [0,1], row-major.
    O(H*W) via one bincount over per-pixel cell ids (no per-pixel Python loop)."""
    pixel_values = np.asarray(mask, dtype=np.float32)
    height_px, width_px = pixel_values.shape
    rows = (np.arange(height_px) * grid) // height_px  # pixel-row -> grid-row
    cols = (np.arange(width_px) * grid) // width_px
    pixel_cell_ids = (rows[:, None] * grid + cols[None, :]).ravel()
    cell_value_sums = np.bincount(pixel_cell_ids, weights=pixel_values.ravel(), minlength=grid * grid)
    cell_pixel_counts = np.bincount(pixel_cell_ids, minlength=grid * grid)
    occupancy = cell_value_sums / np.maximum(cell_pixel_counts, 1)
    return occupancy.astype(np.float32).tolist()


class SegmentationLabeler(Labeler):
    """Camera-supervised labeler driven by a pretrained SEGMENTATION model — the high-confidence
    teacher for the CSI weapon-heatmap head. `segmenter(image) -> list[Segment]` runs the model on one
    frame (RGB or thermal — injected, so the camera choice is deferred); this picks the highest-confidence
    person (-> present + tight bbox) and accepts a weapon ONLY when its mask overlaps that person's by
    >= `overlap_min` (rejects free-floating false positives). The supervised mask
    (weapon when present, else person) is average-pooled to a `grid`x`grid` occupancy heatmap and
    stashed on the Label as the BCE/Dice target the CSI head learns to predict."""

    def __init__(self, segmenter, *, options: LabelerOptions = LabelerOptions(), confidence_threshold=0.5,
                 grid=16, overlap_min=0.5, label_function=presence_label_function):
        super().__init__(label_function)
        if not callable(segmenter):
            raise ValueError("SegmentationLabeler: segmenter must be callable(image) -> list[Segment]")
        self._segmenter = segmenter
        self._person = options.person_class
        self._weapon = set(options.weapon_classes)
        self._confidence_threshold = float(confidence_threshold)
        self._grid = int(grid)
        self._overlap_min = float(overlap_min)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        segments = [segment for segment in self._segmenter(observation)
                    if segment.confidence >= self._confidence_threshold]
        persons = [segment for segment in segments if segment.cls == self._person]
        best_person = max(persons, key=lambda segment: segment.confidence) if persons else None
        # highest-confidence weapon whose mask sits inside the person's; none without a person.
        weapon_segment = None
        if best_person is not None and self._weapon:
            candidates = sorted((segment for segment in segments if segment.cls in self._weapon),
                                key=lambda segment: segment.confidence, reverse=True)
            for segment in candidates:
                if _mask_overlap(segment.mask, best_person.mask) >= self._overlap_min:
                    weapon_segment = segment
                    break
        supervised_segment = weapon_segment if weapon_segment is not None else best_person
        occupancy_grid = (_mask_to_grid(supervised_segment.mask, self._grid)
                          if supervised_segment is not None else [])
        bbox = None
        if supervised_segment is not None:
            bbox = supervised_segment.bbox_xywhn or _mask_bbox_xywhn(supervised_segment.mask)
        return FrameDetection(
            present=best_person is not None,
            weapon=weapon_segment is not None,
            bbox=list(bbox) if bbox is not None else None,
            mask=occupancy_grid,
            mask_grid=self._grid if occupancy_grid else 0,
        )
