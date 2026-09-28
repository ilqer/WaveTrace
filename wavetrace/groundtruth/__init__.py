"""Ground truth: turn a paired (CSI + label source) recording into a labeled dataset.

The camera, script or chip is only the teacher. What gets deployed reads CSI alone.
"""

from wavetrace.groundtruth.alignment import AlignmentResult, align, estimate_clock_offset
from wavetrace.groundtruth.camera_labeler import (
    Detection,
    FrameDetection,
    Labeler,
    LabelerOptions,
    LocationChipLabeler,
    ReplayLabeler,
    ScriptedLabeler,
    Segment,
    SegmentationLabeler,
    ThermalLabeler,
    VisionLabeler,
    YoloLabeler,
    YoloLabelerOptions,
    YoloSegLabeler,
    presence_label_fn,
    weapon_label_fn,
)
from wavetrace.groundtruth.dataset_builder import (
    Dataset,
    build_dataset,
    load_dataset,
    save_dataset,
)

__all__ = [
    "align",
    "estimate_clock_offset",
    "AlignmentResult",
    "Labeler",
    "LabelerOptions",
    "ReplayLabeler",
    "ScriptedLabeler",
    "LocationChipLabeler",
    "ThermalLabeler",
    "VisionLabeler",
    "YoloLabeler",
    "YoloLabelerOptions",
    "SegmentationLabeler",
    "YoloSegLabeler",
    "Segment",
    "Detection",
    "FrameDetection",
    "presence_label_fn",
    "weapon_label_fn",
    "Dataset",
    "build_dataset",
    "save_dataset",
    "load_dataset",
]
