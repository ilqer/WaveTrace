"""Phase 5 — ground-truth pipeline (camera-supervised dataset, OFFLINE).

Turns a paired (CSI + label-source) recording into a serialized labeled dataset {(x_t, label_t)} for
Phase 6/7 CSI-only training. Deployment stays CSI-only; the camera/scripted/chip label is the teacher.
"""

from wavetrace.groundtruth.Align import AlignmentResult, align, estimateClockOffset
from wavetrace.groundtruth.CameraLabeler import (
    Detection,
    Labeler,
    LocationChipLabeler,
    ReplayLabeler,
    ScriptedLabeler,
    Segment,
    SegmentationLabeler,
    ThermalLabeler,
    VisionLabeler,
    YoloLabeler,
    YoloSegLabeler,
    presenceLabelFn,
    weaponLabelFn,
)
from wavetrace.groundtruth.DatasetBuilder import (
    Dataset,
    buildDataset,
    loadDataset,
    saveDataset,
)

__all__ = [
    "align",
    "estimateClockOffset",
    "AlignmentResult",
    "Labeler",
    "ReplayLabeler",
    "ScriptedLabeler",
    "LocationChipLabeler",
    "ThermalLabeler",
    "VisionLabeler",
    "YoloLabeler",
    "SegmentationLabeler",
    "YoloSegLabeler",
    "Segment",
    "Detection",
    "presenceLabelFn",
    "weaponLabelFn",
    "Dataset",
    "buildDataset",
    "saveDataset",
    "loadDataset",
]
