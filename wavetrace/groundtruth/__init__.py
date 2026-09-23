"""Ground truth: turn a paired (CSI + label source) recording into a labeled dataset.

The camera, script or chip is only the teacher. What gets deployed reads CSI alone.
"""

from wavetrace.groundtruth.Align import AlignmentResult, align, estimateClockOffset
from wavetrace.groundtruth.CameraLabeler import (
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
    "presenceLabelFn",
    "weaponLabelFn",
    "Dataset",
    "buildDataset",
    "saveDataset",
    "loadDataset",
]
