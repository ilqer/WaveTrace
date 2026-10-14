"""Ground truth: turn a paired (CSI + label source) recording into a labeled dataset.

The camera, script or chip is only the teacher. What gets deployed reads CSI alone.
"""

from wavetrace.groundtruth.camera_labeler import (
    Detection,
    LabelerOptions,
    LocationChipLabeler,
    ReplayLabeler,
    Segment,
    SegmentationLabeler,
    ThermalLabeler,
    VisionLabeler,
    YoloLabeler,
    YoloLabelerOptions,
    YoloSegLabeler,
)

__all__ = [
    "LabelerOptions",
    "ReplayLabeler",
    "LocationChipLabeler",
    "ThermalLabeler",
    "VisionLabeler",
    "YoloLabeler",
    "YoloLabelerOptions",
    "SegmentationLabeler",
    "YoloSegLabeler",
    "Segment",
    "Detection",
]
