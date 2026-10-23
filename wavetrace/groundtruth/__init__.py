"""Ground truth: turn a paired (CSI + label source) recording into a labeled dataset.

The camera, script or chip is only the teacher. What gets deployed reads CSI alone.

Two kinds share the `Labeler` interface:
  * stream labelers (`ReplayLabeler`, `VisionLabeler`, `SegmentationLabeler`, `ThermalLabeler`) read an
    observation stream on their OWN clock, so their labels must be timestamp-aligned to the CSI windows.
  * time labelers (`ScriptedLabeler`, `LocationChipLabeler`) are functions of CSI-clock time, so
    `build_dataset` evaluates them at each window timestamp and needs no alignment.

Further vision models (MediaPipe, SAM) and the thermal model are seams: subclass and override
`_detect`. They need a model and recordings that are not in this repo.
"""

from wavetrace.groundtruth.location_chip_labeler import LocationChipLabeler
from wavetrace.groundtruth.replay_labeler import ReplayLabeler
from wavetrace.groundtruth.segmentation_labeler import Segment, SegmentationLabeler
from wavetrace.groundtruth.thermal_labeler import ThermalLabeler
from wavetrace.groundtruth.vision_labeler import Detection, LabelerOptions, VisionLabeler
from wavetrace.groundtruth.yolo_labeler import YoloLabeler, YoloLabelerOptions, YoloSegLabeler

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
