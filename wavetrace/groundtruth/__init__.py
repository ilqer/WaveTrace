"""Phase 5 — ground-truth pipeline (camera-supervised dataset, OFFLINE).

Turns a paired (CSI + label-source) recording into a serialized labeled dataset {(x_t, label_t)} for
Phase 6/7 CSI-only training. Deployment stays CSI-only; the camera/scripted/chip label is the teacher.
"""

from wavetrace.groundtruth.Align import AlignmentResult, align, estimate_clock_offset
from wavetrace.groundtruth.CameraLabeler import (
    Labeler,
    LocationChipLabeler,
    ReplayLabeler,
    ScriptedLabeler,
    ThermalLabeler,
    presence_label_fn,
    weapon_label_fn,
)
from wavetrace.groundtruth.DatasetBuilder import (
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
    "ReplayLabeler",
    "ScriptedLabeler",
    "LocationChipLabeler",
    "ThermalLabeler",
    "presence_label_fn",
    "weapon_label_fn",
    "Dataset",
    "build_dataset",
    "save_dataset",
    "load_dataset",
]
