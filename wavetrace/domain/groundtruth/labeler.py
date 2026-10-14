"""Label-source interface: a per-frame observation and a timestamp become a `Label`.

Which target the label names - presence or weapon - is decided by a pluggable `label_function`, so one
labeler serves both. The raw box, keypoints and any weapon position are always preserved, so later
localization work needs no re-run.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from wavetrace import Label


def presence_label_function(raw: "FrameDetection", timestamp: float) -> tuple[int, str]:
    return (1, "present") if raw.present else (0, "absent")


def weapon_label_function(raw: "FrameDetection", timestamp: float) -> tuple[int, str]:
    return (1, "weapon") if raw.weapon else (0, "no_weapon")


@dataclass(frozen=True, slots=True)
class FrameDetection:
    """One `Labeler._detect()` result — the input `Labeler.label()` turns into a core `Label`. Not
    to be confused with `camera_labeler.Detection` (a single object-detector box)."""

    present: bool = False
    weapon: bool = False
    bbox: tuple | list | None = None
    keypoints: list = field(default_factory=list)
    position: tuple | list | None = None
    mask: list | None = None
    mask_grid: int = 0


class Labeler(ABC):
    """Turn an observation (+timestamp) into a core `Label` via a pluggable `label_function`.

    `label_function(raw, timestamp) -> (class_id, name)` sets the stage target; raw bbox/keypoints and an
    optional weapon `position` are copied onto the Label (position -> bbox when no person box exists,
    so the weapon location survives for segment-training)."""

    def __init__(self, label_function=presence_label_function):
        self._label_function = label_function

    @abstractmethod
    def _detect(self, observation, timestamp: float) -> FrameDetection:
        """Return this observation's detection."""

    def label(self, observation, timestamp: float) -> Label:
        raw = self._detect(observation, timestamp)
        class_id, name = self._label_function(raw, timestamp)
        label = Label()
        label.class_id = class_id
        label.name = name
        label.timestamp = float(timestamp)
        box = raw.bbox or raw.position  # person box if present, else weapon location
        if box is not None:
            label.bbox = list(box)
        if raw.keypoints:
            label.keypoints = list(raw.keypoints)
        if raw.mask:  # camera-supervised occupancy grid -> CSI heatmap-head target
            label.mask = [float(mask_value) for mask_value in raw.mask]
            label.mask_grid = int(raw.mask_grid or 0)
        return label

    def label_stream(self, observations) -> list[Label]:
        """observations: iterable of {"t": ts, ...} -> Labels sorted by timestamp (the input `align` expects)."""
        labels = [self.label(observation, observation["t"]) for observation in observations]
        labels.sort(key=lambda label: label.timestamp)
        return labels


class ScriptedLabeler(Labeler):
    """A camera cannot see a concealed weapon, so the label is known by construction: the weapon was
    planted. `spans` is (start, end, present) on the CSI clock, and a timestamp inside a present
    span is labeled weapon. Called as `__call__(t)` / `label_at(t)`."""

    def __init__(self, spans, label_function=weapon_label_function):
        super().__init__(label_function)
        self._present = sorted((float(start_s), float(end_s))
                               for start_s, end_s, is_present in spans if is_present)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        weapon = any(start_s <= timestamp < end_s for start_s, end_s in self._present)
        return FrameDetection(present=weapon, weapon=weapon)

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at
