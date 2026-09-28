"""Label sources behind one interface: a per-frame observation and a timestamp become a `Label`.

Which target the label names - presence or weapon - is decided by a pluggable `label_fn`, so one
labeler serves both. The raw box, keypoints and any weapon position are always preserved, so later
localization work needs no re-run.

Two kinds share the interface:
  * stream labelers (`ReplayLabeler`, `ThermalLabeler`) read an observation stream on their OWN
    clock, so their labels must be timestamp-aligned to the CSI windows.
  * time labelers (`ScriptedLabeler`, `LocationChipLabeler`) are functions of CSI-clock time, so
    DatasetBuilder evaluates them at each window timestamp and needs no alignment.

The concrete vision models (YOLO, MediaPipe, SAM) and the thermal model are seams: subclass and
override `_detect`. They need a model and recordings that are not in this repo.
"""

from abc import ABC, abstractmethod
import bisect
from dataclasses import dataclass, field
import json

import numpy as np

from wavetrace import Label


# ----- label policies: FrameDetection -> (class_id, name) ---------------------------------------

def presence_label_fn(raw: "FrameDetection", timestamp: float) -> tuple[int, str]:
    """Stage A: present/absent from whether a person was detected."""
    return (1, "present") if raw.present else (0, "absent")


def weapon_label_fn(raw: "FrameDetection", timestamp: float) -> tuple[int, str]:
    """Stage E: weapon present/absent (binary — the 'yes/no' the head collapses its heatmap to)."""
    return (1, "weapon") if raw.weapon else (0, "no_weapon")


@dataclass(frozen=True, slots=True)
class FrameDetection:
    """One `Labeler._detect()` result — the input `Labeler.label()` turns into a core `Label`. Not
    to be confused with `Detection` (a single object-detector box). Every labeler subclass fills in
    only the fields it knows; the rest default to absent."""

    present: bool = False
    weapon: bool = False
    bbox: tuple | list | None = None
    keypoints: list = field(default_factory=list)
    position: tuple | list | None = None
    mask: list | None = None
    mask_grid: int = 0


class Labeler(ABC):
    """Turn an observation (+timestamp) into a core `Label` via a pluggable `label_fn`.

    `label_fn(raw, timestamp) -> (class_id, name)` sets the stage target; raw bbox/keypoints and an
    optional weapon `position` are copied onto the Label (position -> bbox when no person box exists,
    so the weapon location survives for segment-training)."""

    def __init__(self, label_fn=presence_label_fn):
        self._label_fn = label_fn

    @abstractmethod
    def _detect(self, observation, timestamp: float) -> FrameDetection:
        """Return this observation's detection."""

    def label(self, observation, timestamp: float) -> Label:
        raw = self._detect(observation, timestamp)
        class_id, name = self._label_fn(raw, timestamp)
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
        """observations: iterable of {"t": ts, ...} -> Labels sorted by timestamp (for Align)."""
        out = [self.label(observation, observation["t"]) for observation in observations]
        out.sort(key=lambda label: label.timestamp)
        return out


class ReplayLabeler(Labeler):
    """Replays pre-computed detections (synthetic fixture or a recorded vision-model output stream).
    observation = {"t": ts, "raw": {detection}}. The concrete YOLO/MediaPipe/SAM adapter is a seam:
    subclass and override `_detect` to run the model on a real frame."""

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        raw = observation.get("raw", observation)  # map the external dict at this one edge
        return FrameDetection(
            present=bool(raw.get("present", False)),
            weapon=bool(raw.get("weapon", False)),
            bbox=raw.get("bbox"),
            keypoints=list(raw.get("keypoints") or []),
            position=raw.get("position"),
            mask=raw.get("mask"),
            mask_grid=int(raw.get("mask_grid") or 0),
        )


@dataclass
class Detection:
    """One object-detector hit, geometry normalized to the frame ([0,1]). `cls` is the model's class
    id; `keypoints` is a flat [x0,y0,x1,y1,...] list (empty unless a pose model)."""

    cls: int
    conf: float
    bbox_xywhn: tuple  # (x, y, w, h) normalized to [0, 1]
    keypoints: list = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LabelerOptions:
    """Detector-selection knobs every `Labeler` subclass reads: which detector class id counts as
    the person, and which class ids count as a weapon."""

    person_class: int = 0
    weapon_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weapon_classes", tuple(int(class_id) for class_id in self.weapon_classes))
        if self.person_class < 0:
            raise ValueError("person_class must be >= 0")


@dataclass(frozen=True, slots=True)
class YoloLabelerOptions(LabelerOptions):
    """Adds the Ultralytics network-selection knobs only `YoloLabeler`/`YoloSegLabeler` read:
    `device` (a torch device string, e.g. "cuda"; `None` = ultralytics' own default) and
    `weights_path` (a weights file for that class to load; `None` = its own `DEFAULT_WEIGHTS`)."""

    device: str | None = None
    weights_path: str | None = None


class VisionLabeler(Labeler):
    """Camera-supervised labeler driven by a REAL object detector (the seam ReplayLabeler stood in
    for). `detector(image) -> list[Detection]` runs the model on one RGB frame; this picks the
    highest-confidence person (-> present + bbox + keypoints) and flags a weapon if any detection's
    class is in `weapon_classes`. The detector is injected, so the policy here is unit-testable with a
    stub and the heavy model (YOLO/MediaPipe) is a thin subclass — see `YoloLabeler`.

    observation = the RGB frame (an (H, W, 3) array or a path the detector accepts); timestamp comes
    from the stream. With `label_fn=presence_label_fn` it teaches Stage A; `weapon_label_fn` +
    `weapon_classes` teaches Stage E from a camera that CAN see the weapon (open-carry tier)."""

    def __init__(self, detector, *, options: LabelerOptions = LabelerOptions(), conf=0.35,
                 label_fn=presence_label_fn):
        super().__init__(label_fn)
        if not callable(detector):
            raise ValueError("VisionLabeler: detector must be callable(image) -> list[Detection]")
        self._detector = detector
        self._person = options.person_class
        self._weapon = set(options.weapon_classes)
        self._conf = float(conf)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        detections = [detection for detection in self._detector(observation) if detection.conf >= self._conf]
        persons = [detection for detection in detections if detection.cls == self._person]
        best = max(persons, key=lambda detection: detection.conf) if persons else None
        weapon = any(detection.cls in self._weapon for detection in detections)
        return FrameDetection(
            present=best is not None,
            weapon=weapon,
            bbox=list(best.bbox_xywhn) if best is not None else None,
            keypoints=list(best.keypoints) if best is not None else [],
        )


class YoloLabeler(VisionLabeler):
    """Ultralytics-YOLO adapter (person detection + optional weapon classes; pose model -> keypoints).
    The dependency is OPTIONAL and imported lazily (`pip install ultralytics`); pass `model=` an
    already-built model object to bypass the import (what the tests do).

    `model`: a ready model object exposing `__call__(image) -> results` — bypasses
    `options.weights_path` entirely. `options.weights_path`: a weights path/name (e.g.
    "yolov8n.pt", "yolov8n-pose.pt") for THIS class to load via ultralytics when `model` isn't
    given; `None` loads `DEFAULT_WEIGHTS`. Results are converted to `Detection`s with boxes in
    normalized xywh; pose keypoints (normalized) ride along when present."""

    DEFAULT_WEIGHTS = "yolov8n.pt"

    def __init__(self, *, options: YoloLabelerOptions = YoloLabelerOptions(), model=None, conf=0.35,
                 imgsz=640, label_fn=presence_label_fn):
        net = model if model is not None else self._load(
            options.weights_path or self.DEFAULT_WEIGHTS, options.device)

        def detector(image):
            results = net(image, imgsz=imgsz, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_detections(results)

        super().__init__(detector, options=options, conf=conf, label_fn=label_fn)

    @staticmethod
    def _load(weights_path, device):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "YoloLabeler needs ultralytics: pip install ultralytics (or pass model= a prebuilt model)"
            ) from e
        net = YOLO(weights_path)
        if device is not None:
            net.to(device)
        return net


def _accepts_kwargs(net) -> bool:
    """ultralytics models take imgsz/verbose kwargs; a bare test stub may not."""
    return hasattr(net, "predict") or hasattr(net, "model")


def _yolo_to_detections(results) -> list[Detection]:
    """Ultralytics Results (a list) -> [Detection] with normalized xywh boxes + optional keypoints."""
    out: list[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        boxes_xywhn = boxes.xywhn.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        keypoints = getattr(result, "keypoints", None)
        keypoints_xyn = (keypoints.xyn.cpu().numpy()
                         if keypoints is not None and keypoints.xyn is not None else None)
        for i in range(len(class_ids)):
            flat_keypoints = keypoints_xyn[i].reshape(-1).tolist() if keypoints_xyn is not None else []
            out.append(Detection(int(class_ids[i]), float(confidences[i]), tuple(boxes_xywhn[i].tolist()),
                                 flat_keypoints))
    return out


# ----- segmentation: pixel masks -> occupancy grid (the CSI heatmap-head target) ---------------

@dataclass
class Segment:
    """One segmentation hit. `mask` is an (H, W) array (bool or [0,1]) of the object's pixels;
    `bbox_xywhn` is optional — derived tightly from the mask when omitted."""

    cls: int
    conf: float
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
    This is the A->E false-positive gate: a real concealed/carried weapon sits ON the body. O(H*W)."""
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
    total = np.bincount(pixel_cell_ids, weights=pixel_values.ravel(), minlength=grid * grid)
    count = np.bincount(pixel_cell_ids, minlength=grid * grid)
    occupancy = total / np.maximum(count, 1)
    return occupancy.astype(np.float32).tolist()


class SegmentationLabeler(Labeler):
    """Camera-supervised labeler driven by a pretrained SEGMENTATION model — the high-confidence
    teacher for the CSI weapon-heatmap head. `segmenter(image) -> list[Segment]` runs the model on one
    frame (RGB or thermal — injected, so the camera choice is deferred); this picks the highest-conf
    person (-> present + tight bbox) and accepts a weapon ONLY when its mask overlaps that person's by
    >= `overlap_min` (the A->E gate that kills free-floating false positives). The supervised mask
    (weapon when present, else person) is average-pooled to a `grid`x`grid` occupancy heatmap and
    stashed on the Label as the BCE/Dice target the CSI head learns to predict.

    A weapon needs BOTH conf >= `conf` AND the mask-overlap gate, so spurious detections never label
    CSI. `grid` is stored with the mask, so the heatmap resolution stays tunable without a type change."""

    def __init__(self, segmenter, *, options: LabelerOptions = LabelerOptions(), conf=0.5, grid=16,
                 overlap_min=0.5, label_fn=presence_label_fn):
        super().__init__(label_fn)
        if not callable(segmenter):
            raise ValueError("SegmentationLabeler: segmenter must be callable(image) -> list[Segment]")
        self._seg = segmenter
        self._person = options.person_class
        self._weapon = set(options.weapon_classes)
        self._conf = float(conf)
        self._grid = int(grid)
        self._overlap_min = float(overlap_min)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        segments = [segment for segment in self._seg(observation) if segment.conf >= self._conf]
        persons = [segment for segment in segments if segment.cls == self._person]
        best = max(persons, key=lambda segment: segment.conf) if persons else None
        # A->E gate: highest-conf weapon whose mask sits inside the person's; none without a person.
        weapon_segment = None
        if best is not None and self._weapon:
            candidates = sorted((segment for segment in segments if segment.cls in self._weapon),
                                key=lambda segment: segment.conf, reverse=True)
            for segment in candidates:
                if _mask_overlap(segment.mask, best.mask) >= self._overlap_min:
                    weapon_segment = segment
                    break
        # supervise on the weapon if we have one
        supervised_segment = weapon_segment if weapon_segment is not None else best
        occupancy_grid = (_mask_to_grid(supervised_segment.mask, self._grid)
                          if supervised_segment is not None else [])
        bbox = None
        if supervised_segment is not None:
            bbox = supervised_segment.bbox_xywhn or _mask_bbox_xywhn(supervised_segment.mask)
        return FrameDetection(
            present=best is not None,
            weapon=weapon_segment is not None,
            bbox=list(bbox) if bbox is not None else None,
            mask=occupancy_grid,
            mask_grid=self._grid if occupancy_grid else 0,
        )


class YoloSegLabeler(SegmentationLabeler):
    """Ultralytics YOLO-seg adapter (e.g. "yolov8n-seg.pt") — the SegmentationLabeler over a real
    model. Optional dep imported lazily; pass `model=` an already-built segmenter object to bypass
    the import (tests). See `YoloLabeler` for the `model` vs. `options.weights_path` split. Results
    carry per-instance masks (`result.masks.data`, (N,H,W)) + normalized boxes -> `Segment`s."""

    DEFAULT_WEIGHTS = "yolov8n-seg.pt"

    def __init__(self, *, options: YoloLabelerOptions = YoloLabelerOptions(), model=None, conf=0.5,
                 grid=16, overlap_min=0.5, imgsz=640, label_fn=presence_label_fn):
        net = model if model is not None else YoloLabeler._load(
            options.weights_path or self.DEFAULT_WEIGHTS, options.device)

        def segmenter(image):
            results = net(image, imgsz=imgsz, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_segments(results)

        super().__init__(segmenter, options=options, conf=conf, grid=grid,
                         overlap_min=overlap_min, label_fn=label_fn)


def _yolo_to_segments(results) -> list[Segment]:
    """Ultralytics seg Results -> [Segment] with binary masks + normalized xywh boxes."""
    out: list[Segment] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        if boxes is None or masks is None:
            continue
        boxes_xywhn = boxes.xywhn.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        soft_masks = masks.data.cpu().numpy()  # (N, H, W) soft masks in [0, 1]
        for i in range(len(class_ids)):
            out.append(Segment(int(class_ids[i]), float(confidences[i]), soft_masks[i] > 0.5,
                               tuple(boxes_xywhn[i].tolist())))
    return out


class ThermalLabeler(Labeler):
    """A thermal camera can see a concealed weapon's cold-metal shadow against body heat, labeling
    data an RGB camera cannot. Accepting a detection only inside a detected person box keeps the
    false-positive rate down. Needs a thermal model and recordings that are not here: wire it by
    overriding `_detect`."""

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        raise NotImplementedError(
            "ThermalLabeler is a hardware seam: provide a thermal detector and gate weapon "
            "detections on a person bbox."
        )


class ScriptedLabeler(Labeler):
    """A camera cannot see a concealed weapon, so the label is known by construction: the weapon was
    planted. `spans` is (start, end, present) on the CSI clock, and a timestamp inside a present
    span is labeled weapon. Called as `__call__(t)` / `label_at(t)`."""

    def __init__(self, spans, label_fn=weapon_label_fn):
        super().__init__(label_fn)
        self._present = sorted((float(start_s), float(end_s))
                               for start_s, end_s, is_present in spans if is_present)

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        weapon = any(start_s <= timestamp < end_s for start_s, end_s in self._present)
        return FrameDetection(present=weapon, weapon=weapon)

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at

    @classmethod
    def from_manifest(cls, path, label_fn=weapon_label_fn) -> "ScriptedLabeler":
        """JSON sidecar: {"spans": [{"start":s,"end":e,"present":bool}, ...]} (Q8)."""
        with open(path) as f:
            manifest = json.load(f)
        spans = [(span["start"], span["end"], span.get("present", True)) for span in manifest["spans"]]
        return cls(spans, label_fn)


class LocationChipLabeler(Labeler):
    """A BLE/UWB tag on the weapon gives precise time and position ground truth, which beats a
    coarse scripted span. `track` is (t, present, position) on the CSI clock, looked up by nearest
    sample. The position is stashed on the Label as a hint for separating the weapon's reflection
    from the body's - a hint only, because WiFi delay resolution is coarse and leaks."""

    def __init__(self, track, label_fn=weapon_label_fn):
        samples = sorted(track, key=lambda sample: sample[0])
        self._t = [float(sample[0]) for sample in samples]
        self._present = [bool(sample[1]) for sample in samples]
        self._pos = [sample[2] for sample in samples]
        super().__init__(label_fn)

    def _nearest(self, timestamp: float) -> int:
        """Index of the track sample nearest in time. O(log n)."""
        if not self._t:
            raise ValueError("LocationChipLabeler: empty track")
        i = bisect.bisect_left(self._t, timestamp)
        if i == 0:
            return 0
        if i >= len(self._t):
            return len(self._t) - 1
        return i if (self._t[i] - timestamp) < (timestamp - self._t[i - 1]) else i - 1

    def _detect(self, observation, timestamp: float) -> FrameDetection:
        i = self._nearest(timestamp)
        present = self._present[i]
        return FrameDetection(
            present=present,
            weapon=present,
            position=list(self._pos[i]) if present and self._pos[i] is not None else None,
        )

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at
