"""Phase 5a — label sources behind one pluggable interface (OFFLINE, no hot path).

A `Labeler` turns a per-frame observation (+timestamp) into a core `Label`. The *stage target*
(class_id, name) is decided by a pluggable `label_fn` so the same labeler serves presence (A) and
weapon (E) with no rework; the raw box/keypoints and the optional weapon `position` are always
preserved so the later localization / weapon-location-attention work needs no re-run.

Two roles share the interface:
  * stream labelers (camera-style) — `ReplayLabeler`, `ThermalLabeler`: consume an observation stream
    with its OWN clock → Labels must be timestamp-Aligned to CSI windows (their clock may be skewed).
  * time labelers — `ScriptedLabeler`, `LocationChipLabeler`: a function of CSI-clock time (the label
    is known by construction / from a chip track), evaluated directly at each window timestamp
    (`__call__(t) -> Label`), so DatasetBuilder needs no alignment for them.

The concrete vision models (YOLO/MediaPipe/SAM for the open + see-through tiers) and the thermal model
are documented SEAMS — subclass and override `_detect`; they need a model + recordings not present in
this environment.
"""

from abc import ABC, abstractmethod
import bisect
from dataclasses import dataclass, field
import json

import numpy as np

from wavetrace import Label


# ----- label policies: raw detection dict -> (class_id, name) -----------------------------------

def presence_label_fn(raw: dict, timestamp: float) -> tuple[int, str]:
    """Stage A: present/absent from whether a person was detected."""
    return (1, "present") if raw.get("present") else (0, "absent")


def weapon_label_fn(raw: dict, timestamp: float) -> tuple[int, str]:
    """Stage E: weapon present/absent (binary — the 'yes/no' the head collapses its heatmap to)."""
    return (1, "weapon") if raw.get("weapon") else (0, "no_weapon")


class Labeler(ABC):
    """Turn an observation (+timestamp) into a core `Label` via a pluggable `label_fn`.

    `label_fn(raw, timestamp) -> (class_id, name)` sets the stage target; raw bbox/keypoints and an
    optional weapon `position` are copied onto the Label (position -> bbox when no person box exists,
    so the weapon location survives for segment-training)."""

    def __init__(self, label_fn=presence_label_fn):
        self._label_fn = label_fn

    @abstractmethod
    def _detect(self, observation, timestamp: float) -> dict:
        """Return a raw detection dict {present, weapon, bbox, keypoints, position}."""

    def label(self, observation, timestamp: float) -> Label:
        raw = self._detect(observation, timestamp)
        class_id, name = self._label_fn(raw, timestamp)
        lab = Label()
        lab.class_id = class_id
        lab.name = name
        lab.timestamp = float(timestamp)
        box = raw.get("bbox") or raw.get("position")  # person box if present, else weapon location
        if box is not None:
            lab.bbox = list(box)
        kp = raw.get("keypoints")
        if kp:
            lab.keypoints = list(kp)
        m = raw.get("mask")
        if m:  # camera-supervised occupancy grid -> CSI heatmap-head target
            lab.mask = [float(v) for v in m]
            lab.mask_grid = int(raw.get("mask_grid") or 0)
        return lab

    def label_stream(self, observations) -> list[Label]:
        """observations: iterable of {"t": ts, ...} -> Labels sorted by timestamp (for Align)."""
        out = [self.label(o, o["t"]) for o in observations]
        out.sort(key=lambda l: l.timestamp)
        return out


class ReplayLabeler(Labeler):
    """Replays pre-computed detections (synthetic fixture or a recorded vision-model output stream).
    observation = {"t": ts, "raw": {detection}}. The concrete YOLO/MediaPipe/SAM adapter is a seam:
    subclass and override `_detect` to run the model on a real frame."""

    def _detect(self, observation, timestamp: float) -> dict:
        return observation.get("raw", observation)


@dataclass
class Detection:
    """One object-detector hit, geometry normalized to the frame ([0,1]). `cls` is the model's class
    id; `keypoints` is a flat [x0,y0,x1,y1,...] list (empty unless a pose model)."""

    cls: int
    conf: float
    bbox_xywhn: tuple  # (x, y, w, h) normalized to [0, 1]
    keypoints: list = field(default_factory=list)


class VisionLabeler(Labeler):
    """Camera-supervised labeler driven by a REAL object detector (the seam ReplayLabeler stood in
    for). `detector(image) -> list[Detection]` runs the model on one RGB frame; this picks the
    highest-confidence person (-> present + bbox + keypoints) and flags a weapon if any detection's
    class is in `weapon_classes`. The detector is injected, so the policy here is unit-testable with a
    stub and the heavy model (YOLO/MediaPipe) is a thin subclass — see `YoloLabeler`.

    observation = the RGB frame (an (H, W, 3) array or a path the detector accepts); timestamp comes
    from the stream. With `label_fn=presence_label_fn` it teaches Stage A; `weapon_label_fn` +
    `weapon_classes` teaches Stage E from a camera that CAN see the weapon (open-carry tier)."""

    def __init__(self, detector, *, person_class=0, weapon_classes=(), conf=0.35,
                 label_fn=presence_label_fn):
        super().__init__(label_fn)
        if not callable(detector):
            raise ValueError("VisionLabeler: detector must be callable(image) -> list[Detection]")
        self._detector = detector
        self._person = int(person_class)
        self._weapon = set(int(c) for c in weapon_classes)
        self._conf = float(conf)

    def _detect(self, observation, timestamp: float) -> dict:
        dets = [d for d in self._detector(observation) if d.conf >= self._conf]
        persons = [d for d in dets if d.cls == self._person]
        best = max(persons, key=lambda d: d.conf) if persons else None
        weapon = any(d.cls in self._weapon for d in dets)
        return {
            "present": best is not None,
            "weapon": weapon,
            "bbox": list(best.bbox_xywhn) if best is not None else None,
            "keypoints": list(best.keypoints) if best is not None else [],
            "position": None,
        }


class YoloLabeler(VisionLabeler):
    """Ultralytics-YOLO adapter (person detection + optional weapon classes; pose model -> keypoints).
    The dependency is OPTIONAL and imported lazily (`pip install ultralytics`); pass `model=` a
    pre-built model object to bypass the import (what the tests do).

    `model`: a weights path/name (e.g. "yolov8n.pt", "yolov8n-pose.pt") loaded via ultralytics, OR a
    ready model object exposing `__call__(image) -> results`. Results are converted to `Detection`s
    with boxes in normalized xywh; pose keypoints (normalized) ride along when present."""

    def __init__(self, model="yolov8n.pt", *, device=None, person_class=0, weapon_classes=(),
                 conf=0.35, imgsz=640, label_fn=presence_label_fn):
        net = self._load(model, device) if isinstance(model, (str, bytes)) else model

        def detector(image):
            results = net(image, imgsz=imgsz, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_detections(results)

        super().__init__(detector, person_class=person_class, weapon_classes=weapon_classes,
                         conf=conf, label_fn=label_fn)

    @staticmethod
    def _load(weights, device):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - exercised only without the optional dep
            raise ImportError(
                "YoloLabeler needs ultralytics: pip install ultralytics (or pass a prebuilt model=)"
            ) from e
        net = YOLO(weights)
        if device is not None:
            net.to(device)
        return net


def _accepts_kwargs(net) -> bool:
    """ultralytics models take imgsz/verbose kwargs; a bare test stub may not."""
    return hasattr(net, "predict") or hasattr(net, "model")


def _yolo_to_detections(results) -> list[Detection]:
    """Ultralytics Results (a list) -> [Detection] with normalized xywh boxes + optional keypoints."""
    out: list[Detection] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue
        xywhn = boxes.xywhn.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        kpts = getattr(res, "keypoints", None)
        kxy = kpts.xyn.cpu().numpy() if kpts is not None and kpts.xyn is not None else None
        for i in range(len(clss)):
            kp = kxy[i].reshape(-1).tolist() if kxy is not None else []
            out.append(Detection(int(clss[i]), float(confs[i]), tuple(xywhn[i].tolist()), kp))
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
    m = np.asarray(mask) > 0
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    if not rows.any():
        return (0.0, 0.0, 0.0, 0.0)
    H, W = m.shape
    y0, y1 = np.argmax(rows), H - np.argmax(rows[::-1])
    x0, x1 = np.argmax(cols), W - np.argmax(cols[::-1])
    return ((x0 + x1) / (2 * W), (y0 + y1) / (2 * H), (x1 - x0) / W, (y1 - y0) / H)


def _mask_overlap(weapon, person) -> float:
    """Fraction of the weapon mask that lies inside the person mask (intersection / weapon area).
    This is the A->E false-positive gate: a real concealed/carried weapon sits ON the body. O(H*W)."""
    w = np.asarray(weapon) > 0
    wa = int(w.sum())
    if wa == 0:
        return 0.0
    return float((w & (np.asarray(person) > 0)).sum()) / wa


def _mask_to_grid(mask, grid: int) -> list:
    """Average-pool an (H, W) mask into a flattened grid*grid occupancy heatmap in [0,1], row-major.
    O(H*W) via one bincount over per-pixel cell ids (no per-pixel Python loop)."""
    m = np.asarray(mask, dtype=np.float32)
    H, W = m.shape
    rows = (np.arange(H) * grid) // H          # pixel-row -> grid-row
    cols = (np.arange(W) * grid) // W
    cell = (rows[:, None] * grid + cols[None, :]).ravel()
    total = np.bincount(cell, weights=m.ravel(), minlength=grid * grid)
    count = np.bincount(cell, minlength=grid * grid)
    occ = total / np.maximum(count, 1)
    return occ.astype(np.float32).tolist()


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

    def __init__(self, segmenter, *, person_class=0, weapon_classes=(), conf=0.5, grid=16,
                 overlap_min=0.5, label_fn=presence_label_fn):
        super().__init__(label_fn)
        if not callable(segmenter):
            raise ValueError("SegmentationLabeler: segmenter must be callable(image) -> list[Segment]")
        self._seg = segmenter
        self._person = int(person_class)
        self._weapon = set(int(c) for c in weapon_classes)
        self._conf = float(conf)
        self._grid = int(grid)
        self._overlap_min = float(overlap_min)

    def _detect(self, observation, timestamp: float) -> dict:
        segs = [s for s in self._seg(observation) if s.conf >= self._conf]
        persons = [s for s in segs if s.cls == self._person]
        best = max(persons, key=lambda s: s.conf) if persons else None
        # A->E gate: highest-conf weapon whose mask sits inside the person's; none without a person.
        weapon_seg = None
        if best is not None and self._weapon:
            cands = sorted((s for s in segs if s.cls in self._weapon),
                           key=lambda s: s.conf, reverse=True)
            for s in cands:
                if _mask_overlap(s.mask, best.mask) >= self._overlap_min:
                    weapon_seg = s
                    break
        src = weapon_seg if weapon_seg is not None else best  # supervise on the weapon if we have one
        mask_grid = _mask_to_grid(src.mask, self._grid) if src is not None else []
        bbox = None
        if src is not None:
            bbox = src.bbox_xywhn or _mask_bbox_xywhn(src.mask)
        return {
            "present": best is not None,
            "weapon": weapon_seg is not None,
            "bbox": list(bbox) if bbox is not None else None,
            "keypoints": [],
            "position": None,
            "mask": mask_grid,
            "mask_grid": self._grid if mask_grid else 0,
        }


class YoloSegLabeler(SegmentationLabeler):
    """Ultralytics YOLO-seg adapter (e.g. "yolov8n-seg.pt") — the SegmentationLabeler over a real
    model. Optional dep imported lazily; pass `model=` a prebuilt object to bypass the import (tests).
    Results carry per-instance masks (`res.masks.data`, (N,H,W)) + normalized boxes -> `Segment`s."""

    def __init__(self, model="yolov8n-seg.pt", *, device=None, person_class=0, weapon_classes=(),
                 conf=0.5, grid=16, overlap_min=0.5, imgsz=640, label_fn=presence_label_fn):
        net = YoloLabeler._load(model, device) if isinstance(model, (str, bytes)) else model

        def segmenter(image):
            results = net(image, imgsz=imgsz, verbose=False) if _accepts_kwargs(net) else net(image)
            return _yolo_to_segments(results)

        super().__init__(segmenter, person_class=person_class, weapon_classes=weapon_classes,
                         conf=conf, grid=grid, overlap_min=overlap_min, label_fn=label_fn)


def _yolo_to_segments(results) -> list[Segment]:
    """Ultralytics seg Results -> [Segment] with binary masks + normalized xywh boxes."""
    out: list[Segment] = []
    for res in results:
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        if boxes is None or masks is None:
            continue
        xywhn = boxes.xywhn.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        data = masks.data.cpu().numpy()  # (N, H, W) soft masks in [0, 1]
        for i in range(len(clss)):
            out.append(Segment(int(clss[i]), float(confs[i]), data[i] > 0.5, tuple(xywhn[i].tolist())))
    return out


class ThermalLabeler(Labeler):
    """SEAM — concealed-tier thermal labeler (plan §5 iii, ref jimaging-11-00072). A thermal camera
    can often see a concealed weapon's cold-metal thermal shadow against body heat, labeling data the
    RGB camera can't. Their FP trick — accept a weapon detection only INSIDE a detected person bbox —
    is exactly our A→E gate. Needs a thermal model + recordings (absent here); wire by overriding
    `_detect` to run the thermal detector and gate it on the person box."""

    def _detect(self, observation, timestamp: float) -> dict:
        raise NotImplementedError(
            "ThermalLabeler is a hardware seam: provide a thermal detector and gate weapon "
            "detections on a person bbox (A→E gate). See plan §5 iii."
        )


class ScriptedLabeler(Labeler):
    """Concealed-tier (plan §5 ii): a camera cannot see a concealed weapon, so the label is known by
    construction (planted weapon). `spans` = iterable of (start, end, present) in the CSI clock; a
    timestamp inside a present span is labeled weapon. Time-style: call `__call__(t)`/`label_at(t)`.
    Default `label_fn` = weapon_label_fn."""

    def __init__(self, spans, label_fn=weapon_label_fn):
        super().__init__(label_fn)
        self._present = sorted((float(s), float(e)) for s, e, p in spans if p)

    def _detect(self, observation, timestamp: float) -> dict:
        weapon = any(s <= timestamp < e for s, e in self._present)
        return {"weapon": weapon, "present": weapon}

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at

    @classmethod
    def from_manifest(cls, path, label_fn=weapon_label_fn) -> "ScriptedLabeler":
        """JSON sidecar: {"spans": [{"start":s,"end":e,"present":bool}, ...]} (Q8)."""
        with open(path) as f:
            m = json.load(f)
        spans = [(x["start"], x["end"], x.get("present", True)) for x in m["spans"]]
        return cls(spans, label_fn)


class LocationChipLabeler(Labeler):
    """Concealed-tier (plan §5 iv, user idea): a BLE/UWB tag on the weapon gives precise time +
    position ground truth — stronger than a coarse scripted time span. `track` = iterable of
    (t, present, position) in the CSI clock; nearest-sample lookup yields present/absent + the weapon
    `position`, stashed on the Label so a later stage can use it as a delay/attention hint to segment
    the weapon's reflection from the body's (caveat: WiFi delay resolution is coarse + leakage risk →
    use it as a HINT, not a hard crop; see Phase-7 notes). Time-style: `__call__(t)`/`label_at(t)`."""

    def __init__(self, track, label_fn=weapon_label_fn):
        samples = sorted(track, key=lambda r: r[0])
        self._t = [float(r[0]) for r in samples]
        self._present = [bool(r[1]) for r in samples]
        self._pos = [r[2] for r in samples]
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

    def _detect(self, observation, timestamp: float) -> dict:
        i = self._nearest(timestamp)
        present = self._present[i]
        return {
            "weapon": present,
            "present": present,
            "position": list(self._pos[i]) if present and self._pos[i] is not None else None,
        }

    def label_at(self, timestamp: float) -> Label:
        return self.label(None, timestamp)

    __call__ = label_at
