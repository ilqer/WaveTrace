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
import json

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
