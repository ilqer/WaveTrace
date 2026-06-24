"""MacBook webcam camera-labeling core: offline YOLO labeling of (ts, frame) streams and the
throttled frame buffer. The cv2/ultralytics deps are NOT needed here — `stream_labels` is detector-
agnostic (stub detector via VisionLabeler) and `record_frames` takes an injected grab/clock."""

import threading

from wavetrace.groundtruth.CameraLabeler import (VisionLabeler, Detection,
                                                 presence_label_fn, weapon_label_fn)
from wavetrace.groundtruth.Webcam import (record_frames, record_labels_online, stream_labels,
                                          COCO_WEAPON_CLASSES)


def _stub_detector(image):
    """person->[person box]; weapon->[person + knife(43)]; else nothing."""
    if image == "person":
        return [Detection(0, 0.9, (0.5, 0.5, 0.2, 0.4))]
    if image == "weapon":
        return [Detection(0, 0.9, (0.5, 0.5, 0.2, 0.4)), Detection(43, 0.8, (0.6, 0.6, 0.1, 0.1))]
    return []


def test_stream_labels_presence_sorted_and_classified():
    lab = VisionLabeler(_stub_detector, label_fn=presence_label_fn)
    out = stream_labels(lab, [(2.0, "person"), (1.0, "empty")])
    assert [l.timestamp for l in out] == [1.0, 2.0]      # sorted by time (for align)
    cls = {l.timestamp: l.class_id for l in out}
    assert cls[2.0] == 1 and cls[1.0] == 0               # person -> present, empty -> absent


def test_stream_labels_open_carry_weapon():
    lab = VisionLabeler(_stub_detector, weapon_classes=COCO_WEAPON_CLASSES,
                        label_fn=weapon_label_fn)
    out = stream_labels(lab, [(0.0, "weapon"), (1.0, "person")])
    cls = {l.timestamp: l.class_id for l in out}
    assert cls[0.0] == 1     # visible knife -> weapon
    assert cls[1.0] == 0     # person only -> no weapon


def test_stream_labels_max_frames():
    lab = VisionLabeler(_stub_detector, label_fn=presence_label_fn)
    out = stream_labels(lab, [(0.0, "person")] * 5, max_frames=2)
    assert len(out) == 2


def test_record_frames_buffers_every_grab():
    seq = iter([0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.30])
    clock = lambda: next(seq, 999.0)
    calls = {"n": 0}
    def grab():
        calls["n"] += 1
        return ("frame", calls["n"])
    out = record_frames(grab, 0.2, fps=20.0, sleep=lambda s: None, clock=clock)
    assert all(isinstance(x, tuple) and x[0] == "frame" for x in out)
    assert len(out) == calls["n"]            # every grabbed frame is buffered


def test_record_frames_stop_event_returns_empty():
    ev = threading.Event(); ev.set()
    calls = {"n": 0}
    def grab():
        calls["n"] += 1
        return ("f", 1)
    out = record_frames(grab, 5.0, fps=10.0, stop=ev, sleep=lambda s: None, clock=lambda: 0.0)
    assert out == [] and calls["n"] == 0     # pre-set stop -> no capture


def test_record_labels_online_labels_live_and_calls_back():
    """Online path runs the labeler per grabbed frame and fires on_label live; returns sorted Labels."""
    lab = VisionLabeler(_stub_detector, label_fn=presence_label_fn)
    imgs = iter([(2.0, "person"), (1.0, "empty"), (0.5, "person")])
    ticks = {"t": 0.0}
    def clock():                       # advancing clock so the duration window actually closes
        t = ticks["t"]; ticks["t"] += 0.001; return t
    seen = []
    out = record_labels_online(lambda: next(imgs, None), lab, 0.05, fps=1000.0,
                               on_label=lambda l: seen.append(l.class_id),
                               sleep=lambda s: None, clock=clock)
    assert [l.timestamp for l in out] == [0.5, 1.0, 2.0]   # sorted for align
    assert len(seen) == 3                                    # live callback per frame
    assert sum(l.class_id for l in out) == 2                # two 'person' -> present


def test_coco_weapon_default_is_knife():
    assert 43 in COCO_WEAPON_CLASSES
