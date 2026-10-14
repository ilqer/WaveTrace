import threading

from wavetrace.groundtruth.camera_labeler import (VisionLabeler, Detection, LabelerOptions,
                                                 presence_label_function, weapon_label_function)
from wavetrace.groundtruth.webcam import (record_frames, record_labels_online, stream_labels,
                                          COCO_WEAPON_CLASSES)


def _stub_detector(image):
    """person->[person box]; weapon->[person + knife(43)]; else nothing."""
    if image == "person":
        return [Detection(0, 0.9, (0.5, 0.5, 0.2, 0.4))]
    if image == "weapon":
        return [Detection(0, 0.9, (0.5, 0.5, 0.2, 0.4)), Detection(43, 0.8, (0.6, 0.6, 0.1, 0.1))]
    return []


def test_stream_labels_presence_sorted_and_classified():
    labeler = VisionLabeler(_stub_detector, label_function=presence_label_function)
    out = stream_labels(labeler, [(2.0, "person"), (1.0, "empty")])
    assert [label.timestamp for label in out] == [1.0, 2.0]      # sorted by time (for align)
    class_id_by_timestamp = {label.timestamp: label.class_id for label in out}
    # person -> present, empty -> absent
    assert class_id_by_timestamp[2.0] == 1 and class_id_by_timestamp[1.0] == 0


def test_stream_labels_open_carry_weapon():
    labeler = VisionLabeler(_stub_detector, options=LabelerOptions(weapon_classes=COCO_WEAPON_CLASSES),
                        label_function=weapon_label_function)
    out = stream_labels(labeler, [(0.0, "weapon"), (1.0, "person")])
    class_id_by_timestamp = {label.timestamp: label.class_id for label in out}
    assert class_id_by_timestamp[0.0] == 1     # visible knife -> weapon
    assert class_id_by_timestamp[1.0] == 0     # person only -> no weapon


def test_stream_labels_max_frames():
    labeler = VisionLabeler(_stub_detector, label_function=presence_label_function)
    out = stream_labels(labeler, [(0.0, "person")] * 5, max_frames=2)
    assert len(out) == 2


def test_record_frames_buffers_every_grab():
    clock_readings_s = iter([0.0, 0.0, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.30])
    clock = lambda: next(clock_readings_s, 999.0)
    calls = {"n": 0}
    def grab():
        calls["n"] += 1
        return ("frame", calls["n"])
    out = record_frames(grab, 0.2, frame_rate_hz=20.0, sleep=lambda seconds: None, clock=clock)
    assert all(isinstance(item, tuple) and item[0] == "frame" for item in out)
    assert len(out) == calls["n"]


def test_record_frames_stop_event_returns_empty():
    stop_event = threading.Event(); stop_event.set()
    calls = {"n": 0}
    def grab():
        calls["n"] += 1
        return ("f", 1)
    out = record_frames(grab, 5.0, frame_rate_hz=10.0, stop=stop_event, sleep=lambda seconds: None,
                        clock=lambda: 0.0)
    assert out == [] and calls["n"] == 0     # pre-set stop -> no capture


def test_record_labels_online_labels_live_and_calls_back():
    labeler = VisionLabeler(_stub_detector, label_function=presence_label_function)
    timed_images = iter([(2.0, "person"), (1.0, "empty"), (0.5, "person")])
    ticks = {"t": 0.0}
    def clock():                       # advancing clock so the duration window actually closes
        time_s = ticks["t"]; ticks["t"] += 0.001; return time_s
    seen = []
    out = record_labels_online(lambda: next(timed_images, None), labeler, 0.05, frame_rate_hz=1000.0,
                               on_label=lambda label: seen.append(label.class_id),
                               sleep=lambda seconds: None, clock=clock)
    assert [label.timestamp for label in out] == [0.5, 1.0, 2.0]   # sorted for align
    assert len(seen) == 3
    assert sum(label.class_id for label in out) == 2                # two 'person' -> present


def test_coco_weapon_default_is_knife():
    assert 43 in COCO_WEAPON_CLASSES
