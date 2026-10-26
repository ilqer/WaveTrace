import os
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from wavetrace.groundtruth.vision_labeler import VisionLabeler, Detection, LabelerOptions
from wavetrace.domain.groundtruth.labeler import presence_label_function, weapon_label_function
from wavetrace.groundtruth.webcam import (WebcamCapture, WebcamOptions, record_frames, record_labels_online,
                                          stream_labels, COCO_WEAPON_CLASSES)


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


TINY_WEBCAM = WebcamOptions(index=0, width=2, height=1)
FRAME_SIZE_BYTES = 2 * 1 * 3


class _FakeFfmpeg:
    """A Popen whose stdout is a real pipe the test writes raw RGB bytes into."""

    def __init__(self, command, *, stdout, stderr, bufsize, ignores_terminate=False):
        read_fd, self.write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb", buffering=bufsize)
        self.calls = []
        self._ignores_terminate = ignores_terminate

    def send(self, data):
        os.write(self.write_fd, data)

    def hang_up(self):
        os.close(self.write_fd)

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")

    def wait(self, timeout=None):
        self.calls.append("wait")
        if self._ignores_terminate and "kill" not in self.calls:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return 0


@pytest.fixture
def spawn_ffmpeg(monkeypatch):
    """Returns spawn(first_bytes, **fake_options) -> the fake ffmpeg the next WebcamCapture.open starts."""
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/ffmpeg")
    processes = []

    def spawn(first_bytes, **fake_options):
        def start(command, **popen_options):
            process = _FakeFfmpeg(command, **popen_options, **fake_options)
            if first_bytes is None:
                process.hang_up()
            else:
                process.send(first_bytes)
            processes.append(process)
            return process
        monkeypatch.setattr(subprocess, "Popen", start)
        return processes

    return spawn


def test_webcam_reads_whole_frames_and_reaps_ffmpeg_on_close(spawn_ffmpeg):
    processes = spawn_ffmpeg(bytes(range(6)))
    with WebcamCapture(TINY_WEBCAM, clock=lambda: 12.5) as webcam:
        process = processes[0]
        process.send(bytes(range(6, 9)))
        process.send(bytes(range(9, 12)))
        first, second = webcam.read(), webcam.read()
    assert first[0] == 12.5 and first[1].tolist() == [[[0, 1, 2], [3, 4, 5]]]
    assert np.array_equal(second[1].ravel(), np.arange(6, 12))
    assert process.calls == ["terminate", "wait"] and process.stdout.closed


def test_a_webcam_that_sends_no_frame_is_reaped_before_the_error(spawn_ffmpeg):
    processes = spawn_ffmpeg(None)
    with pytest.raises(RuntimeError, match="cannot open webcam index 0"):
        WebcamCapture(TINY_WEBCAM).open()
    assert processes[0].calls == ["terminate", "wait"] and processes[0].stdout.closed


def test_close_kills_an_ffmpeg_that_ignores_terminate_and_still_waits(spawn_ffmpeg):
    processes = spawn_ffmpeg(bytes(FRAME_SIZE_BYTES), ignores_terminate=True)
    WebcamCapture(TINY_WEBCAM).open().close()
    assert processes[0].calls == ["terminate", "wait", "kill", "wait"]


def test_a_stop_ends_a_read_from_a_stalled_ffmpeg(spawn_ffmpeg):
    processes = spawn_ffmpeg(bytes(FRAME_SIZE_BYTES))
    stop = threading.Event()
    with WebcamCapture(TINY_WEBCAM, stop=stop) as webcam:
        webcam.read()
        threading.Timer(0.05, stop.set).start()
        read_start_s = time.monotonic()
        assert webcam.read() is None
        assert time.monotonic() - read_start_s < 2.0
    processes[0].hang_up()
