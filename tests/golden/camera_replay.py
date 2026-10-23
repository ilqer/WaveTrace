"""Offline camera-collect replay shared by the camera goldens: a fake YOLO-seg net whose person and knife
masks follow a fixed schedule, a fake webcam stamping frame k at CLOCK_START_S + k/fps (the CSI clock), a
synchronous stand-in for the camera thread, and a monotonic clock the fake UDP socket advances."""

import socket
import threading
import time
from types import SimpleNamespace

import numpy as np

from mesh_replay import (
    CLOCK_START_S, RX_NODE_IDS, TX_MACS, FakeClock, FakeUdpSocket, encode_datagrams, generate_link_frames,
)
from wavetrace.groundtruth.yolo_labeler import YoloLabeler
from wavetrace.groundtruth.webcam import WebcamCapture, record_labels_online

PERSON_CLASS_ID, KNIFE_CLASS_ID = 0, 43
MASK_SIZE_PX = 32
PERSON_SPAN_S = (2.0, 4.0)
KNIFE_SPAN_S = (2.5, 3.5)
CAPTURE_S = 6.0
DURATION_S = 8.0
RECEIVE_TIMEOUT_S = 1.0


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


def _rectangle_mask(top_px, bottom_px, left_px, right_px):
    mask = np.zeros((MASK_SIZE_PX, MASK_SIZE_PX), dtype=np.float32)
    mask[top_px:bottom_px, left_px:right_px] = 1.0
    return mask


def _segment_frame(camera_time_s):
    """The person walks left to right through PERSON_SPAN_S, holding a visible knife through KNIFE_SPAN_S."""
    detections = []
    if PERSON_SPAN_S[0] <= camera_time_s < PERSON_SPAN_S[1]:
        left_px = 4 + int(16 * (camera_time_s - PERSON_SPAN_S[0]) / (PERSON_SPAN_S[1] - PERSON_SPAN_S[0]))
        detections.append((PERSON_CLASS_ID, 0.9, (left_px / MASK_SIZE_PX, 0.5, 0.25, 0.5),
                           _rectangle_mask(8, 24, left_px, left_px + 8)))
        if KNIFE_SPAN_S[0] <= camera_time_s < KNIFE_SPAN_S[1]:
            detections.append((KNIFE_CLASS_ID, 0.8, (left_px / MASK_SIZE_PX, 0.5, 0.1, 0.1),
                               _rectangle_mask(14, 18, left_px + 2, left_px + 6)))
    boxes = SimpleNamespace(
        xywhn=_Tensor(np.asarray([box for _, _, box, _ in detections], dtype=np.float32).reshape(-1, 4)),
        cls=_Tensor(np.asarray([class_id for class_id, _, _, _ in detections], dtype=np.float32)),
        conf=_Tensor(np.asarray([confidence for _, confidence, _, _ in detections], dtype=np.float32)))
    masks = np.stack([mask for _, _, _, mask in detections]) if detections else np.zeros(
        (0, MASK_SIZE_PX, MASK_SIZE_PX), dtype=np.float32)
    return [SimpleNamespace(boxes=boxes, masks=SimpleNamespace(data=_Tensor(masks)))]


class SynchronousThread:
    """Runs its target inside `start`, so the camera labels every frame before the CSI capture begins."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


def patch_camera(monkeypatch, frame_rate_hz):
    """Fake YOLO-seg net, webcam and camera pacing clock; the image handed to the net is the camera time."""
    monkeypatch.setattr(YoloLabeler, "_load", staticmethod(lambda weights_path, device: _segment_frame))
    num_reads = [0]

    def read_frame(webcam):
        camera_time_s = num_reads[0] / frame_rate_hz
        num_reads[0] += 1
        return CLOCK_START_S + camera_time_s, camera_time_s

    monkeypatch.setattr(WebcamCapture, "open", lambda webcam: webcam)
    monkeypatch.setattr(WebcamCapture, "read", read_frame)
    monkeypatch.setattr(WebcamCapture, "close", lambda webcam: None)
    camera_clock = FakeClock(0.0)

    def sleep(seconds):
        camera_clock.now_s += seconds

    monkeypatch.setattr(record_labels_online, "__kwdefaults__",
                        {**record_labels_online.__kwdefaults__, "sleep": sleep, "clock": camera_clock})
    monkeypatch.setattr(threading, "Thread", SynchronousThread)


def capture_datagrams(condition_kwargs):
    per_node = [encode_datagrams(rx_node_id, {
        tx_index: generate_link_frames(rx_node_id, tx_index, CAPTURE_S, **condition_kwargs)
        for tx_index in range(len(TX_MACS))}) for rx_node_id in RX_NODE_IDS]
    return [datagram for step in zip(*per_node) for datagram in step]


def present_person():
    return {"presence_spans": [PERSON_SPAN_S], "presence_turbulence_std": 0.15}


def visible_knife():
    return {"presence_spans": [PERSON_SPAN_S], "presence_turbulence_std": 0.15,
            "weapon_spans": [KNIFE_SPAN_S], "weapon_signature_depth": 0.4}


def patch_mesh(monkeypatch, datagrams):
    """Replays `datagrams`, then times out until the capture's monotonic deadline passes."""
    clock = FakeClock(CLOCK_START_S)

    def time_out():
        clock.now_s += RECEIVE_TIMEOUT_S
        raise socket.timeout

    fake_socket = FakeUdpSocket(clock, datagrams, on_drained=time_out)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: fake_socket)
    monkeypatch.setattr(time, "monotonic", clock)
