import asyncio
import json
import socket
from types import SimpleNamespace

from fixtures.synthetic_csi import generate_quiet_baseline
from wavetrace.groundtruth.yolo_labeler import YoloLabeler
from wavetrace.groundtruth.webcam import WebcamCapture
from wavetrace.pipeline import calibrate_source
from wavetrace.entrypoints.api.streamer import WaveTraceRunner


class _RecordingQueue:
    def __init__(self, messages, queue_name):
        self._messages = messages
        self._queue_name = queue_name

    def put(self, message):
        self._messages.append((self._queue_name, message))


class _FrameListSource:
    def __init__(self, frames):
        self._frames = frames

    def frames(self):
        return iter(self._frames)


class _PortInUseSocket:
    def setsockopt(self, *option):
        pass

    def bind(self, address):
        raise OSError(48, "Address already in use")

    def close(self):
        pass


def _refuse_camera(webcam):
    raise RuntimeError("no camera in tests")


def test_a_port_that_cannot_be_bound_is_logged_as_the_real_error_and_ends_the_run(tmp_path, monkeypatch):
    baseline = generate_quiet_baseline(num_antennas=1, num_subcarriers=32, sample_rate_hz=100.0, num_frames=100,
                                       seed=1)
    calibrate_source(_FrameListSource(baseline), tmp_path / "cal" / "node1", baseline_packets=100)
    monkeypatch.setattr(YoloLabeler, "_load", staticmethod(lambda weights_path, device: lambda image: []))
    monkeypatch.setattr(WebcamCapture, "open", _refuse_camera)
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: _PortInUseSocket())
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: None)
    messages = []
    runner = WaveTraceRunner(None, *(_RecordingQueue(messages, name)
                                     for name in ("inference", "stream", "logs", "training", "telemetry")))
    request = SimpleNamespace(calibration=str(tmp_path / "cal"), train_data=str(tmp_path), duration=0.2, cam_index=0,
                              per_link=False, col_stage="presence", yolo_weights="yolov8n-seg.pt", udp_port=5599)
    runner.run_camera_collection_request(request)
    logs = [message.split("] ", 1)[1] for queue_name, message in messages if queue_name == "logs"]
    assert any(line.startswith("ERROR") and "OSError" in line and "Address already in use" in line for line in logs)
    assert not any("UnboundLocalError" in line for line in logs)
    assert messages[-1] == ("inference", json.dumps({"event": "pipeline_done"}))
    assert not runner.is_running
