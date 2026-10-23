"""The dashboard's HTTP and WebSocket routes, driven through one `TestClient` inside the app's lifespan.

Only libraries (`subprocess`, `shutil`, `serial`), the foxglove bridge object and `DeviceHub` /
`WaveTraceRunner` methods are patched, never a route module, so these tests hold wherever a route lives.
"""

import asyncio
import json
import shutil
import subprocess
import threading
import time
import warnings
from types import SimpleNamespace

import pytest
import serial
import serial.tools.list_ports

with warnings.catch_warnings():
    # foxglove_websocket imports the legacy websockets API; starlette's TestClient warns that it runs on httpx
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient
    from web.app import app
    from web.foxglove import foxglove_server

from web.device_ctl import DeviceHub
from web.streamer import WaveTraceRunner

MESSAGE_BY_CHANNEL = {
    "inference": '{"class": 1, "conf": 0.9}',
    "stream": '{"ic": [0.5, -1.5]}',
    "logs": "plain log line",
    "training": '{"type": "epoch", "epoch": 1}',
    "telemetry": '{"hz": 12.5}',
}
SOCKET_TIMEOUT_S = 10.0


@pytest.fixture(scope="module")
def foxglove_payloads():
    return {"inference": [], "stream": []}


@pytest.fixture(scope="module")
def client(foxglove_payloads):
    async def start_no_server():
        return None

    async def record_inference(payload):
        foxglove_payloads["inference"].append(payload)

    async def record_stream(payload):
        foxglove_payloads["stream"].append(payload)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(foxglove_server, "start", start_no_server)
        patcher.setattr(foxglove_server, "publish_inference", record_inference)
        patcher.setattr(foxglove_server, "publish_stream", record_stream)
        with TestClient(app) as test_client:
            yield test_client


def _receive_text(websocket) -> str:
    """Fails instead of hanging when the server never sends."""
    received = []
    reader = threading.Thread(target=lambda: received.append(websocket.receive_text()), daemon=True)
    reader.start()
    reader.join(SOCKET_TIMEOUT_S)
    assert received, f"no message within {SOCKET_TIMEOUT_S} s"
    return received[0]


def _without_time_of_day(log_line: str) -> str:
    return log_line.split("] ", 1)[1]


def _wait_until(predicate) -> None:
    deadline_s = time.monotonic() + SOCKET_TIMEOUT_S
    while not predicate():
        assert time.monotonic() < deadline_s, "condition not reached in time"
        time.sleep(0.01)


def test_state_and_stop_before_any_start(client):
    assert client.get("/api/pipeline/state").json() == {"isRunning": False}
    assert client.post("/api/action/stop").json() == {"status": "stopped"}


def test_each_pipeline_channel_sends_its_queue_to_its_socket_and_foxglove_sees_inference_and_stream(
        client, foxglove_payloads, monkeypatch):
    def put_one_message_per_queue(runner, request):
        for channel_name, message in MESSAGE_BY_CHANNEL.items():
            queue = getattr(runner, f"{channel_name}_queue")
            asyncio.run_coroutine_threadsafe(queue.put(message), runner.loop).result()

    monkeypatch.setattr(WaveTraceRunner, "run_collection_request", put_one_message_per_queue)
    foxglove_payloads["inference"].clear()
    foxglove_payloads["stream"].clear()
    with (client.websocket_connect("/ws/inference") as inference_socket,
          client.websocket_connect("/ws/stream") as stream_socket,
          client.websocket_connect("/ws/logs") as logs_socket,
          client.websocket_connect("/ws/training") as training_socket,
          client.websocket_connect("/ws/telemetry") as telemetry_socket):
        assert client.post("/api/action/start", json={"action": "collect"}).json() == {"status": "started"}
        socket_by_channel = {"inference": inference_socket, "stream": stream_socket, "logs": logs_socket,
                             "training": training_socket, "telemetry": telemetry_socket}
        received_by_channel = {channel_name: _receive_text(socket)
                               for channel_name, socket in socket_by_channel.items()}
    assert received_by_channel == MESSAGE_BY_CHANNEL
    assert foxglove_payloads == {"inference": [{"class": 1, "conf": 0.9}], "stream": [{"ic": [0.5, -1.5]}]}


def test_an_inference_source_that_cannot_be_built_logs_a_fatal_error(client):
    with client.websocket_connect("/ws/logs") as logs_socket:
        response = client.post("/api/action/start", json={"action": "run", "synthetic": True, "subcarriers": 0})
        assert response.json() == {"status": "started"}
        log_lines = [_receive_text(logs_socket) for _ in range(2)]
    # the runner's log and start_action's error reach the queue through different loop callbacks, so either can
    # arrive first
    assert "FATAL ERROR: CsiFrame dimensions must be non-zero" in log_lines
    assert any(line.endswith("] [SIM] Synthetic source (no hardware connected).") for line in log_lines)


def test_a_synthetic_calibration_runs_to_pipeline_done(client, foxglove_payloads, tmp_path):
    calibration_dir = tmp_path / "cal"
    foxglove_payloads["inference"].clear()
    with (client.websocket_connect("/ws/logs") as logs_socket,
          client.websocket_connect("/ws/inference") as inference_socket):
        response = client.post("/api/action/start", json={
            "action": "calib", "synthetic": True, "duration": 2.0, "baseline_packets": 100,
            "cal_out": str(calibration_dir)})
        assert response.json() == {"status": "started"}
        assert [_without_time_of_day(_receive_text(logs_socket)) for _ in range(3)] == [
            "[SIM] Synthetic source (no hardware connected).",
            f"Starting Calibration -> {calibration_dir}",
            f"Calibration complete: {calibration_dir}",
        ]
        assert _receive_text(inference_socket) == '{"event": "pipeline_done"}'
        assert client.post("/api/action/stop").json() == {"status": "stopped"}
        assert _without_time_of_day(_receive_text(logs_socket)) == "System halt requested."
    assert (calibration_dir / "meta.json").exists()
    assert foxglove_payloads["inference"] == [{"event": "pipeline_done"}]
    assert client.get("/api/pipeline/state").json() == {"isRunning": False}


def test_an_unknown_action_starts_and_runs_nothing(client):
    assert client.post("/api/action/start", json={"action": "noop"}).json() == {"status": "started"}
    assert client.get("/api/pipeline/state").json() == {"isRunning": False}


def test_model_weights_refuse_a_path_outside_output(client):
    assert client.get("/api/model/weights", params={"model": "../x"}).json() == {
        "error": "model path must be inside output/ and must not escape it"}


def test_fusion_weights_refuse_a_path_outside_output(client):
    assert client.get("/api/fusion/weights", params={"path": "/etc/x"}).json() == {
        "error": "path escapes output/: /etc/x"}


def test_model_upload_refuses_a_destination_outside_output(client):
    assert client.post("/api/model/upload", json={"file_b64": "", "dest": "../x"}).json() == {
        "error": "path escapes output/: ../x"}


def test_calibration_info_without_a_calibration(client, tmp_path):
    missing_dir = str(tmp_path / "missing")
    assert client.get("/api/calib/info", params={"path": missing_dir}).json() == {
        "error": f"no calibration at {missing_dir} (run Calib first)"}


@pytest.mark.parametrize(("highest_subcarrier", "bandwidth_label"), [
    (63, "HT20 · 2.4 GHz"), (127, "HT40 · 2.4 GHz"), (255, "HT80 · 5 GHz")])
def test_calibration_info_names_the_bandwidth_from_the_widest_subcarrier(client, tmp_path, highest_subcarrier,
                                                                          bandwidth_label):
    (tmp_path / "meta.json").write_text(json.dumps(
        {"image_subcarriers": [0, 5, highest_subcarrier], "subcarriers": [5, 9]}))
    assert client.get("/api/calib/info", params={"path": str(tmp_path)}).json() == {
        "K": highest_subcarrier + 1, "bw_label": bandwidth_label, "n_selected": 2, "n_image": 3,
        "path": str(tmp_path)}


def test_weapon_litmus_without_recordings(client, tmp_path):
    assert client.get("/api/weapon/litmus", params={"root": str(tmp_path)}).json() == {
        "error": f"no weapon recordings under {tmp_path}/weapon_rec/*/<clear|weapon>/node*/"}


def test_path_scan_of_an_empty_directory(client, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert client.get("/api/paths/scan").json() == {"calibrations": [], "models": [], "datasets": []}


def test_browse_returns_the_chosen_path_without_its_trailing_slash(client, monkeypatch):
    commands = []

    def choose_a_file(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="/Users/me/model.joblib\n")

    monkeypatch.setattr(subprocess, "run", choose_a_file)
    response = client.get("/api/paths/browse", params={"type": "file", "prompt": "Pick", "ext": "joblib, npy"})
    assert response.json() == {"path": "/Users/me/model.joblib"}
    assert commands == [["osascript", "-e",
                         'POSIX path of (choose file with prompt "Pick" of type {"joblib", "npy"})']]


def test_browse_reports_a_cancelled_dialog(client, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    assert client.get("/api/paths/browse").json() == {"path": None, "cancelled": True}


def test_serial_ports_list_esp_like_ports_first(client, monkeypatch):
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [
        SimpleNamespace(device="/dev/ttyS0", description="", hwid=""),
        SimpleNamespace(device="/dev/cu.usbserial-1", description="CP2102", hwid="USB VID:PID=10C4")])
    assert client.get("/api/serial/ports").json() == {"ports": [
        {"device": "/dev/cu.usbserial-1", "description": "CP2102", "hwid": "USB VID:PID=10C4", "likely_esp": True},
        {"device": "/dev/ttyS0", "description": "", "hwid": "", "likely_esp": False}]}


def test_device_state_starts_idle(client):
    assert client.get("/api/device/state").json() == {"monitoring": [], "runningScripts": [], "busy": False}


def test_a_pi_run_without_a_host_reports_on_the_device_socket(client):
    with client.websocket_connect("/ws/device") as device_socket:
        assert client.post("/api/pi/run", json={"host": "", "command": "ls"}).json() == {
            "status": "running", "host": ""}
        assert _receive_text(device_socket).startswith(
            '{"source": "pi", "line": "no Pi host configured", "level": "error", "t": ')


def test_a_script_run_rejects_a_non_python_script_on_the_device_socket(client):
    with client.websocket_connect("/ws/device") as device_socket:
        assert client.post("/api/script/run", json={"script": "flash.sh"}).json() == {
            "status": "running", "script": "flash.sh"}
        assert _receive_text(device_socket).startswith(
            '{"source": "script", "line": "Invalid script name", "level": "error", "t": ')


def test_a_serial_monitor_reports_an_open_failure_and_stops_without_a_body(client, monkeypatch):
    def refuse_to_open(port, baud, timeout):
        raise OSError("no such port")

    monkeypatch.setattr(serial, "Serial", refuse_to_open)
    with client.websocket_connect("/ws/device") as device_socket:
        assert client.post("/api/serial/monitor/start", json={"port": "/dev/fake"}).json() == {
            "status": "monitoring", "port": "/dev/fake", "baud": 115200}
        assert _receive_text(device_socket).startswith(
            '{"source": "tty:fake", "line": "open failed: no such port", "level": "error", "t": ')
    assert client.get("/api/device/state").json()["monitoring"] == ["/dev/fake"]
    assert client.post("/api/serial/monitor/stop").json() == {"status": "stopped_all"}
    assert client.get("/api/device/state").json()["monitoring"] == []


def test_flash_hands_the_request_to_the_device_hub(client, monkeypatch):
    flash_calls = []
    monkeypatch.setattr(DeviceHub, "flash", lambda hub, role, node_id, port, clean: flash_calls.append(
        (role, node_id, port, clean)))
    assert client.post("/api/flash", json={"port": "/dev/fake", "node_id": 3}).json() == {
        "status": "flashing", "role": "node", "port": "/dev/fake", "clean": False}
    _wait_until(lambda: flash_calls)
    assert flash_calls == [("node", 3, "/dev/fake", False)]


def test_device_stop_and_input_without_a_running_process(client):
    assert client.post("/api/device/stop").json() == {"status": "idle"}
    assert client.post("/api/device/stop", json={"proc_id": "flash"}).json() == {"status": "idle"}
    assert client.post("/api/device/input", json={"proc_id": "pi", "input": "y\n"}).json() == {
        "status": "not_found"}


def test_camera_check_without_ffmpeg(client, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert client.get("/api/camera/check").json() == {
        "ok": False, "error": "ffmpeg not found — install it: brew install ffmpeg"}


def test_camera_check_names_a_denied_camera_permission(client, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/ffmpeg")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: SimpleNamespace(
        returncode=1, stdout=b"", stderr=b"[avfoundation] Camera permission denied"))
    assert client.get("/api/camera/check").json() == {
        "ok": False,
        "error": "Camera permission denied — allow Terminal in System Settings → Privacy → Camera"}


def test_camera_stream_without_ffmpeg_is_an_empty_multipart_body(client, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    response = client.get("/api/camera/stream")
    assert (response.status_code, response.content) == (200, b"")
    assert response.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
    assert client.post("/api/camera/stop").json() == {"ok": True}
