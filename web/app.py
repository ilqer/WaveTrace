from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import json
import uvicorn

from web.server_options import WebServerOptions
from web.streamer import WaveTraceRunner
from web.foxglove import foxglove_server
from web.device_ctl import DeviceHub, list_serial_ports


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue, stream_queue, logs_queue, training_queue, telemetry_queue
    global device_queue, device_hub
    inference_queue = asyncio.Queue()
    stream_queue = asyncio.Queue()
    logs_queue = asyncio.Queue()
    training_queue = asyncio.Queue()
    telemetry_queue = asyncio.Queue()
    device_queue = asyncio.Queue()
    device_hub = DeviceHub(asyncio.get_running_loop(), device_queue)
    await foxglove_server.start()
    asyncio.create_task(broadcast_inference())
    asyncio.create_task(broadcast_stream())
    asyncio.create_task(broadcast_logs())
    asyncio.create_task(broadcast_training())
    asyncio.create_task(broadcast_telemetry())
    asyncio.create_task(broadcast_device())
    yield
    global runner
    if runner:
        runner.stop()
        await asyncio.sleep(0.5)


# joblib.load unpickles, so both reads and writes are confined to output/.
ALLOWED_ROOT = os.path.realpath("output")

JPEG_START_OF_IMAGE = b"\xff\xd8"
JPEG_END_OF_IMAGE = b"\xff\xd9"

def _resolve_output_path(path: str) -> str:
    resolved_path = os.path.realpath(path)
    if os.path.commonpath([resolved_path, ALLOWED_ROOT]) != ALLOWED_ROOT:
        raise ValueError(f"path escapes output/: {path}")
    return resolved_path


app = FastAPI(title="WaveTrace Lab Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner_task = None
runner = None
inference_queue = None
stream_queue = None
logs_queue = None
training_queue = None
telemetry_queue = None
device_queue = None
device_hub = None

class StartRequest(BaseModel):
    action: str = "run"  # run | calib | collect | train | camera_collect

    synthetic: bool = False
    antennas: int = 2
    subcarriers: int = 64
    fs: float = 100.0
    duration: float = 60.0
    seed: int = 0
    udp_port: int = 9876  # must match the port the firmware pushes CSI to

    mode: str = "presence"
    calibration: str = "data/2g4_ht40/ui/cal"
    model: str = "data/2g4_ht40/ui/model/model.joblib"
    gainLock: bool = True  # noqa: N815 - request-body key; clients send it under this name
    vote: bool = True
    frame_average: int = 1
    use_baseline: bool = False

    baseline_packets: int = 300
    cal_out: str = "data/2g4_ht40/ui/cal"

    col_stage: str = "presence"
    col_spans: str = "0:5,10:15,20:25"
    col_window: int = 128
    col_hop: int = 32
    subtract_ic_baseline: bool = True

    train_backend: str = "mlp"
    train_out: str = "data/2g4_ht40/ui/model"
    train_data: str = "data/2g4_ht40/ui/ds"  # a dataset dir, or a parent whose node*/ subdirs are pooled

    cam_url: str = "/api/camera/stream"
    cam_index: int = 0
    per_link: bool = False
    yolo_weights: str = "yolov8n-seg.pt"

clients_inference = set()
clients_stream = set()
clients_logs = set()
clients_training = set()
clients_telemetry = set()
clients_device = set()

async def broadcast_inference():
    while True:
        data = await inference_queue.get()
        await foxglove_server.publish_inference(json.loads(data))
        for client in list(clients_inference):
            try:
                await client.send_text(data)
            except Exception:
                clients_inference.discard(client)

async def broadcast_stream():
    while True:
        data = await stream_queue.get()
        await foxglove_server.publish_stream(json.loads(data))
        for client in list(clients_stream):
            try:
                await client.send_text(data)
            except Exception:
                clients_stream.discard(client)

async def broadcast_logs():
    while True:
        data = await logs_queue.get()
        for client in list(clients_logs):
            try:
                await client.send_text(data)
            except Exception:
                clients_logs.discard(client)

async def broadcast_training():
    while True:
        data = await training_queue.get()
        for client in list(clients_training):
            try:
                await client.send_text(data)
            except Exception:
                clients_training.discard(client)

async def broadcast_telemetry():
    while True:
        data = await telemetry_queue.get()
        for client in list(clients_telemetry):
            try:
                await client.send_text(data)
            except Exception:
                clients_telemetry.discard(client)

async def broadcast_device():
    while True:
        data = await device_queue.get()
        for client in list(clients_device):
            try:
                await client.send_text(data)
            except Exception:
                clients_device.discard(client)

@app.post("/api/action/start")
async def start_action(request: StartRequest):
    global runner, runner_task, inference_queue, stream_queue, logs_queue
    
    if runner and runner.is_running:
        runner.stop()
        await asyncio.sleep(0.5)

    loop = asyncio.get_running_loop()
    runner = WaveTraceRunner(loop, inference_queue, stream_queue, logs_queue, training_queue,
                             telemetry_queue)
    
    def run_request():
        try:
            if request.action == "run":
                runner.run_inference_request(request)
            elif request.action == "calib":
                runner.run_calibration_request(request)
            elif request.action == "collect":
                runner.run_collection_request(request)
            elif request.action == "train":
                runner.run_training_request(request)
            elif request.action == "camera_collect":
                runner.run_camera_collection_request(request)
        except Exception as error:
            loop.call_soon_threadsafe(logs_queue.put_nowait, f"FATAL ERROR: {str(error)}")

    # run_request blocks, so it runs on a worker thread; stop_inference awaits the task later.
    runner_task = asyncio.create_task(asyncio.to_thread(run_request))
    return {"status": "started"}

@app.get("/api/pipeline/state")
async def pipeline_state():
    global runner
    return {"isRunning": runner.is_running if runner else False}

@app.post("/api/action/stop")
async def stop_inference():
    global runner, runner_task
    if runner:
        runner.stop()
    if runner_task:
        try:
            await asyncio.wait_for(asyncio.shield(runner_task), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass
        runner_task = None
    return {"status": "stopped"}

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    clients_inference.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_inference.discard(websocket)

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    clients_stream.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_stream.discard(websocket)

@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await websocket.accept()
    clients_logs.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_logs.discard(websocket)

@app.websocket("/ws/training")
async def websocket_training(websocket: WebSocket):
    await websocket.accept()
    clients_training.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_training.discard(websocket)

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    clients_telemetry.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_telemetry.discard(websocket)

@app.get("/api/model/weights")
async def model_weights(model: str, mode: str = "weapon"):
    """Per-antenna learned CNN channel weights (L2 norms of first conv filters, normalized)."""
    from wavetrace.adapters.recognition.session import open_session
    from wavetrace.recognition.explain import cnn_channel_weights
    try:
        safe_model_path = _resolve_output_path(model)
    except ValueError:
        return {"error": "model path must be inside output/ and must not escape it"}
    try:
        session = open_session(mode, safe_model_path)
        per_antenna_weights = cnn_channel_weights(session.head)
        return {"per_antenna": per_antenna_weights.tolist() if per_antenna_weights is not None else None}
    except Exception as error:
        return {"error": str(error)}

@app.get("/api/fusion/weights")
async def fusion_weights(path: str):
    """Softmax over the saved combiner's coefficients gives per-band trust."""
    import joblib, numpy as np
    try:
        blob = joblib.load(_resolve_output_path(path))
        coefficients = blob["combiner"].coef_.ravel()
        exponentials = np.exp(coefficients - coefficients.max())
        band_weights = exponentials / exponentials.sum()
        return {"bands": blob["band_order"], "weights": [round(float(weight), 3) for weight in band_weights]}
    except Exception as error:
        return {"error": str(error)}

@app.get("/api/weapon/litmus")
async def weapon_litmus(root: str = "data", node: int | None = None, per_link: bool = False):
    """Static σ²[p] separation between clear and weapon captures, per node or per tx→rx link.
    Each row's `hist` holds histogram bins for the client's PDF overlay."""
    from wavetrace.diagnostics import gather_sigma2, json_histogram, key_label, separation, verdict
    try:
        sigma2_by_key = gather_sigma2(root, node, per_link=per_link)
        if not sigma2_by_key:
            return {"error": f"no weapon recordings under {root}/weapon_rec/*/<clear|weapon>/node*/"}

        def _compute_auc(key):
            separation_result = separation(sigma2_by_key[key].get("clear", _empty_array()),
                                           sigma2_by_key[key].get("weapon", _empty_array()))
            return separation_result["auc"] if separation_result else 0.0

        rows = []
        for key in sorted(sigma2_by_key,
                          key=lambda group_key: (-_compute_auc(group_key), key_label(group_key))):
            clear_sigma2 = sigma2_by_key[key].get("clear", _empty_array())
            weapon_sigma2 = sigma2_by_key[key].get("weapon", _empty_array())
            separation_result = separation(clear_sigma2, weapon_sigma2)
            label = key_label(key)
            if separation_result is None:
                rows.append({"label": label, "ok": False, "reason": "need both clear and weapon captures"})
                continue
            rows.append({"label": label, "auc": round(separation_result["auc"], 3),
                         "lower_when_armed": separation_result["lower_when_armed"],
                         "cohens_d": round(separation_result["cohens_d"], 2),
                         "n_clear": separation_result["n_clear"], "n_weapon": separation_result["n_weapon"],
                         "verdict": verdict(separation_result["auc"]),
                         "hist": json_histogram(clear_sigma2, weapon_sigma2)
                         if clear_sigma2.size >= 10 and weapon_sigma2.size >= 10 else None})
        return {"rows": rows, "per_link": per_link}
    except Exception as error:
        return {"error": str(error)}


@app.get("/api/calib/info")
async def calib_info(path: str = "output/calib"):
    """K is max(image_subcarriers)+1: the highest subcarrier index the radio reported during
    calibration. It pins the width the rest of the pipeline must use."""
    import json as _json
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return {"error": f"no calibration at {path} (run Calib first)"}
    try:
        with open(meta_path) as meta_file:
            meta = _json.load(meta_file)
        image_subcarriers = meta.get("image_subcarriers") or meta.get("subcarriers") or []
        subcarrier_count = int(max(image_subcarriers)) + 1 if image_subcarriers else 0
        if subcarrier_count <= 96:
            bandwidth_label = "HT20 · 2.4 GHz"
        elif subcarrier_count <= 200:
            bandwidth_label = "HT40 · 2.4 GHz"
        else:
            bandwidth_label = "HT80 · 5 GHz"
        return {
            "K": subcarrier_count,
            "bw_label": bandwidth_label,
            "n_selected": len(meta.get("subcarriers") or []),
            "n_image": len(image_subcarriers),
            "path": path,
        }
    except Exception as error:
        return {"error": str(error)}


@app.get("/api/paths/scan")
async def scan_paths():
    import glob as _glob

    def _scan():
        calibration_dirs = sorted(set(
            os.path.dirname(matched_path)
            for matched_path in _glob.glob("data/**/meta.json", recursive=True)
                       + _glob.glob("output/**/meta.json", recursive=True)
        ))
        model_files = sorted(
            _glob.glob("data/**/model.joblib", recursive=True)
            + _glob.glob("output/**/model.joblib", recursive=True)
        )
        mesh_roots = sorted(set(
            os.path.dirname(matched_path)
            for matched_path in _glob.glob("data/**/node*/model.joblib", recursive=True)
                       + _glob.glob("output/**/node*/model.joblib", recursive=True)
        ))
        dataset_dirs = sorted(set(
            os.path.dirname(matched_path)
            for matched_path in _glob.glob("data/**/X_features.npy", recursive=True)
                       + _glob.glob("output/**/X_features.npy", recursive=True)
        ))
        pooled_dataset_dirs = sorted(set(
            os.path.dirname(os.path.dirname(matched_path))
            for matched_path in _glob.glob("data/**/X_features.npy", recursive=True)
                       + _glob.glob("output/**/X_features.npy", recursive=True)
            if os.path.basename(os.path.dirname(matched_path)) not in (".", "")
        ))
        return {
            "calibrations": calibration_dirs,
            "models": model_files + [mesh_root for mesh_root in mesh_roots if mesh_root not in model_files],
            "datasets": dataset_dirs + [pooled_dir for pooled_dir in pooled_dataset_dirs
                                        if pooled_dir not in dataset_dirs],
        }

    return await asyncio.to_thread(_scan)


@app.get("/api/paths/browse")
async def browse_path(type: str = "dir", prompt: str = "Select path", ext: str = ""):
    """macOS only: the dialog comes from osascript.
    Returns {"path": "/abs/path"}, or {"path": null} when the user cancels."""
    import subprocess as _subprocess

    def _open_dialog():
        if type == "dir":
            script = f'POSIX path of (choose folder with prompt "{prompt}")'
        else:
            if ext:
                allowed_extensions = ("{" + ", ".join(f'"{extension.strip()}"'
                                                                     for extension in ext.split(","))
                                      + "}")
                script = (f'POSIX path of (choose file with prompt "{prompt}" '
                          f'of type {allowed_extensions})')
            else:
                script = f'POSIX path of (choose file with prompt "{prompt}")'

        try:
            result = _subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {"path": None, "cancelled": True}
            # osascript returns the path with a trailing newline
            chosen = result.stdout.strip().rstrip("/")
            return {"path": chosen}
        except _subprocess.TimeoutExpired:
            return {"path": None, "cancelled": True}
        except FileNotFoundError:
            return {"path": None, "error": "osascript not found — macOS only"}
        except Exception as error:
            return {"path": None, "error": str(error)}

    return await asyncio.to_thread(_open_dialog)


def _empty_array():
    import numpy as np
    return np.array([])


class ModelUploadRequest(BaseModel):
    file_b64: str
    dest: str = "output/model.pkl/model.joblib"

@app.post("/api/model/upload")
async def model_upload(request: ModelUploadRequest):
    import base64
    try:
        destination_path = _resolve_output_path(request.dest)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as model_file:
            model_file.write(base64.b64decode(request.file_b64))
        return {"status": "uploaded", "dest": destination_path}
    except Exception as error:
        return {"error": str(error)}

class MonitorRequest(BaseModel):
    port: str
    baud: int = 115200

class FlashRequest(BaseModel):
    role: str = "node"          # node | rx | tx
    node_id: int | None = None
    port: str
    clean: bool = False

class PiRequest(BaseModel):
    host: str                   # user@host
    command: str

class ScriptRequest(BaseModel):
    script: str
    args: str = ""

class StopMonitorRequest(BaseModel):
    port: str | None = None

@app.get("/api/serial/ports")
async def serial_ports():
    return {"ports": list_serial_ports()}

@app.get("/api/device/state")
async def device_state():
    return device_hub.get_state()

class SerialMonitorRequest(BaseModel):
    port: str
    baud: int = 115200

@app.post("/api/serial/monitor/start")
async def serial_monitor_start(request: SerialMonitorRequest):
    return device_hub.start_monitor(request.port, request.baud)

@app.post("/api/serial/monitor/stop")
async def serial_monitor_stop(request: StopMonitorRequest = None):
    port = request.port if request else None
    return device_hub.stop_monitor(port)

@app.post("/api/flash")
async def flash(request: FlashRequest):
    # the build-and-flash blocks, so it runs on a worker thread and the event loop keeps streaming
    asyncio.create_task(asyncio.to_thread(device_hub.flash, request.role, request.node_id, request.port,
                                          request.clean))
    return {"status": "flashing", "role": request.role, "port": request.port, "clean": request.clean}

@app.post("/api/pi/run")
async def pi_run(request: PiRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.run_on_pi, request.host, request.command))
    return {"status": "running", "host": request.host}

@app.post("/api/script/run")
async def script_run(request: ScriptRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.run_script, request.script, request.args))
    return {"status": "running", "script": request.script}



class StopProcRequest(BaseModel):
    proc_id: str | None = None

@app.post("/api/device/stop")
async def device_stop(request: StopProcRequest = None):
    process_id = request.proc_id if request else None
    return device_hub.stop_process(process_id)

class InputRequest(BaseModel):
    proc_id: str
    input: str

@app.post("/api/device/input")
async def device_input(request: InputRequest):
    return device_hub.send_input(request.proc_id, request.input)

@app.websocket("/ws/device")
async def websocket_device(websocket: WebSocket):
    await websocket.accept()
    clients_device.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_device.discard(websocket)


import threading as _threading
_yolo_cache: dict = {}
_yolo_lock = _threading.Lock()


def _load_yolo(weights: str = "yolov8n-seg.pt"):
    with _yolo_lock:
        if weights not in _yolo_cache:
            try:
                from ultralytics import YOLO
                _yolo_cache[weights] = YOLO(weights)
            except Exception as error:
                _yolo_cache[weights] = None   # cache the failure so we don't retry on every frame
                print(f"[YOLO] load failed: {error}")
    return _yolo_cache.get(weights)


def _annotate_frame(model, frame, weapon_class_ids=(43,)):
    import cv2, numpy as np
    results = model(frame, verbose=False)
    annotated_frame = frame.copy()
    for result in results:
        boxes = result.boxes
        masks = result.masks
        for i, detection_box in enumerate(boxes):
            class_id = int(detection_box.cls[0])
            confidence = float(detection_box.conf[0])
            is_weapon = class_id in weapon_class_ids
            color = (30, 120, 255) if is_weapon else (50, 220, 80)   # BGR: orange / green
            label = f"{'WEAPON' if is_weapon else model.names.get(class_id, str(class_id))} {confidence:.0%}"
            if masks is not None and i < len(masks.xy):
                mask_polygon_px = masks.xy[i].astype(np.int32)
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [mask_polygon_px], color)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.55, overlay, 0.45, 0)
                cv2.polylines(annotated_frame, [mask_polygon_px], True, color, 2)
            left_px, top_px, right_px, bottom_px = map(int, detection_box.xyxy[0])
            cv2.rectangle(annotated_frame, (left_px, top_px), (right_px, bottom_px), color, 1)
            (text_width_px, text_height_px), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated_frame, (left_px, top_px - text_height_px - 4),
                          (left_px + text_width_px + 2, top_px), color, -1)
            cv2.putText(annotated_frame, label, (left_px + 1, top_px - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated_frame


# Frames come from an ffmpeg subprocess, not cv2.VideoCapture: on macOS, AVFoundation's run loop
# segfaults on a background thread. cv2 is used only for YOLO annotation.

import subprocess as _subprocess
import shutil as _shutil


def _find_ffmpeg() -> str:
    ffmpeg_path = _shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg not found — install it: brew install ffmpeg"
        )
    return ffmpeg_path


@app.get("/api/camera/check")
async def camera_check(cam_index: int = 0):
    def _probe():
        try:
            ffmpeg = _find_ffmpeg()
        except RuntimeError as error:
            return {"ok": False, "error": str(error)}
        ffmpeg_command = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
            "-i", f"{cam_index}:none",
            "-vframes", "1", "-f", "rawvideo", "-vcodec", "rawvideo", "-",
        ]
        try:
            result = _subprocess.run(ffmpeg_command, capture_output=True, timeout=10)
            if result.returncode != 0 or not result.stdout:
                stderr_tail = result.stderr.decode(errors="replace")[-400:]
                if "permission" in stderr_tail.lower() or "authorization" in stderr_tail.lower():
                    return {"ok": False,
                            "error": "Camera permission denied — allow Terminal in System Settings → Privacy → Camera"}
                return {"ok": False, "error": f"ffmpeg exit {result.returncode}: {stderr_tail}"}
            return {"ok": True, "width": 1280, "height": 720, "cam_index": cam_index}
        except _subprocess.TimeoutExpired:
            return {"ok": False, "error": "Camera probe timed out"}
        except Exception as error:
            return {"ok": False, "error": str(error)}
    return await asyncio.to_thread(_probe)


_camera_active = False

@app.post("/api/camera/stop")
def stop_camera():
    global _camera_active
    _camera_active = False
    return {"ok": True}


@app.get("/api/camera/stream")
async def camera_stream(request: Request, index: int = 0, annotate: bool = False,
                        weights: str = "yolov8n-seg.pt"):
    """ffmpeg's mjpeg output is one continuous byte stream, so frames are cut at the JPEG
    SOI/EOI markers and sent as multipart/x-mixed-replace chunks."""
    global _camera_active
    _camera_active = True
    
    model = await asyncio.to_thread(_load_yolo, weights) if annotate else None

    async def _generate():
        try:
            ffmpeg = _find_ffmpeg()
        except RuntimeError:
            return  # ffmpeg missing: an empty body makes the browser's <img> fire onError

        ffmpeg_command = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", f"{index}:none",
            "-f", "mjpeg",
            "-q:v", "5",
            "-",
        ]

        ffmpeg_process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        unparsed_bytes = b""
        frame_index = 0

        try:
            while _camera_active:
                if await request.is_disconnected():
                    break

                chunk = await ffmpeg_process.stdout.read(65536)
                if not chunk:
                    break
                unparsed_bytes += chunk
                while True:
                    frame_start_byte = unparsed_bytes.find(JPEG_START_OF_IMAGE)
                    if frame_start_byte == -1:
                        unparsed_bytes = b""
                        break
                    frame_end_byte = unparsed_bytes.find(JPEG_END_OF_IMAGE, frame_start_byte + 2)
                    if frame_end_byte == -1:
                        # no EOI yet: keep the partial frame and read more
                        unparsed_bytes = unparsed_bytes[frame_start_byte:]
                        break
                    jpeg_frame = unparsed_bytes[frame_start_byte: frame_end_byte + 2]
                    unparsed_bytes = unparsed_bytes[frame_end_byte + 2:]

                    frame_index += 1

                    if model is not None:
                        # throttle YOLO to 5 Hz: 1 frame in every 6 of the 30 Hz source
                        if frame_index % 6 != 0:
                            continue

                        import cv2, numpy as np
                        encoded_jpeg = np.frombuffer(jpeg_frame, dtype=np.uint8)
                        frame = cv2.imdecode(encoded_jpeg, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = await asyncio.to_thread(_annotate_frame, model, frame)
                            _, annotated_jpeg = cv2.imencode(".jpg", frame,
                                                             [cv2.IMWRITE_JPEG_QUALITY, 75])
                            jpeg_frame = annotated_jpeg.tobytes()

                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg_frame)).encode()
                           + b"\r\n\r\n" + jpeg_frame + b"\r\n")
        finally:
            try:
                ffmpeg_process.kill()
            except ProcessLookupError:
                pass

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


app.mount("/", StaticFiles(directory="web/ui/dist", html=True), name="static")

if __name__ == "__main__":
    server_options = WebServerOptions()
    uvicorn.run("web.app:app", host=server_options.host, port=server_options.port,
                reload=server_options.reload,
                reload_dirs=list(server_options.reload_dirs),        # uvicorn wants list, not tuple
                reload_includes=list(server_options.reload_includes))
