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

from wavetrace.Cli import _sourceFromArgs, _parseSpans
from web.streamer import WaveTraceRunner
from web.foxglove import fg_server
from web.device_ctl import DeviceHub, listSerialPorts


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
    await fg_server.start()
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


# Confine arbitrary file writes and pickle (RCE) via joblib.load to output/ dir.
ALLOWED_ROOT = os.path.realpath("output")

def _safeOutputPath(path: str) -> str:
    full = os.path.realpath(path)
    if os.path.commonpath([full, ALLOWED_ROOT]) != ALLOWED_ROOT:
        raise ValueError(f"path escapes output/: {path}")
    return full


app = FastAPI(title="WaveTrace Lab Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared State
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
    # Action
    action: str = "run" # run | calib | collect | train
    
    # Source
    synthetic: bool = False
    antennas: int = 2
    subcarriers: int = 64
    fs: float = 100.0
    duration: float = 60.0
    seed: int = 0
    udp_port: int = 9876  # MUST match the firmware/run_* port (nodes push CSI to 9876)
    
    # Run
    mode: str = "presence"
    calibration: str = "data/2g4_ht40/ui/cal"
    model: str = "data/2g4_ht40/ui/model/model.joblib"
    gainLock: bool = True
    vote: bool = True
    frame_average: int = 1
    use_baseline: bool = False
    
    # Calib
    baseline_packets: int = 300
    cal_out: str = "data/2g4_ht40/ui/cal"
    
    # Collect
    col_stage: str = "presence"
    col_spans: str = "0:5,10:15,20:25"
    col_window: int = 128
    col_hop: int = 32
    subtract_ic_baseline: bool = True  # weapon IC background subtraction (Item 10/CAUSE 2B) — default ON
    
    # Train
    train_backend: str = "mlp"
    train_out: str = "data/2g4_ht40/ui/model"
    train_data: str = "data/2g4_ht40/ui/ds"  # dataset dir or cumulative pool parent (globs node*/)

    # Hardware
    cam_url: str = "/api/camera/stream"
    cam_index: int = 0
    per_link: bool = False
    yolo_weights: str = "yolov8n-seg.pt"

class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

clients_inference = set()
clients_stream = set()
clients_logs = set()
clients_training = set()
clients_telemetry = set()
clients_device = set()

async def broadcast_inference():
    while True:
        data = await inference_queue.get()
        await fg_server.publish_inference(json.loads(data))
        for client in list(clients_inference):
            try:
                await client.send_text(data)
            except Exception:
                clients_inference.discard(client)

async def broadcast_stream():
    while True:
        data = await stream_queue.get()
        await fg_server.publish_stream(json.loads(data))
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
async def startInference(req: StartRequest):
    global runner, runner_task, inference_queue, stream_queue, logs_queue
    
    if runner and runner.is_running:
        runner.stop()
        await asyncio.sleep(0.5)

    loop = asyncio.get_running_loop()
    runner = WaveTraceRunner(loop, inference_queue, stream_queue, logs_queue, training_queue,
                             telemetry_queue)
    
    def runBlocking():
        try:
            if req.action == "run":
                runner.startInferenceManaged(req)
            elif req.action == "calib":
                runner.startCalibrationManaged(req)
            elif req.action == "collect":
                runner.startCollectionManaged(req)
            elif req.action == "train":
                runner.startTrainingManaged(req)
            elif req.action == "camera_collect":
                runner.startCameraCollectManaged(req)
        except Exception as e:
            loop.call_soon_threadsafe(logs_queue.put_nowait, f"FATAL ERROR: {str(e)}")

    # runBlocking is blocking; run it in a worker thread so stop_inference can join runner_task later.
    runner_task = asyncio.create_task(asyncio.to_thread(runBlocking))
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
    from wavetrace.recognition import modeSession
    from wavetrace.recognition.Explain import cnnChannelWeights
    try:
        safeModel = _safeOutputPath(model)
    except ValueError:
        return {"error": "model path must be inside output/ and must not escape it"}
    try:
        sess = modeSession(mode, safeModel)
        w = cnnChannelWeights(sess.head)
        return {"per_antenna": w.tolist() if w is not None else None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/fusion/weights")
async def fusion_weights(path: str):
    """Learned per-band trust from a saved BandFusion model."""
    import joblib, numpy as np
    try:
        blob = joblib.load(_safeOutputPath(path))
        coef = blob["combiner"].coef_.ravel()
        ex = np.exp(coef - coef.max()); w = ex / ex.sum()
        return {"bands": blob["band_order"], "weights": [round(float(x), 3) for x in w]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/weapon/litmus")
async def weapon_litmus(root: str = "data", node: int | None = None, per_link: bool = False):
    """Static σ²[p] check: per-node (default) or tx→rx link (per_link=true).
    Rows sorted by AUC desc. Includes histogram bins for PDF overlay."""
    from experiments.weapon_litmus import gather_sigma2, separation, _verdict, _key_label, json_hist
    try:
        data = gather_sigma2(root, node, per_link=per_link)
        if not data:
            return {"error": f"no weapon recordings under {root}/weapon_rec/*/<clear|weapon>/node*/"}

        def _aucOf(key):
            s = separation(data[key].get("clear", _npEmpty()), data[key].get("weapon", _npEmpty()))
            return s["auc"] if s else 0.0

        out = []
        for key in sorted(data, key=lambda k: (-_aucOf(k), _key_label(k))):
            c = data[key].get("clear", _npEmpty())
            w = data[key].get("weapon", _npEmpty())
            s = separation(c, w)
            label = _key_label(key)
            if s is None:
                out.append({"label": label, "ok": False, "reason": "need both clear and weapon captures"})
                continue
            out.append({"label": label, "auc": round(s["auc"], 3),
                        "lower_when_armed": s["lower_when_armed"], "cohens_d": round(s["cohens_d"], 2),
                        "n_clear": s["n_clear"], "n_weapon": s["n_weapon"],
                        "verdict": _verdict(s["auc"]),
                        "hist": json_hist(c, w) if c.size >= 10 and w.size >= 10 else None})
        return {"rows": out, "per_link": per_link}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/calib/info")
async def calib_info(path: str = "output/calib"):
    """Reads saved calib for pinned subcarrier width.
    K is max(image_subcarriers)+1 (highest index from radio during calib).
    bw_label maps K to HT20/HT40/HT80."""
    import json as _json
    metaPath = os.path.join(path, "meta.json")
    if not os.path.exists(metaPath):
        return {"error": f"no calibration at {path} (run Calib first)"}
    try:
        with open(metaPath) as f:
            meta = _json.load(f)
        imgSubc = meta.get("image_subcarriers") or meta.get("subcarriers") or []
        K = int(max(imgSubc)) + 1 if imgSubc else 0
        if K <= 96:
            bwLabel = "HT20 · 2.4 GHz"
        elif K <= 200:
            bwLabel = "HT40 · 2.4 GHz"
        else:
            bwLabel = "HT80 · 5 GHz"
        return {
            "K": K,
            "bw_label": bwLabel,
            "n_selected": len(meta.get("subcarriers") or []),
            "n_image": len(imgSubc),
            "path": path,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/paths/scan")
async def scan_paths():
    """Scans project for calib dirs, models, and datasets. Populates UI path-pickers."""
    import glob as _glob

    def _scan():
        calDirs = sorted(set(
            os.path.dirname(p)
            for p in _glob.glob("data/**/meta.json", recursive=True)
                       + _glob.glob("output/**/meta.json", recursive=True)
        ))
        modelFiles = sorted(
            _glob.glob("data/**/model.joblib", recursive=True)
            + _glob.glob("output/**/model.joblib", recursive=True)
        )
        meshRoots = sorted(set(
            os.path.dirname(p)
            for p in _glob.glob("data/**/node*/model.joblib", recursive=True)
                       + _glob.glob("output/**/node*/model.joblib", recursive=True)
        ))
        datasetDirs = sorted(set(
            os.path.dirname(p)
            for p in _glob.glob("data/**/X_features.npy", recursive=True)
                       + _glob.glob("output/**/X_features.npy", recursive=True)
        ))
        # Parent dirs of multiple dataset subdirs (cumulative pool roots)
        poolDirs = sorted(set(
            os.path.dirname(os.path.dirname(p))
            for p in _glob.glob("data/**/X_features.npy", recursive=True)
                       + _glob.glob("output/**/X_features.npy", recursive=True)
            if os.path.basename(os.path.dirname(p)) not in (".", "")
        ))
        return {
            "calibrations": calDirs,
            "models": modelFiles + [r for r in meshRoots if r not in modelFiles],
            "datasets": datasetDirs + [d for d in poolDirs if d not in datasetDirs],
        }

    return await asyncio.to_thread(_scan)


@app.get("/api/paths/browse")
async def browse_path(type: str = "dir", prompt: str = "Select path", ext: str = ""):
    """Opens macOS Finder dialog (osascript) and returns chosen path.
    type: 'dir' or 'file'. ext: csv extensions (file mode).
    Returns {"path": "/abs/path"} or {"path": null} on cancel."""
    import subprocess as _sp

    def _openDialog():
        if type == "dir":
            script = f'POSIX path of (choose folder with prompt "{prompt}")'
        else:
            if ext:
                extList = "{" + ", ".join(f'"{e.strip()}"' for e in ext.split(",")) + "}"
                script = (f'POSIX path of (choose file with prompt "{prompt}" '
                          f'of type {extList})')
            else:
                script = f'POSIX path of (choose file with prompt "{prompt}")'

        try:
            result = _sp.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {"path": None, "cancelled": True}
            # osascript returns path with trailing newline; strip it
            chosen = result.stdout.strip().rstrip("/")
            return {"path": chosen}
        except _sp.TimeoutExpired:
            return {"path": None, "cancelled": True}
        except FileNotFoundError:
            return {"path": None, "error": "osascript not found — macOS only"}
        except Exception as e:
            return {"path": None, "error": str(e)}

    return await asyncio.to_thread(_openDialog)


def _npEmpty():
    import numpy as np
    return np.array([])


class ModelUploadRequest(BaseModel):
    file_b64: str
    dest: str = "output/model.pkl/model.joblib"

@app.post("/api/model/upload")
async def model_upload(req: ModelUploadRequest):
    """Receives base64 PC-trained model.joblib and writes to Pi."""
    import base64
    try:
        dest = _safeOutputPath(req.dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(base64.b64decode(req.file_b64))
        return {"status": "uploaded", "dest": dest}
    except Exception as e:
        return {"error": str(e)}

# ---- Hardware: serial discovery / monitor, flashing, Pi capture control --------------
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
    return {"ports": listSerialPorts()}

@app.get("/api/device/state")
async def device_state():
    return device_hub.getState()

class SerialMonitorRequest(BaseModel):
    port: str
    baud: int = 115200

@app.post("/api/serial/monitor/start")
async def serial_monitor_start(req: SerialMonitorRequest):
    return device_hub.startMonitor(req.port, req.baud)

@app.post("/api/serial/monitor/stop")
async def serial_monitor_stop(req: StopMonitorRequest = None):
    p = req.port if req else None
    return device_hub.stopMonitor(p)

@app.post("/api/flash")
async def flash(req: FlashRequest):
    # flashing blocks (build+flash); run in a worker thread so the event loop keeps streaming
    asyncio.create_task(asyncio.to_thread(device_hub.flash, req.role, req.node_id, req.port, req.clean))
    return {"status": "flashing", "role": req.role, "port": req.port, "clean": req.clean}

@app.post("/api/pi/run")
async def pi_run(req: PiRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.runPi, req.host, req.command))
    return {"status": "running", "host": req.host}

@app.post("/api/script/run")
async def script_run(req: ScriptRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.runScript, req.script, req.args))
    return {"status": "running", "script": req.script}



class StopProcRequest(BaseModel):
    proc_id: str | None = None

@app.post("/api/device/stop")
async def device_stop(req: StopProcRequest = None):
    p = req.proc_id if req else None
    return device_hub.stopProc(p)

class InputRequest(BaseModel):
    proc_id: str
    input: str

@app.post("/api/device/input")
async def device_input(req: InputRequest):
    return device_hub.sendInput(req.proc_id, req.input)

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


def _loadYolo(weights: str = "yolov8n-seg.pt"):
    """Thread-safely loads and caches YOLO model."""
    with _yolo_lock:
        if weights not in _yolo_cache:
            try:
                from ultralytics import YOLO
                _yolo_cache[weights] = YOLO(weights)
            except Exception as e:
                _yolo_cache[weights] = None   # cache failure so we don't retry every frame
                print(f"[YOLO] load failed: {e}")
    return _yolo_cache.get(weights)


def _annotateFrame(model, frame, weapon_classes=(43,)):
    """Draws YOLO seg masks and labels on frame copy (Green = person, orange = weapon)."""
    import cv2, numpy as np
    results = model(frame, verbose=False)
    out = frame.copy()
    for r in results:
        boxes = r.boxes
        masks = r.masks
        for i, box in enumerate(boxes):
            clsId = int(box.cls[0])
            conf = float(box.conf[0])
            isWeapon = clsId in weapon_classes
            color = (30, 120, 255) if isWeapon else (50, 220, 80)   # BGR: orange / green
            label = f"{'WEAPON' if isWeapon else model.names.get(clsId, str(clsId))} {conf:.0%}"
            if masks is not None and i < len(masks.xy):
                pts = masks.xy[i].astype(np.int32)
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], color)
                out = cv2.addWeighted(out, 0.55, overlay, 0.45, 0)
                cv2.polylines(out, [pts], True, color, 2)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# Use ffmpeg subprocess (not cv2.VideoCapture) to avoid macOS AVFoundation run-loop segfault on background threads. cv2 is only for YOLO annotation.

import subprocess as _subprocess
import shutil as _shutil


def _ffmpegBin() -> str:
    """Return the ffmpeg executable path, or raise RuntimeError."""
    p = _shutil.which("ffmpeg")
    if p is None:
        raise RuntimeError(
            "ffmpeg not found — install it: brew install ffmpeg"
        )
    return p


def _ffmpegGrabOne(index: int) -> bytes | None:
    """Captures one JPEG frame from camera `index` via ffmpeg. Returns raw bytes or None.
    macOS: ffmpeg uses AVFoundation, triggering permission dialog on first run (no Terminal grant needed)."""
    try:
        ffmpeg = _ffmpegBin()
    except RuntimeError:
        return None
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "avfoundation",
        "-framerate", "30",
        "-video_size", "1280x720",
        "-i", f"{index}:none",
        "-vframes", "1",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "-",
    ]
    try:
        result = _subprocess.run(cmd, capture_output=True, timeout=10)
        return result.stdout if result.returncode == 0 and result.stdout else None
    except Exception:
        return None


@app.get("/api/camera/check")
async def camera_check(cam_index: int = 0):
    """One-frame probe via ffmpeg: checks camera access and returns resolution."""
    def _probe():
        try:
            ffmpeg = _ffmpegBin()
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}
        import json as _json
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-framerate", "30", "-video_size", "1280x720",
            "-i", f"{cam_index}:none",
            "-vframes", "1", "-f", "rawvideo", "-vcodec", "rawvideo", "-",
        ]
        try:
            r = _subprocess.run(cmd, capture_output=True, timeout=10)
            if r.returncode != 0 or not r.stdout:
                stderr = r.stderr.decode(errors="replace")[-400:]
                if "permission" in stderr.lower() or "authorization" in stderr.lower():
                    return {"ok": False,
                            "error": "Camera permission denied — allow Terminal in System Settings → Privacy → Camera"}
                return {"ok": False, "error": f"ffmpeg exit {r.returncode}: {stderr}"}
            return {"ok": True, "width": 1280, "height": 720, "cam_index": cam_index}
        except _subprocess.TimeoutExpired:
            return {"ok": False, "error": "Camera probe timed out"}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return await asyncio.to_thread(_probe)


_camera_active = False

@app.post("/api/camera/stop")
def cameraStop():
    global _camera_active
    _camera_active = False
    return {"ok": True}


@app.get("/api/camera/stream")
async def camera_stream(request: Request, index: int = 0, annotate: bool = False,
                        weights: str = "yolov8n-seg.pt"):
    """Webcam MJPEG stream via asyncio subprocess (no threads/queues).
    Parses ffmpeg stdout for JPEG SOI/EOI markers for multipart chunks.
    annotate=true overlays YOLO seg masks (cv2 decode/encode only)."""
    global _camera_active
    _camera_active = True
    
    model = await asyncio.to_thread(_loadYolo, weights) if annotate else None

    async def _generate():
        try:
            ffmpeg = _ffmpegBin()
        except RuntimeError:
            return  # ffmpeg not found — browser <img> fires onError

        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", "1280x720",
            "-i", f"{index}:none",
            "-f", "mjpeg",
            "-q:v", "5",
            "-",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        SOI = b"\xff\xd8"
        EOI = b"\xff\xd9"
        buf = b""
        frameIdx = 0

        try:
            while _camera_active:
                if await request.is_disconnected():
                    break
                    
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(SOI)
                    if s == -1:
                        buf = b""
                        break
                    e = buf.find(EOI, s + 2)
                    if e == -1:
                        buf = buf[s:]   # keep partial frame, wait for more data
                        break
                    jpg = buf[s: e + 2]
                    buf = buf[e + 2:]
                    
                    frameIdx += 1

                    if model is not None:
                        # Throttle YOLO to 5fps (1 out of every 6 frames from 30fps source)
                        if frameIdx % 6 != 0:
                            continue
                            
                        import cv2, numpy as np
                        arr = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = await asyncio.to_thread(_annotateFrame, model, frame)
                            _, enc = cv2.imencode(".jpg", frame,
                                                  [cv2.IMWRITE_JPEG_QUALITY, 75])
                            jpg = enc.tobytes()

                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode()
                           + b"\r\n\r\n" + jpg + b"\r\n")
        finally:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    return StreamingResponse(_generate(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


app.mount("/", StaticFiles(directory="web/ui/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True,
                reload_dirs=["web", "wavetrace"],
                reload_includes=["*.py"])
