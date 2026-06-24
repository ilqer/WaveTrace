from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
import os
import json
import uvicorn

from wavetrace.Cli import _source_from_args, _parse_spans
from web.streamer import WaveTraceRunner
from web.foxglove import fg_server
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


# Model load/write endpoints below confine paths to output/ — joblib.load is pickle (RCE) and an
# unrestricted dest is an arbitrary file write, so reject absolute paths and any ".." escape.
ALLOWED_ROOT = os.path.realpath("output")

def _safe_output_path(path: str) -> str:
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
    calibration: str = "output/calib"
    model: str = "output/model.pkl/model.joblib"
    gain_lock: bool = True
    vote: bool = True
    frame_average: int = 1
    use_baseline: bool = False
    
    # Calib
    baseline_packets: int = 300
    cal_out: str = "output/calib"
    
    # Collect
    col_stage: str = "presence"
    col_spans: str = "0:5,10:15,20:25"
    col_window: int = 128
    col_hop: int = 32
    subtract_ic_baseline: bool = True  # weapon IC background subtraction (Item 10/CAUSE 2B) — default ON
    
    # Train
    train_backend: str = "mlp"
    train_out: str = "output/model.pkl"
    train_data: str = "output/dataset_ui"  # dataset dir or cumulative pool parent (globs node*/)

    # Hardware
    cam_url: str = "http://192.168.1.100/mjpeg"

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
async def start_inference(req: StartRequest):
    global runner, runner_task, inference_queue, stream_queue, logs_queue
    
    if runner and runner.is_running:
        runner.stop()
        await asyncio.sleep(0.5)

    loop = asyncio.get_running_loop()
    runner = WaveTraceRunner(loop, inference_queue, stream_queue, logs_queue, training_queue,
                             telemetry_queue)
    
    def run_blocking():
        try:
            if req.action == "run":
                runner.start_inference_managed(req)
            elif req.action == "calib":
                runner.start_calibration_managed(req)
            elif req.action == "collect":
                runner.start_collection_managed(req)
            elif req.action == "train":
                runner.start_training_managed(req)
        except Exception as e:
            loop.call_soon_threadsafe(logs_queue.put_nowait, f"FATAL ERROR: {str(e)}")

    # run_blocking is a blocking pipeline loop; hand it to a worker thread and keep the real handle
    # (add_task returns None) so stop_inference can join it after runner.stop() flips the stop flag.
    runner_task = asyncio.create_task(asyncio.to_thread(run_blocking))
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
    from wavetrace.recognition import mode_session
    from wavetrace.recognition.Explain import cnn_channel_weights
    try:
        sess = mode_session(mode, model)
        w = cnn_channel_weights(sess.head)
        return {"per_antenna": w.tolist() if w is not None else None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/fusion/weights")
async def fusion_weights(path: str):
    """Learned per-band trust from a saved BandFusion model."""
    import joblib, numpy as np
    try:
        blob = joblib.load(_safe_output_path(path))
        coef = blob["combiner"].coef_.ravel()
        ex = np.exp(coef - coef.max()); w = ex / ex.sum()
        return {"bands": blob["band_order"], "weights": [round(float(x), 3) for x in w]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/weapon/litmus")
async def weapon_litmus(root: str = "data", node: int | None = None, per_link: bool = False):
    """Static σ²[p] go/no-go: per-node (default) or per directed tx→rx link (per_link=true).
    Rows are sorted by AUC descending. Each row includes histogram bins for the PDF overlay."""
    from weapon_litmus import gather_sigma2, separation, _verdict, _key_label, json_hist
    try:
        data = gather_sigma2(root, node, per_link=per_link)
        if not data:
            return {"error": f"no weapon recordings under {root}/weapon_rec/*/<clear|weapon>/node*/"}

        def _auc_of(key):
            s = separation(data[key].get("clear", _np_empty()), data[key].get("weapon", _np_empty()))
            return s["auc"] if s else 0.0

        out = []
        for key in sorted(data, key=lambda k: (-_auc_of(k), _key_label(k))):
            c = data[key].get("clear", _np_empty())
            w = data[key].get("weapon", _np_empty())
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
    """Read a saved calibration directory and return the pinned subcarrier width.
    K is derived from max(image_subcarriers)+1 — the highest subcarrier index the
    radio produced during calibration. bw_label maps that to HT20/HT40/HT80."""
    import json as _json
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return {"error": f"no calibration at {path} (run Calib first)"}
    try:
        with open(meta_path) as f:
            meta = _json.load(f)
        img_subc = meta.get("image_subcarriers") or meta.get("subcarriers") or []
        K = int(max(img_subc)) + 1 if img_subc else 0
        if K <= 96:
            bw_label = "HT20 · 2.4 GHz"
        elif K <= 200:
            bw_label = "HT40 · 2.4 GHz"
        else:
            bw_label = "HT80 · 5 GHz"
        return {
            "K": K,
            "bw_label": bw_label,
            "n_selected": len(meta.get("subcarriers") or []),
            "n_image": len(img_subc),
            "path": path,
        }
    except Exception as e:
        return {"error": str(e)}


def _np_empty():
    import numpy as np
    return np.array([])


class ModelUploadRequest(BaseModel):
    file_b64: str
    dest: str = "output/model.pkl/model.joblib"

@app.post("/api/model/upload")
async def model_upload(req: ModelUploadRequest):
    """Receive a PC-trained model.joblib (base64) and write it to the Pi."""
    import base64
    try:
        dest = _safe_output_path(req.dest)
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
    return {"ports": list_serial_ports()}

@app.get("/api/device/state")
async def device_state():
    return device_hub.get_state()

class SerialMonitorRequest(BaseModel):
    port: str
    baud: int = 115200

@app.post("/api/serial/monitor/start")
async def serial_monitor_start(req: SerialMonitorRequest):
    return device_hub.start_monitor(req.port, req.baud)

@app.post("/api/serial/monitor/stop")
async def serial_monitor_stop(req: StopMonitorRequest = None):
    p = req.port if req else None
    return device_hub.stop_monitor(p)

@app.post("/api/flash")
async def flash(req: FlashRequest):
    # flashing blocks (build+flash); run in a worker thread so the event loop keeps streaming
    asyncio.create_task(asyncio.to_thread(device_hub.flash, req.role, req.node_id, req.port, req.clean))
    return {"status": "flashing", "role": req.role, "port": req.port, "clean": req.clean}

@app.post("/api/pi/run")
async def pi_run(req: PiRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.run_pi, req.host, req.command))
    return {"status": "running", "host": req.host}

@app.post("/api/script/run")
async def script_run(req: ScriptRequest):
    asyncio.create_task(asyncio.to_thread(device_hub.run_script, req.script, req.args))
    return {"status": "running", "script": req.script}



class StopProcRequest(BaseModel):
    proc_id: str | None = None

@app.post("/api/device/stop")
async def device_stop(req: StopProcRequest = None):
    p = req.proc_id if req else None
    return device_hub.stop_proc(p)

class InputRequest(BaseModel):
    proc_id: str
    input: str

@app.post("/api/device/input")
async def device_input(req: InputRequest):
    return device_hub.send_input(req.proc_id, req.input)

@app.websocket("/ws/device")
async def websocket_device(websocket: WebSocket):
    await websocket.accept()
    clients_device.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients_device.discard(websocket)


app.mount("/", StaticFiles(directory="web/ui/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
