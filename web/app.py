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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_queue, stream_queue, logs_queue, training_queue, telemetry_queue
    inference_queue = asyncio.Queue()
    stream_queue = asyncio.Queue()
    logs_queue = asyncio.Queue()
    training_queue = asyncio.Queue()
    telemetry_queue = asyncio.Queue()
    await fg_server.start()
    asyncio.create_task(broadcast_inference())
    asyncio.create_task(broadcast_stream())
    asyncio.create_task(broadcast_logs())
    asyncio.create_task(broadcast_training())
    asyncio.create_task(broadcast_telemetry())
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
    
    # Run
    mode: str = "presence"
    calibration: str = "output/calib"
    model: str = "output/model.pkl/model.joblib"
    gain_lock: bool = True
    vote: bool = False
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
    
    # Train
    train_backend: str = "mlp"
    train_out: str = "output/model.pkl"

    # Hardware
    cam_url: str = "http://192.168.1.100/mjpeg"
    nodes: str = "192.168.1.101, 192.168.1.102"

class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

clients_inference = set()
clients_stream = set()
clients_logs = set()
clients_training = set()
clients_telemetry = set()

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

@app.post("/api/model/upload")
async def model_upload(file_b64: str, dest: str = "output/model.pkl/model.joblib"):
    """Receive a PC-trained model.joblib (base64) and write it to the Pi."""
    import base64
    try:
        dest = _safe_output_path(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(base64.b64decode(file_b64))
        return {"status": "uploaded", "dest": dest}
    except Exception as e:
        return {"error": str(e)}

app.mount("/", StaticFiles(directory="web/ui/dist", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web.app:app", host="0.0.0.0", port=8000, reload=True)
