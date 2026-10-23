import asyncio

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

from web.dashboard_state import Dashboard
from web.streamer import WaveTraceRunner

router = APIRouter()


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

    train_backend: str = "mlp"  # 'heatmap' trains the heatmap head instead of the stage's heads
    train_out: str = "data/2g4_ht40/ui/model"  # ignored: heads go to the collect scripts' default model roots
    # a capture-profile root whose stage pool is trained; for 'heatmap', one <root>/cam_ds/heatmap/<session> dir
    train_data: str = "data/2g4_ht40/ui/ds"

    cam_url: str = "/api/camera/stream"
    cam_index: int = 0
    camera_collect: bool = False  # train: presence trains the camera pool cam_ds/presence; count has none
    per_link: bool = False
    yolo_weights: str = "yolov8n-seg.pt"


@router.post("/api/action/start")
async def start_action(request: StartRequest, dashboard: Dashboard):
    if dashboard.runner and dashboard.runner.is_running:
        dashboard.runner.stop()
        await asyncio.sleep(0.5)

    loop = asyncio.get_running_loop()
    dashboard.runner = WaveTraceRunner(loop, dashboard.inference.queue, dashboard.stream.queue,
                                       dashboard.logs.queue, dashboard.training.queue, dashboard.telemetry.queue)

    def run_request():
        try:
            if request.action == "run":
                dashboard.runner.run_inference_request(request)
            elif request.action == "calib":
                dashboard.runner.run_calibration_request(request)
            elif request.action == "collect":
                dashboard.runner.run_collection_request(request)
            elif request.action == "train":
                dashboard.runner.run_training_request(request)
            elif request.action == "camera_collect":
                dashboard.runner.run_camera_collection_request(request)
        except Exception as error:
            loop.call_soon_threadsafe(dashboard.logs.queue.put_nowait, f"FATAL ERROR: {str(error)}")

    # run_request blocks, so it runs on a worker thread.
    dashboard.runner_task = asyncio.create_task(asyncio.to_thread(run_request))
    return {"status": "started"}


@router.get("/api/pipeline/state")
async def pipeline_state(dashboard: Dashboard):
    return {"isRunning": dashboard.runner.is_running if dashboard.runner else False}


@router.post("/api/action/stop")
async def stop_inference(dashboard: Dashboard):
    if dashboard.runner:
        dashboard.runner.stop()
    if dashboard.runner_task:
        try:
            await asyncio.wait_for(asyncio.shield(dashboard.runner_task), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            pass
        dashboard.runner_task = None
    return {"status": "stopped"}


@router.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.inference.serve(websocket)


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.stream.serve(websocket)


@router.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.logs.serve(websocket)


@router.websocket("/ws/training")
async def websocket_training(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.training.serve(websocket)


@router.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.telemetry.serve(websocket)
