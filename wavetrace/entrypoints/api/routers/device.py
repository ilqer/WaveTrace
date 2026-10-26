import asyncio

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

from wavetrace.entrypoints.api.dashboard_state import Dashboard
from wavetrace.entrypoints.api.device_ctl import list_serial_ports

router = APIRouter()


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


class SerialMonitorRequest(BaseModel):
    port: str
    baud: int = 115200


class StopProcRequest(BaseModel):
    proc_id: str | None = None


class InputRequest(BaseModel):
    proc_id: str
    input: str


@router.get("/api/serial/ports")
async def serial_ports():
    return {"ports": list_serial_ports()}


@router.get("/api/device/state")
async def device_state(dashboard: Dashboard):
    return dashboard.device_hub.get_state()


@router.post("/api/serial/monitor/start")
async def serial_monitor_start(request: SerialMonitorRequest, dashboard: Dashboard):
    return dashboard.device_hub.start_monitor(request.port, request.baud)


@router.post("/api/serial/monitor/stop")
async def serial_monitor_stop(dashboard: Dashboard, request: StopMonitorRequest = None):
    port = request.port if request else None
    return dashboard.device_hub.stop_monitor(port)


@router.post("/api/flash")
async def flash(request: FlashRequest, dashboard: Dashboard):
    # the build-and-flash blocks, so it runs on a worker thread and the event loop keeps streaming
    asyncio.create_task(asyncio.to_thread(dashboard.device_hub.flash, request.role, request.node_id, request.port,
                                          request.clean))
    return {"status": "flashing", "role": request.role, "port": request.port, "clean": request.clean}


@router.post("/api/pi/run")
async def pi_run(request: PiRequest, dashboard: Dashboard):
    asyncio.create_task(asyncio.to_thread(dashboard.device_hub.run_on_pi, request.host, request.command))
    return {"status": "running", "host": request.host}


@router.post("/api/script/run")
async def script_run(request: ScriptRequest, dashboard: Dashboard):
    asyncio.create_task(asyncio.to_thread(dashboard.device_hub.run_script, request.script, request.args))
    return {"status": "running", "script": request.script}


@router.post("/api/device/stop")
async def device_stop(dashboard: Dashboard, request: StopProcRequest = None):
    process_id = request.proc_id if request else None
    return dashboard.device_hub.stop_process(process_id)


@router.post("/api/device/input")
async def device_input(request: InputRequest, dashboard: Dashboard):
    return dashboard.device_hub.send_input(request.proc_id, request.input)


@router.websocket("/ws/device")
async def websocket_device(websocket: WebSocket, dashboard: Dashboard):
    await dashboard.device.serve(websocket)
