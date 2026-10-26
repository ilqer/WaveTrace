import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from wavetrace.entrypoints.api.dashboard_state import BroadcastChannel, DashboardState
from wavetrace.entrypoints.api.device_ctl import DeviceHub
from wavetrace.entrypoints.api.foxglove import foxglove_server
from wavetrace.entrypoints.api.routers import camera, device, model, paths, pipeline
from wavetrace.entrypoints.api.server_options import WebServerOptions


@asynccontextmanager
async def lifespan(app: FastAPI):
    device_channel = BroadcastChannel()
    dashboard = DashboardState(
        inference=BroadcastChannel(foxglove_server.publish_inference),
        stream=BroadcastChannel(foxglove_server.publish_stream),
        logs=BroadcastChannel(),
        training=BroadcastChannel(),
        telemetry=BroadcastChannel(),
        device=device_channel,
        device_hub=DeviceHub(asyncio.get_running_loop(), device_channel.queue),
    )
    app.state.dashboard = dashboard
    await foxglove_server.start()
    for channel in (dashboard.inference, dashboard.stream, dashboard.logs, dashboard.training,
                    dashboard.telemetry, dashboard.device):
        asyncio.create_task(channel.run())
    yield
    if dashboard.runner:
        dashboard.runner.stop()
        await asyncio.sleep(0.5)


app = FastAPI(title="WaveTrace Lab Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(model.router)
app.include_router(paths.router)
app.include_router(device.router)
app.include_router(camera.router)
# a catch-all, so it must be registered after every API route
app.mount("/", StaticFiles(directory="web/ui/dist", html=True), name="static")

if __name__ == "__main__":
    server_options = WebServerOptions()
    uvicorn.run("wavetrace.entrypoints.api.app:app", host=server_options.host, port=server_options.port,
                reload=server_options.reload,
                reload_dirs=list(server_options.reload_dirs),        # uvicorn wants list, not tuple
                reload_includes=list(server_options.reload_includes))
