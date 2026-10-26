"""The dashboard's runtime state: built once by the app's lifespan, handed to routes through `Dashboard`."""

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, WebSocket, WebSocketDisconnect
from starlette.requests import HTTPConnection

from wavetrace.entrypoints.api.device_ctl import DeviceHub
from wavetrace.entrypoints.api.streamer import WaveTraceRunner


class BroadcastChannel:
    """Sends each text message put on `queue` to every connected WebSocket client, in queue order; a client
    whose send fails is dropped. `mirror_payload`, when given, is awaited with the message's JSON payload
    before any client gets the message."""

    def __init__(self, mirror_payload: Callable[[dict], Awaitable[None]] | None = None):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._mirror_payload = mirror_payload
        self._clients: set[WebSocket] = set()

    async def run(self) -> None:
        while True:
            message = await self.queue.get()
            if self._mirror_payload is not None:
                await self._mirror_payload(json.loads(message))
            for client in list(self._clients):
                try:
                    await client.send_text(message)
                except Exception:
                    self._clients.discard(client)

    async def serve(self, websocket: WebSocket) -> None:
        """Returns when the client disconnects; text the client sends is ignored."""
        await websocket.accept()
        self._clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            self._clients.discard(websocket)


@dataclass(slots=True)
class DashboardState:
    """runner / runner_task: the latest pipeline request and its worker-thread task, None before the first."""

    inference: BroadcastChannel
    stream: BroadcastChannel
    logs: BroadcastChannel
    training: BroadcastChannel
    telemetry: BroadcastChannel
    device: BroadcastChannel
    device_hub: DeviceHub
    runner: WaveTraceRunner | None = None
    runner_task: asyncio.Task | None = None


def get_dashboard_state(connection: HTTPConnection) -> DashboardState:
    """Takes HTTPConnection, not Request: a Request dependency fails to resolve on a WebSocket route."""
    return connection.app.state.dashboard


Dashboard = Annotated[DashboardState, Depends(get_dashboard_state)]
