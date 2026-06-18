import asyncio
import json
import base64
import time
from foxglove_websocket.server import FoxgloveServer

class FoxgloveIntegration:
    def __init__(self):
        self.server = None
        self.chan_verdict = None
        self.chan_variance = None
        
    async def start(self):
        self.server = FoxgloveServer("0.0.0.0", 8765, "WaveTrace Lab")
        self.server.start()
        
        self.chan_verdict = await self.server.add_channel({
            "topic": "/wavetrace/verdict",
            "encoding": "json",
            "schemaName": "wavetrace.Verdict",
            "schema": json.dumps({
                "type": "object",
                "properties": {
                    "class": {"type": "number"},
                    "conf": {"type": "number"},
                    "mode": {"type": "string"},
                    "timestamp": {"type": "number"}
                }
            })
        })

        self.chan_variance = await self.server.add_channel({
            "topic": "/wavetrace/variance",
            "encoding": "json",
            "schemaName": "wavetrace.Variance",
            "schema": json.dumps({
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "timestamp": {"type": "number"}
                }
            })
        })

    async def publish_inference(self, data: dict):
        if not self.server: return
        msg = {
            "class": data.get("class", 0),
            "conf": data.get("conf", 0.0),
            "mode": data.get("mode", ""),
            "timestamp": time.time_ns()
        }
        await self.server.send_message(self.chan_verdict, time.time_ns(), json.dumps(msg).encode('utf8'))

    async def publish_stream(self, data: dict):
        if not self.server: return
        # Extract variance/mean from IC vector and send to Foxglove
        ic = data.get("ic")
        if ic and len(ic) > 0:
            val = sum(abs(x) for x in ic) / len(ic)
            msg = {
                "value": float(val),
                "timestamp": time.time_ns()
            }
            await self.server.send_message(self.chan_variance, time.time_ns(), json.dumps(msg).encode('utf8'))

fg_server = FoxgloveIntegration()
