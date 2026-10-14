import json
import time
from foxglove_websocket.server import FoxgloveServer

class FoxgloveIntegration:
    def __init__(self):
        self.server = None
        self.verdict_channel_id = None
        self.variance_channel_id = None
        
    async def start(self):
        self.server = FoxgloveServer("0.0.0.0", 8765, "WaveTrace Lab")
        self.server.start()
        
        self.verdict_channel_id = await self.server.add_channel({
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

        self.variance_channel_id = await self.server.add_channel({
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
        message = {
            "class": data.get("class", 0),
            "conf": data.get("conf", 0.0),
            "mode": data.get("mode", ""),
            "timestamp": time.time_ns()
        }
        await self.server.send_message(self.verdict_channel_id, time.time_ns(),
                                       json.dumps(message).encode('utf8'))

    async def publish_stream(self, data: dict):
        if not self.server: return
        intercarrier_features = data.get("ic")
        if intercarrier_features and len(intercarrier_features) > 0:
            mean_abs_feature = (sum(abs(feature) for feature in intercarrier_features)
                                / len(intercarrier_features))
            message = {
                "value": float(mean_abs_feature),
                "timestamp": time.time_ns()
            }
            await self.server.send_message(self.variance_channel_id, time.time_ns(),
                                           json.dumps(message).encode('utf8'))

foxglove_server = FoxgloveIntegration()
