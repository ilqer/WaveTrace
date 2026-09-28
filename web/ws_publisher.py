import json
import asyncio
from typing import Any

from wavetrace.output.publisher import Publisher, result_to_dict

class WsPublisher(Publisher):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, *, mode: str = ""):
        super().__init__(mode=mode)
        self.loop = loop
        self.queue = queue

    def publish(self, result: Any) -> None:
        data = result_to_dict(result, mode=self.mode)
        # publish() runs on a background thread, so hop the put onto the event loop
        asyncio.run_coroutine_threadsafe(self.queue.put(json.dumps(data)), self.loop)
