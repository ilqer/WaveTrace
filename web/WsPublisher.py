import json
import asyncio
from typing import Any

from wavetrace.output.Publisher import Publisher, result_to_dict

class WsPublisher(Publisher):
    """
    WebSocket publisher.
    Pushes inference results to an asyncio queue thread-safely.
    """
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, *, mode: str = ""):
        super().__init__(mode=mode)
        self.loop = loop
        self.queue = queue

    def publish(self, result: Any) -> None:
        data = result_to_dict(result, mode=self.mode)
        # Thread-safe push since publish() is called from a background thread
        asyncio.run_coroutine_threadsafe(self.queue.put(json.dumps(data)), self.loop)
