import asyncio
import json

import pytest

from wavetrace.entrypoints.api.streamer import WaveTraceRunner

QUEUE_NAMES = ("inference", "stream", "logs", "training", "telemetry")


class RecordedMessages(list):
    """(queue name, message) pairs in the order the runner put them."""

    def payloads(self, queue_name: str) -> list:
        return [json.loads(message) for name, message in self if name == queue_name]

    def logs(self) -> list[str]:
        """Log lines without their time of day."""
        return [message.split("] ", 1)[1] for name, message in self if name == "logs"]


class _RecordingQueue:
    def __init__(self, messages: RecordedMessages, queue_name: str, on_put):
        self._messages = messages
        self._queue_name = queue_name
        self._on_put = on_put

    def put(self, message):
        self._messages.append((self._queue_name, message))
        if self._on_put is not None:
            self._on_put(message)


@pytest.fixture
def build_recording_runner(monkeypatch):
    """Returns build(on_put_by_queue) -> (runner, messages). The runner's queues record each message as it is put,
    on the runner's own thread, so an on_put callback can stop the runner at an exact message; real asyncio queues
    would hand the message to the event loop first and lose that ordering."""
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", lambda coroutine, loop: None)

    def build(on_put_by_queue=None) -> tuple[WaveTraceRunner, RecordedMessages]:
        on_put_by_queue = on_put_by_queue or {}
        messages = RecordedMessages()
        queues = [_RecordingQueue(messages, name, on_put_by_queue.get(name)) for name in QUEUE_NAMES]
        return WaveTraceRunner(None, *queues), messages

    return build
