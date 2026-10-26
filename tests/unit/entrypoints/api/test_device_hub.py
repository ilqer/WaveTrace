import asyncio
import json
import subprocess
import threading
import time

import pytest
import serial

from wavetrace.entrypoints.api.device_ctl import DeviceHub


class _FakeProcess:
    """`stdout` yields `lines`, then raises `read_error` when one is given."""

    def __init__(self, lines, read_error=None):
        self._lines = lines
        self._read_error = read_error
        self.stdin = None
        self.is_killed = False
        self.is_reaped = False

    @property
    def stdout(self):
        yield from self._lines
        if self._read_error is not None:
            raise self._read_error

    def __enter__(self):
        return self

    def __exit__(self, *exception_info):
        self.wait()

    def poll(self):
        return None

    def kill(self):
        self.is_killed = True

    def wait(self):
        self.is_reaped = True
        return 0


@pytest.fixture
def hub():
    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever)
    loop_thread.start()
    yield DeviceHub(loop, asyncio.Queue())
    loop.call_soon_threadsafe(loop.stop)
    loop_thread.join()
    loop.close()


def _published_lines(hub):
    async def _drain():
        return [json.loads(hub.queue.get_nowait())["line"] for _ in range(hub.queue.qsize())]

    return asyncio.run_coroutine_threadsafe(_drain(), hub.loop).result(timeout=5.0)


def _spawn(monkeypatch, process):
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: process)
    return process


def test_a_finished_command_is_reaped_and_its_output_and_exit_code_published(hub, monkeypatch):
    process = _spawn(monkeypatch, _FakeProcess(["capturing\n", "done\n"]))
    hub.run_on_pi("pi.local", "capture")
    assert process.is_reaped and not process.is_killed
    assert hub.get_state()["busy"] is False
    assert _published_lines(hub) == ["ssh pi.local: capture", "capturing", "done", "exit code 0"]


def test_an_output_that_cannot_be_relayed_kills_and_reaps_the_command(hub, monkeypatch):
    undecodable_output = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    process = _spawn(monkeypatch, _FakeProcess(["capturing\n"], read_error=undecodable_output))
    with pytest.raises(UnicodeDecodeError):
        hub.run_on_pi("pi.local", "capture")
    assert process.is_killed and process.is_reaped
    assert hub.get_state()["busy"] is False


class _FailingSerialPort:
    def __init__(self):
        self.is_closed = False

    def readline(self):
        raise serial.SerialException("device disconnected")

    def close(self):
        self.is_closed = True


def test_a_monitor_read_error_closes_the_port(hub, monkeypatch):
    serial_port = _FailingSerialPort()
    monkeypatch.setattr(serial, "Serial", lambda port, baud, timeout: serial_port)
    hub.start_monitor("/dev/tty.fake")
    deadline_s = time.monotonic() + 5.0
    while not serial_port.is_closed and time.monotonic() < deadline_s:
        time.sleep(0.01)
    hub.stop_monitor("/dev/tty.fake")
    assert serial_port.is_closed
    assert _published_lines(hub) == [
        "opened /dev/tty.fake @ 115200", "read error: device disconnected", "closed /dev/tty.fake"]
