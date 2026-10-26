"""Streams device output line by line to the /ws/device queue, tagged with its source.
One long-running operation per serial port: flashing stops an active monitor on that port."""

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time

import serial
import serial.tools.list_ports as list_ports

# idf.py only exists once export.sh has been sourced.
IDF_EXPORT = os.path.expanduser(os.environ.get("IDF_EXPORT", "~/esp/esp-idf/export.sh"))
FIRMWARE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "firmware"))


def list_serial_ports() -> list[dict]:
    ports = []
    for port_info in list_ports.comports():
        device = port_info.device
        is_usb = ("usb" in device.lower()) or ("USB" in (port_info.hwid or ""))
        ports.append({
            "device": device,
            "description": port_info.description or "",
            "hwid": port_info.hwid or "",
            "likely_esp": is_usb,
        })
    ports.sort(key=lambda port_entry: (not port_entry["likely_esp"], port_entry["device"]))
    return ports


class DeviceHub:
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self._monitors: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._processes: dict[str, subprocess.Popen] = {}

    def get_state(self) -> dict:
        return {
            "monitoring": list(self._monitors.keys()),
            "runningScripts": [key.split(":", 1)[1] for key in self._processes.keys()
                               if key.startswith("script:")],
            "busy": any(key in ("flash", "pi") for key in self._processes.keys())
        }

    def _publish(self, source: str, line: str, level: str = "info") -> None:
        payload = json.dumps({"source": source, "line": line.rstrip("\n"),
                              "level": level, "t": time.time()})
        # called from worker threads → hop back onto the event loop
        asyncio.run_coroutine_threadsafe(self.queue.put(payload), self.loop)

    def start_monitor(self, port: str, baud: int = 115200) -> dict:
        if port in self._monitors:
            return {"status": "monitoring", "port": port, "baud": baud}
        stop_event = threading.Event()
        monitor_thread = threading.Thread(target=self._monitor_serial, args=(port, baud, stop_event),
                                          daemon=True)
        self._monitors[port] = (monitor_thread, stop_event)
        monitor_thread.start()
        return {"status": "monitoring", "port": port, "baud": baud}

    def _monitor_serial(self, port: str, baud: int, stop_event: threading.Event) -> None:
        source = f"tty:{os.path.basename(port)}"
        try:
            serial_connection = serial.Serial(port, baud, timeout=0.5)
        except Exception as error:
            self._publish(source, f"open failed: {error}", level="error")
            return
        self._publish(source, f"opened {port} @ {baud}", level="system")
        try:
            while not stop_event.is_set():
                try:
                    raw_line = serial_connection.readline()
                except Exception as error:
                    self._publish(source, f"read error: {error}", level="error")
                    break
                if raw_line:
                    self._publish(source, raw_line.decode("utf-8", errors="replace"))
        finally:
            serial_connection.close()
            self._publish(source, f"closed {port}", level="system")

    def stop_monitor(self, port: str | None = None) -> dict:
        if port is None:
            for monitor_thread, stop_event in list(self._monitors.values()):
                stop_event.set()
                monitor_thread.join(timeout=2.0)
            self._monitors.clear()
            return {"status": "stopped_all"}
        
        if port in self._monitors:
            monitor_thread, stop_event = self._monitors.pop(port)
            stop_event.set()
            monitor_thread.join(timeout=2.0)
        return {"status": "stopped", "port": port}

    def send_input(self, process_id: str, data: str) -> dict:
        if process_id in self._processes:
            process = self._processes[process_id]
            if process.stdin:
                try:
                    process.stdin.write(data)
                    process.stdin.flush()
                    return {"status": "ok"}
                except Exception as error:
                    return {"status": "error", "error": str(error)}
        return {"status": "not_found"}

    def _stream(self, source: str, process_id: str, argv: list[str], cwd: str | None = None) -> int:
        try:
            process = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True, bufsize=1)
            self._processes[process_id] = process
        except Exception as error:
            self._publish(source, f"spawn failed: {error}", level="error")
            return -1
        for line in process.stdout:  # type: ignore[union-attr]
            self._publish(source, line)
        code = process.wait()
        self._processes.pop(process_id, None)
        self._publish(source, f"exit code {code}",
                      level="system" if code == 0 else "error")
        return code

    def flash(self, role: str, node_id: int | None, port: str, clean: bool = False) -> None:
        if port in self._monitors:
            self._publish("flash", f"stopping monitor on {port} first", level="system")
            self.stop_monitor(port)
        if not os.path.exists(IDF_EXPORT):
            self._publish("flash", f"IDF export.sh not found at {IDF_EXPORT}; "
                          f"set IDF_EXPORT env var", level="error")
            return
        # flash.sh takes no NODE_ID for the tx role
        if role == "tx":
            flash_command = f"./flash.sh tx {shlex.quote(port)}"
        else:
            if node_id is None:
                self._publish("flash", "node/rx flash needs a NODE_ID", level="error")
                return
            flash_command = f"./flash.sh {role} {int(node_id)} {shlex.quote(port)}"
        # flash.sh reads both: NO_MONITOR skips its blocking monitor step, CLEAN=1 wipes
        # sdkconfig and build for a full rebuild.
        clean_env_prefix = "CLEAN=1 " if clean else ""
        shell_command = (f"source {shlex.quote(IDF_EXPORT)} && "
                         f"{clean_env_prefix}NO_MONITOR=1 {flash_command}")
        if clean:
            self._publish("flash", "clean rebuild: wiping sdkconfig + build (full recompile)", level="system")
        self._publish("flash", f"$ {flash_command}", level="system")
        self._stream("flash", "flash", ["bash", "-lc", shell_command], cwd=FIRMWARE_DIR)

    def run_on_pi(self, host: str, command: str) -> None:
        if not host:
            self._publish("pi", "no Pi host configured", level="error")
            return
        self._publish("pi", f"ssh {host}: {command}", level="system")
        # -tt forces a pty so long-running capture scripts flush their output live
        self._stream("pi", "pi", ["ssh", "-tt", host, command])

    def run_script(self, script_name: str, arguments: str = "") -> None:
        if not script_name.endswith(".py"):
            self._publish("script", "Invalid script name", level="error")
            return
        repo_root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        command = ["python", "-u", script_name]
        if arguments:
            command.extend(shlex.split(arguments))
        self._publish(f"script:{script_name}", f"Running: {' '.join(command)}", level="system")
        self._stream(f"script:{script_name}", f"script:{script_name}", command, cwd=repo_root_dir)



    def stop_process(self, process_id: str | None = None) -> dict:
        if process_id is None:
            for default_process_id in ["flash", "pi"]:
                if (default_process_id in self._processes
                        and self._processes[default_process_id].poll() is None):
                    self._processes[default_process_id].terminate()
                    return {"status": "terminating", "proc_id": default_process_id}
            return {"status": "idle"}
        
        if process_id in self._processes and self._processes[process_id].poll() is None:
            self._processes[process_id].terminate()
            return {"status": "terminating", "proc_id": process_id}
        return {"status": "idle"}
