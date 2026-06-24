"""Hardware device control for the dashboard: serial port discovery, live ESP serial
monitor, firmware flashing, and Pi capture control over SSH.

Everything streams line-by-line to one asyncio queue (the /ws/device socket) tagged with a
`source` so the UI can colour/route it. The backend owns at most ONE long-running device op
at a time (a serial port has a single owner), so starting a flash stops an active monitor."""

import asyncio
import json
import os
import shlex
import subprocess
import threading
import time

import serial
import serial.tools.list_ports as list_ports

# IDF must be sourced for idf.py to exist (it isn't on PATH in a plain shell). flash.sh's header
# documents ~/esp/esp-idf/export.sh; allow override via env for non-default installs.
IDF_EXPORT = os.path.expanduser(os.environ.get("IDF_EXPORT", "~/esp/esp-idf/export.sh"))
FIRMWARE_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "firmware"))


def list_serial_ports() -> list[dict]:
    """All serial devices, USB ones first (the ESPs); Bluetooth/debug noise sinks to the bottom."""
    ports = []
    for p in list_ports.comports():
        dev = p.device
        is_usb = ("usb" in dev.lower()) or ("USB" in (p.hwid or ""))
        ports.append({
            "device": dev,
            "description": p.description or "",
            "hwid": p.hwid or "",
            "likely_esp": is_usb,
        })
    ports.sort(key=lambda x: (not x["likely_esp"], x["device"]))
    return ports


class DeviceHub:
    """Single-owner hub for the serial port. Publishes every line to `queue` from worker threads."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self._monitors: dict[str, tuple[threading.Thread, threading.Event]] = {}
        self._procs: dict[str, subprocess.Popen] = {}

    def get_state(self) -> dict:
        return {
            "monitoring": list(self._monitors.keys()),
            "runningScripts": [k.split(":", 1)[1] for k in self._procs.keys() if k.startswith("script:")],
            "busy": any(k in ("flash", "pi") for k in self._procs.keys())
        }

    def _publish(self, source: str, line: str, level: str = "info") -> None:
        payload = json.dumps({"source": source, "line": line.rstrip("\n"),
                              "level": level, "t": time.time()})
        # called from worker threads → hop back onto the event loop
        asyncio.run_coroutine_threadsafe(self.queue.put(payload), self.loop)

    # ---- serial monitor -------------------------------------------------------------
    def start_monitor(self, port: str, baud: int = 115200) -> dict:
        if port in self._monitors:
            return {"status": "monitoring", "port": port, "baud": baud}
        stop_ev = threading.Event()
        t = threading.Thread(target=self._monitor_loop, args=(port, baud, stop_ev), daemon=True)
        self._monitors[port] = (t, stop_ev)
        t.start()
        return {"status": "monitoring", "port": port, "baud": baud}

    def _monitor_loop(self, port: str, baud: int, stop_ev: threading.Event) -> None:
        src = f"tty:{os.path.basename(port)}"
        try:
            ser = serial.Serial(port, baud, timeout=0.5)
        except Exception as e:
            self._publish(src, f"open failed: {e}", level="error")
            return
        self._publish(src, f"opened {port} @ {baud}", level="system")
        try:
            while not stop_ev.is_set():
                try:
                    raw = ser.readline()
                except Exception as e:
                    self._publish(src, f"read error: {e}", level="error")
                    break
                if raw:
                    self._publish(src, raw.decode("utf-8", errors="replace"))
        finally:
            ser.close()
            self._publish(src, f"closed {port}", level="system")

    def stop_monitor(self, port: str | None = None) -> dict:
        if port is None:
            for p, (t, ev) in list(self._monitors.items()):
                ev.set()
                t.join(timeout=2.0)
            self._monitors.clear()
            return {"status": "stopped_all"}
        
        if port in self._monitors:
            t, ev = self._monitors.pop(port)
            ev.set()
            t.join(timeout=2.0)
        return {"status": "stopped", "port": port}

    def send_input(self, proc_id: str, data: str) -> dict:
        """Send input to a running process's stdin."""
        if proc_id in self._procs:
            proc = self._procs[proc_id]
            if proc.stdin:
                try:
                    proc.stdin.write(data)
                    proc.stdin.flush()
                    return {"status": "ok"}
                except Exception as e:
                    return {"status": "error", "error": str(e)}
        return {"status": "not_found"}

    # ---- subprocess streaming (flash + ssh) -----------------------------------------
    def _stream(self, source: str, proc_id: str, argv: list[str], cwd: str | None = None) -> int:
        """Run argv, pumping combined stdout/stderr to the device socket; returns exit code."""
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE, text=True, bufsize=1)
            self._procs[proc_id] = proc
        except Exception as e:
            self._publish(source, f"spawn failed: {e}", level="error")
            return -1
        for line in proc.stdout:  # type: ignore[union-attr]
            self._publish(source, line)
        code = proc.wait()
        self._procs.pop(proc_id, None)
        self._publish(source, f"exit code {code}",
                      level="system" if code == 0 else "error")
        return code

    def flash(self, role: str, node_id: int | None, port: str, clean: bool = False) -> None:
        """Flash one board via firmware/flash.sh (NO_MONITOR so the call returns)."""
        if port in self._monitors:
            self._publish("flash", f"stopping monitor on {port} first", level="system")
            self.stop_monitor(port)
        if not os.path.exists(IDF_EXPORT):
            self._publish("flash", f"IDF export.sh not found at {IDF_EXPORT}; "
                          f"set IDF_EXPORT env var", level="error")
            return
        # build the flash.sh invocation per role (tx takes no NODE_ID)
        if role == "tx":
            fcmd = f"./flash.sh tx {shlex.quote(port)}"
        else:
            if node_id is None:
                self._publish("flash", "node/rx flash needs a NODE_ID", level="error")
                return
            fcmd = f"./flash.sh {role} {int(node_id)} {shlex.quote(port)}"
        # login shell sources IDF, NO_MONITOR drops the blocking monitor step in flash.sh.
        # CLEAN=1 makes flash.sh wipe sdkconfig+build so sdkconfig.defaults re-applies (full rebuild).
        clean_env = "CLEAN=1 " if clean else ""
        inner = f"source {shlex.quote(IDF_EXPORT)} && {clean_env}NO_MONITOR=1 {fcmd}"
        if clean:
            self._publish("flash", "clean rebuild: wiping sdkconfig + build (full recompile)", level="system")
        self._publish("flash", f"$ {fcmd}", level="system")
        self._stream("flash", "flash", ["bash", "-lc", inner], cwd=FIRMWARE_DIR)

    def run_pi(self, host: str, command: str) -> None:
        """Run a command on the Pi over SSH and stream its output (capture control, Nexmon setup)."""
        if not host:
            self._publish("pi", "no Pi host configured", level="error")
            return
        self._publish("pi", f"ssh {host}: {command}", level="system")
        # -tt forces a pty so long-running capture scripts flush their output live
        self._stream("pi", "pi", ["ssh", "-tt", host, command])

    def run_script(self, script_name: str, args: str = "") -> None:
        """Run a local python script from the root directory and stream its output."""
        if not script_name.endswith(".py"):
            self._publish("script", "Invalid script name", level="error")
            return
        root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
        cmd = ["python", "-u", script_name]
        if args:
            cmd.extend(shlex.split(args))
        self._publish(f"script:{script_name}", f"Running: {' '.join(cmd)}", level="system")
        self._stream(f"script:{script_name}", f"script:{script_name}", cmd, cwd=root_dir)



    def stop_proc(self, proc_id: str | None = None) -> dict:
        if proc_id is None:
            for pid in ["flash", "pi"]:
                if pid in self._procs and self._procs[pid].poll() is None:
                    self._procs[pid].terminate()
                    return {"status": "terminating", "proc_id": pid}
            return {"status": "idle"}
        
        if proc_id in self._procs and self._procs[proc_id].poll() is None:
            self._procs[proc_id].terminate()
            return {"status": "terminating", "proc_id": proc_id}
        return {"status": "idle"}
