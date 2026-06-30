"""Per-node health monitor — the PC-side view of the mesh heartbeats (HEALTH_UDP_PORT=9877).

Each node sends a heartbeat every HEALTH_MS (2 s). This prints a live table so you can see, at a
glance, which boards are up, their CSI rate, gain state, RSSI, free heap, leader, and clock-sync.

    .venv/bin/python scripts/health_monitor.py          # port 9877
    .venv/bin/python scripts/health_monitor.py 9877

A node going RED/STALE (no heartbeat > 6 s) is the "not delivering to PC" signal; the on-board USB
serial log is the complementary health view when a node has no PC link at all."""

import collections
import json
import socket
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9877
STALE_S = 6.0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(0.5)

# Discovery broadcast: send a ping every 2s so nodes can find this PC's IP
DISCOVERY_PORT = 9878
discovery_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
discovery_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
last_ping = 0

last = {}   # node -> (recv_time, health dict)
print(f"health monitor on udp/{PORT}  (Ctrl+C to stop)\n")
try:
    while True:
        now = time.time()
        # Send discovery ping every 2 seconds
        if now - last_ping > 2.0:
            discovery_sock.sendto(b"WAVETRACE_PING", ("255.255.255.255", DISCOVERY_PORT))
            last_ping = now

        try:
            payload, addr = sock.recvfrom(2048)
            h = json.loads(payload.decode("utf-8", "replace").splitlines()[0])
            if h.get("type") == "health":
                last[int(h["node"])] = (time.time(), h, addr[0])
        except (socket.timeout, ValueError, KeyError, IndexError):
            pass
        # redraw table
        now = time.time()
        rows = ["  node  ip              age   csi_hz tx_hz peers leader gain  agc rssi heap(KB) up(s) clk"]
        for n in sorted(last):
            t0, h, ip = last[n]
            age = now - t0
            mark = "STALE" if age > STALE_S else f"{age:4.1f}s"
            rows.append(
                f"  {n:<4}  {ip:<14}  {mark:>5} {h.get('csi_hz',0):>6} {h.get('tx_hz',0):>5} "
                f"{h.get('peers',0):>5} {h.get('leader','?'):>6} {h.get('gain','?'):>4} "
                f"{h.get('agc',0):>3} {h.get('rssi',0):>4} {h.get('heap',0)//1024:>7} "
                f"{h.get('up_s',0):>5} {'ok' if h.get('synced') else 'no':>3}")
        print("\033[2J\033[H" + "\n".join(rows), flush=True)  # clear + home, then table
        time.sleep(0.0)
except KeyboardInterrupt:
    pass
finally:
    sock.close()
