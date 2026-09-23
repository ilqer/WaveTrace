"""Live table of the mesh heartbeats arriving on udp/9877.

Nodes beat every 2 s, so STALE_S allows three missed beats.
"""

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

# nodes learn this PC's IP from this broadcast
DISCOVERY_PORT = 9878
discoverySock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
discoverySock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
lastPing = 0

last = {}
print(f"health monitor on udp/{PORT}  (Ctrl+C to stop)\n")
try:
    while True:
        now = time.time()
        if now - lastPing > 2.0:
            discoverySock.sendto(b"WAVETRACE_PING", ("255.255.255.255", DISCOVERY_PORT))
            lastPing = now

        try:
            payload, addr = sock.recvfrom(2048)
            h = json.loads(payload.decode("utf-8", "replace").splitlines()[0])
            if h.get("type") == "health":
                last[int(h["node"])] = (time.time(), h, addr[0])
        except (socket.timeout, ValueError, KeyError, IndexError):
            pass
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
        print("\033[2J\033[H" + "\n".join(rows), flush=True)  # ansi: clear screen, cursor home
        time.sleep(0.0)
except KeyboardInterrupt:
    pass
finally:
    sock.close()
