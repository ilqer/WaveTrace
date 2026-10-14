"""Live table of the mesh heartbeats arriving on udp/9877.

Nodes beat every 2 s, so STALE_S allows three missed beats.
"""

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
discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
last_ping_time_s = 0

latest_by_node = {}
print(f"health monitor on udp/{PORT}  (Ctrl+C to stop)\n")
try:
    while True:
        now = time.time()
        if now - last_ping_time_s > 2.0:
            discovery_socket.sendto(b"WAVETRACE_PING", ("255.255.255.255", DISCOVERY_PORT))
            last_ping_time_s = now

        try:
            payload, sender_address = sock.recvfrom(2048)
            health = json.loads(payload.decode("utf-8", "replace").splitlines()[0])
            if health.get("type") == "health":
                latest_by_node[int(health["node"])] = (time.time(), health, sender_address[0])
        except (socket.timeout, ValueError, KeyError, IndexError):
            pass
        now = time.time()
        rows = ["  node  ip              age   csi_hz tx_hz peers leader gain  agc rssi heap(KB) up(s) clk"]
        for node_id in sorted(latest_by_node):
            last_heard_s, health, node_address = latest_by_node[node_id]
            age_s = now - last_heard_s
            mark = "STALE" if age_s > STALE_S else f"{age_s:4.1f}s"
            rows.append(
                f"  {node_id:<4}  {node_address:<14}  {mark:>5} {health.get('csi_hz',0):>6} "
                f"{health.get('tx_hz',0):>5} {health.get('peers',0):>5} {health.get('leader','?'):>6} "
                f"{health.get('gain','?'):>4} {health.get('agc',0):>3} {health.get('rssi',0):>4} "
                f"{health.get('heap',0)//1024:>7} "
                f"{health.get('up_s',0):>5} {'ok' if health.get('synced') else 'no':>3}")
        print("\033[2J\033[H" + "\n".join(rows), flush=True)  # ansi: clear screen, cursor home
        time.sleep(0.0)
except KeyboardInterrupt:
    pass
finally:
    sock.close()
