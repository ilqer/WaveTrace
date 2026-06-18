"""Full-mesh bring-up check: count CSI arrivals per directed (tx -> rx) link and their rates.

Run on the PC while on the mesh router, with the nodes powered on:
    .venv/bin/python mesh_verify.py            # port 9876, prints once per second
    .venv/bin/python mesh_verify.py 9876 20    # explicit port, run 20 s

Wire format (batched, plan §3): each datagram = a JSON header line {"v","node","ntp_ms","n"} where
`node` is the RECEIVER, followed by `n` esp-csi CSV lines whose MAC column (the TRANSMITTER) is the
tx identity. So a link = (tx_mac_short -> rx_node). For N nodes you expect N*(N-1) links at roughly
equal rates. A missing link = that ordered pair never completed (board off, token not reaching it,
gain SKIP/too-close, or wrong PC_IP). No CsiFrame/DSP dependency, so it runs before the pipeline is
wired for (tx,rx)."""

import collections
import json
import socket
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9876
RUN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9  # default: until Ctrl+C

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

total = collections.Counter()   # (tx, rx) -> all-time frames
window = collections.Counter()  # (tx, rx) -> frames this second
dgrams = 0
start = last_print = time.time()
print(f"listening on udp/{PORT}  (Ctrl+C to stop)")


def short(mac):
    """Last two MAC octets as the tx label (e.g. '4f:9c')."""
    return ":".join(mac.split(":")[-2:]) if ":" in mac else mac


try:
    while time.time() - start < RUN_S:
        try:
            payload, _ = sock.recvfrom(65535)
        except socket.timeout:
            payload = None
        if payload:
            lines = payload.decode("utf-8", "replace").splitlines()
            if not lines:
                continue
            try:
                rx = int(json.loads(lines[0])["node"])
            except (ValueError, KeyError):
                continue
            dgrams += 1
            for line in lines[1:]:
                parts = line.split(",", 24)
                if len(parts) >= 3 and parts[0] == "CSI_DATA":
                    link = (short(parts[2]), rx)
                    total[link] += 1
                    window[link] += 1
        now = time.time()
        if now - last_print >= 1.0:
            links = " ".join(f"{tx}->{rx}:{window[(tx, rx)]}"
                             for (tx, rx) in sorted(total)) or "(no frames)"
            print(f"[{now - start:5.1f}s] hz  {links}")
            window.clear()
            last_print = now
except KeyboardInterrupt:
    pass
finally:
    sock.close()

print("\n=== totals (tx -> rx : frames) ===")
for (tx, rx) in sorted(total):
    print(f"  {tx} -> {rx} : {total[(tx, rx)]}")
print(f"distinct links seen: {len(total)}   datagrams: {dgrams}")
