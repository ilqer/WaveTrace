"""Full-mesh bring-up check: count CSI arrivals per directed (tx -> rx) link and their rates.

Run on the PC while on the mesh router, with the nodes powered on:
    .venv/bin/python mesh_verify.py            # port 9876, prints once per second
    .venv/bin/python mesh_verify.py 9876 20    # explicit port, run 20 s

Wire format (binary, little-endian): each datagram = a 13-byte header {magic,ver,node,ntp_ms,n}
where `node` is the RECEIVER and ver is 2 (ESP int8) or 3 (Pi int16), followed by `n` records
mac[6]|ts(u32)|len(u16)|CSI bytes whose MAC is the TRANSMITTER. So a link = (tx_mac_short -> rx_node).
For N nodes you expect N*(N-1) links at roughly equal rates. A missing link = that ordered pair never
completed (board off, token not reaching it, gain SKIP/too-close, or wrong PC_IP). Parses the header
inline (no DSP/numpy dependency), so it runs before the pipeline is wired."""

import collections
import socket
import struct
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9876
RUN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9  # default: until Ctrl+C

_HDR = struct.Struct("<BBBQH")  # magic, ver, node, ntp_ms, n -> 13 bytes (matches wavetrace/Source.py)
_MAGIC = 0x57

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)  # absorb bursts (see Source.bind_udp)
except OSError:
    pass
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

total = collections.Counter()   # (tx, rx) -> all-time frames
window = collections.Counter()  # (tx, rx) -> frames this second
dgrams = 0
start = last_print = time.time()
print(f"listening on udp/{PORT}  (Ctrl+C to stop)")


def count_links(payload):
    """Walk one binary batch, tallying (tx_short, rx_node) per record. Skips CSI bytes (no decode)."""
    if len(payload) < _HDR.size:
        return
    magic, ver, rx, _ntp, n = _HDR.unpack_from(payload, 0)
    if magic != _MAGIC or ver not in (2, 3):
        return
    step = 2 if ver == 2 else 4   # bytes/subcarrier: int8 (2) vs int16 (4)
    off, end = _HDR.size, len(payload)
    for _ in range(n):
        if off + 12 > end:
            break
        tx = f"{payload[off + 4]:02x}:{payload[off + 5]:02x}"  # last two octets of the tx MAC
        L = struct.unpack_from("<H", payload, off + 10)[0]
        off += 12
        if L % step != 0 or off + L > end:
            break
        off += L
        total[(tx, rx)] += 1
        window[(tx, rx)] += 1


try:
    while time.time() - start < RUN_S:
        try:
            payload, _ = sock.recvfrom(65535)
        except socket.timeout:
            payload = None
        if payload:
            dgrams += 1
            count_links(payload)
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
