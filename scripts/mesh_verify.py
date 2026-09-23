"""Count CSI arrivals per directed (tx -> rx) link and print their rates.

Wire format (binary, little-endian): each datagram is a 13-byte header {magic, ver, node, ntp_ms, n}
where `node` is the receiver and ver is 2 (ESP int8) or 3 (Pi int16), followed by `n` records
mac[6]|ts(u32)|len(u16)|CSI bytes whose mac is the transmitter, so a link is (tx_mac_short ->
rx_node) and N nodes give N*(N-1) links at roughly equal rates. The header is parsed here instead of
through the pipeline, so this runs with no DSP or numpy dependency."""

import collections
import socket
import struct
import sys
import time

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9876
RUN_S = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9  # default: until Ctrl+C

_HDR = struct.Struct("<BBBQH")  # 13 bytes; must match wavetrace/Source.py
_MAGIC = 0x57

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)  # absorb bursts, as Source.bindUdp does
except OSError:
    pass
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

total = collections.Counter()   # (tx, rx) -> all-time frames
window = collections.Counter()  # (tx, rx) -> frames this second
dgrams = 0
start = lastPrint = time.time()
print(f"listening on udp/{PORT}  (Ctrl+C to stop)")


def countLinks(payload):
    """Tally (tx_short, rx_node) per record in one batch, stepping over the CSI bytes undecoded."""
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
            countLinks(payload)
        now = time.time()
        if now - lastPrint >= 1.0:
            links = " ".join(f"{tx}->{rx}:{window[(tx, rx)]}"
                             for (tx, rx) in sorted(total)) or "(no frames)"
            print(f"[{now - start:5.1f}s] hz  {links}")
            window.clear()
            lastPrint = now
except KeyboardInterrupt:
    pass
finally:
    sock.close()

print("\n=== totals (tx -> rx : frames) ===")
for (tx, rx) in sorted(total):
    print(f"  {tx} -> {rx} : {total[(tx, rx)]}")
print(f"distinct links seen: {len(total)}   datagrams: {dgrams}")
