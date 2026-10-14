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

_HDR = struct.Struct("<BBBQH")  # 13 bytes; must match wavetrace/adapters/source/wire_format.py
_MAGIC = 0x57

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    # absorb bursts, as Source.bind_udp_socket does
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
except OSError:
    pass
sock.bind(("0.0.0.0", PORT))
sock.settimeout(1.0)

total = collections.Counter()   # (tx, rx) -> all-time frames
window = collections.Counter()  # (tx, rx) -> frames this second
datagram_count = 0
start = last_print_time_s = time.time()
print(f"listening on udp/{PORT}  (Ctrl+C to stop)")


def count_links(payload):
    """Tally (tx_short, rx_node) per record in one batch, stepping over the CSI bytes undecoded."""
    if len(payload) < _HDR.size:
        return
    magic, wire_version, rx, _ntp, record_count = _HDR.unpack_from(payload, 0)
    if magic != _MAGIC or wire_version not in (2, 3):
        return
    bytes_per_subcarrier = 2 if wire_version == 2 else 4   # int8 (2) vs int16 (4) I/Q pair
    offset_bytes, end_offset = _HDR.size, len(payload)
    for _ in range(record_count):
        if offset_bytes + 12 > end_offset:
            break
        # last two octets of the tx MAC
        tx = f"{payload[offset_bytes + 4]:02x}:{payload[offset_bytes + 5]:02x}"
        csi_length_bytes = struct.unpack_from("<H", payload, offset_bytes + 10)[0]
        offset_bytes += 12
        if csi_length_bytes % bytes_per_subcarrier != 0 or offset_bytes + csi_length_bytes > end_offset:
            break
        offset_bytes += csi_length_bytes
        total[(tx, rx)] += 1
        window[(tx, rx)] += 1


try:
    while time.time() - start < RUN_S:
        try:
            payload, _ = sock.recvfrom(65535)
        except socket.timeout:
            payload = None
        if payload:
            datagram_count += 1
            count_links(payload)
        now = time.time()
        if now - last_print_time_s >= 1.0:
            links = " ".join(f"{tx}->{rx}:{window[(tx, rx)]}"
                             for (tx, rx) in sorted(total)) or "(no frames)"
            print(f"[{now - start:5.1f}s] hz  {links}")
            window.clear()
            last_print_time_s = now
except KeyboardInterrupt:
    pass
finally:
    sock.close()

print("\n=== totals (tx -> rx : frames) ===")
for (tx, rx) in sorted(total):
    print(f"  {tx} -> {rx} : {total[(tx, rx)]}")
print(f"distinct links seen: {len(total)}   datagrams: {datagram_count}")
