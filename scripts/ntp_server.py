#!/usr/bin/env python3
"""Minimal LAN SNTP server, for a mesh router with no internet.

Nodes sync to it so their frame timestamps line up across the mesh to about a millisecond; an
unsynced mesh still runs, but frames from different nodes can no longer be aligned.
"""

import socket
import struct
import sys
import time

NTP_EPOCH = 2208988800  # seconds from 1900-01-01 (NTP epoch) to 1970-01-01 (Unix epoch)


def _ntp_timestamp(unix_time_s: float):
    ntp_seconds = int(unix_time_s) + NTP_EPOCH
    ntp_fraction = int((unix_time_s - int(unix_time_s)) * (1 << 32)) & 0xFFFFFFFF
    return ntp_seconds & 0xFFFFFFFF, ntp_fraction


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 123
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except PermissionError:
        sys.exit(f"ERROR: permission denied for udp/{port}. \n"
                 f"port 123 requires root on this OS.\n"
                 f"run with: sudo .venv/bin/python scripts/ntp_server.py\n"
                 f"(or use a high port for testing: ntp_server.py 1230, but ESP nodes won't find it by default)")
    except Exception as error:
        sys.exit(f"ERROR: could not bind to udp/{port}: {error}")

    print(f"SNTP server on udp/{port}  (stratum 1, system clock)  — Ctrl+C to stop")
    print("make sure your ESP32 Config.h has PC_IP set to this machine's IP.")
    served_count = 0
    while True:
        data, client_address = sock.recvfrom(512)
        receive_time_s = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] request from {client_address[0]}")
        if len(data) < 48:
            print(f"  short packet ({len(data)} bytes), ignoring")
            continue
        origin = data[40:48]  # Client's transmit timestamp (echoed).
        reference_ntp_s, reference_ntp_fraction = _ntp_timestamp(receive_time_s - 1.0)
        receive_ntp_s, receive_ntp_fraction = _ntp_timestamp(receive_time_s)
        tx_ntp_s, tx_ntp_fraction = _ntp_timestamp(time.time())
        reply = struct.pack("!B B b b", (0 << 6) | (4 << 3) | 4, 1, 4, -20)  # LI=0,VN=4,Mode=4 server
        reply += struct.pack("!I", 0) + struct.pack("!I", 0) + b"LOCL"        # root delay/disp, refid
        reply += struct.pack("!II", reference_ntp_s, reference_ntp_fraction)  # reference timestamp
        reply += origin                                                      # originate (client's TX)
        reply += struct.pack("!II", receive_ntp_s, receive_ntp_fraction)   # receive timestamp
        reply += struct.pack("!II", tx_ntp_s, tx_ntp_fraction)             # transmit timestamp
        sock.sendto(reply, client_address)
        served_count += 1
        if served_count % 20 == 1:
            print(f"served {served_count} requests (last: {client_address[0]})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
