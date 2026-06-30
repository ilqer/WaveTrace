#!/usr/bin/env python3
"""Minimal LAN SNTP server — the shared clock for the WaveTrace mesh when there is NO internet.

Run on the PC (the one at PC_IP). The ESP nodes have SNTP_SERVER=PC_IP, so they discipline their
clocks to this machine and the PC can align CSI frames across nodes (ms-level; that is all the
learned fusion needs — there is no cross-node phase coherence regardless).

    sudo .venv/bin/python scripts/ntp_server.py        # UDP port 123 needs root

If the deployment router DOES have internet, you don't need this — point SNTP_SERVER at a real NTP
host instead. Without a synced clock the mesh still runs; only cross-node fusion alignment is lost.
"""

import socket
import struct
import sys
import time

NTP_EPOCH = 2208988800  # seconds from 1900-01-01 (NTP epoch) to 1970-01-01 (Unix epoch)


def _ntp_ts(t: float):
    """Unix seconds -> (NTP seconds, NTP fraction) 32-bit pair."""
    sec = int(t) + NTP_EPOCH
    frac = int((t - int(t)) * (1 << 32)) & 0xFFFFFFFF
    return sec & 0xFFFFFFFF, frac


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 123
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", port))
    except PermissionError:
        sys.exit(f"ERROR: Permission denied for udp/{port}. \n"
                 f"Standard NTP uses port 123, which requires root on this OS.\n"
                 f"Run with: sudo .venv/bin/python scripts/ntp_server.py\n"
                 f"(Or use a high port for testing: ntp_server.py 1230, but ESP nodes won't find it by default)")
    except Exception as e:
        sys.exit(f"ERROR: Could not bind to udp/{port}: {e}")

    print(f"SNTP server on udp/{port}  (stratum 1, system clock)  — Ctrl+C to stop")
    print(f"Make sure your ESP32 config.h has PC_IP set to this machine's IP.")
    served = 0
    while True:
        data, addr = sock.recvfrom(512)
        recv_t = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] request from {addr[0]}")
        if len(data) < 48:
            print(f"  ignoring short packet ({len(data)} bytes)")
            continue
        origin = data[40:48]  # client's transmit timestamp -> echoed as our originate timestamp
        ref_s, ref_f = _ntp_ts(recv_t - 1.0)
        rec_s, rec_f = _ntp_ts(recv_t)
        tx_s, tx_f = _ntp_ts(time.time())
        pkt = struct.pack("!B B b b", (0 << 6) | (4 << 3) | 4, 1, 4, -20)  # LI=0,VN=4,Mode=4 server
        pkt += struct.pack("!I", 0) + struct.pack("!I", 0) + b"LOCL"        # root delay/disp, refid
        pkt += struct.pack("!II", ref_s, ref_f)                            # reference timestamp
        pkt += origin                                                      # originate (client's TX)
        pkt += struct.pack("!II", rec_s, rec_f)                            # receive timestamp
        pkt += struct.pack("!II", tx_s, tx_f)                              # transmit timestamp
        sock.sendto(pkt, addr)
        served += 1
        if served % 20 == 1:
            print(f"served {served} requests (last: {addr[0]})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
