#!/usr/bin/env python3
"""Entry point: stream 5 GHz CSI from the Pi's Nexmon firmware to the WaveTrace host as node 5.

  Nexmon firmware --(UDP 5500, on-Pi)--> NexmonReader --> quantize --> BatchPublisher
                                                                  --(UDP 9876, eth0)--> Mac

Run on the Pi after Part A (firmware) and Part B (modem B pinned + Mac illuminating) of the plan:
    python3 pi5_csi_node.py
Edit pi/config.py first (PC_IP, AP_BSSID, EXPECT_S).
"""
import time

import config
from nexmon_reader import NexmonReader
from publisher import BatchPublisher, mac_to_bytes, quantize_csi


def main() -> None:
    ap_mac = mac_to_bytes(config.AP_BSSID)
    reader = NexmonReader(config.NEXMON_PORT, expect_s=config.EXPECT_S, ap_mac=ap_mac)
    pub = BatchPublisher(config.PC_IP, config.UDP_PORT, config.NODE_ID, config.AP_BSSID)

    print(
        f"[pi5-csi] node={config.NODE_ID} -> {config.PC_IP}:{config.UDP_PORT} | "
        f"AP={config.AP_BSSID} | S={config.EXPECT_S} | nexmon:{config.NEXMON_PORT}",
        flush=True,
    )

    sent = 0
    t_report = time.monotonic()
    t_flush = t_report
    try:
        for _ts, _mac, csi in reader.frames():
            # ts_us: monotonic microsecond counter (host only needs consistent per-link spacing).
            ts_us = time.monotonic_ns() // 1000
            pub.add(quantize_csi(csi, config.CSI_SCALE), ts_us)
            sent += 1

            now = time.monotonic()
            # Bound latency at low rates: force a flush ~50 ms even if the MTU batch isn't full.
            if now - t_flush >= 0.05:
                pub.flush()
                t_flush = now
            if now - t_report >= 1.0:
                print(f"[pi5-csi] {sent} frames/s (S={csi.size})", flush=True)
                sent = 0
                t_report = now
    except KeyboardInterrupt:
        pass
    finally:
        pub.close()
        print("\n[pi5-csi] stopped", flush=True)


if __name__ == "__main__":
    main()
