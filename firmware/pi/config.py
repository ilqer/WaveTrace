"""Runtime config for Pi 5 GHz CSI node. Fill TODOs and run pi5_csi_node.py.
Appears as node 5 alongside ESP32 mesh."""

# Backhaul to Mac on same UDP port as ESP. Must be LAN IP.
PC_IP = "TODO_MAC_LAN_IP"          # e.g. "10.8.1.103"
UDP_PORT = 9876

# --- Node identity ---
NODE_ID = 5  # unique vs ESP nodes 1-4

# Modem B 5 GHz BSSID. Must match -m in makecsiparams.
AP_BSSID = "TODO_MODEM_B_5G_BSSID"  # e.g. "aa:bb:cc:dd:ee:ff"

# Channel/width for makecsiparams (e.g. "36/80").
CHANNEL_SPEC = "36/80"

# Local nexmon firmware to host port.
NEXMON_PORT = 5500

# CSI config. HT80 reports 256 subcarriers.
EXPECT_S = 256
# Wire version 3: int16 I/Q. Keeps absolute amplitude for weapon feature.
WIRE_VER = 3
# Fixed int16 scale (1.0). Do not auto-scale or weapon feature fails.
CSI_SCALE = 1.0


def validate() -> None:
    """Fail loudly if the TODO placeholders were not filled in (called at node startup)."""
    missing = [name for name in ("PC_IP", "AP_BSSID") if globals()[name].startswith("TODO_")]
    if missing:
        raise SystemExit(
            f"[config] set {', '.join(missing)} in firmware/pi/config.py before running "
            f"(PC_IP = Mac LAN IP, AP_BSSID = modem B 5 GHz BSSID)."
        )
