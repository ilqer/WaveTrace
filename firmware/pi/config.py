"""Runtime config for the Pi 5 GHz CSI node. Fill in the three TODO values, then run pi5_csi_node.py.

This node is additive: it shows up to the host as node 5 alongside the ESP32 mesh (1-4).
"""

# --- Backhaul: Pi (eth0) -> Mac. Same UDP port the ESP nodes use. ---
# PC_IP MUST be the Mac's address on the wired backhaul LAN. A wrong PC_IP (or a macOS firewall
# blocking UDP 9876) is the #1 cause of an empty mesh_verify/health that looks like a capture bug.
PC_IP = "TODO_MAC_LAN_IP"          # e.g. "10.8.1.103"
UDP_PORT = 9876

# --- Node identity ---
NODE_ID = 5  # unique vs ESP nodes 1-4

# --- Sensing link: modem B 5 GHz BSSID = the TRANSMITTER we measure CSI from. ---
# MUST equal the -m argument passed to makecsiparams (see start_capture.sh). On the host this
# becomes the per-link tx id: link = (last-two-octets-of-AP_BSSID -> node 5).
AP_BSSID = "TODO_MODEM_B_5G_BSSID"  # e.g. "aa:bb:cc:dd:ee:ff"

# Modem B channel/width, used only by start_capture.sh's makecsiparams call. HT80 -> "36/80".
CHANNEL_SPEC = "36/80"

# --- Local: nexmon firmware -> this host (never leaves the Pi). ---
NEXMON_PORT = 5500

# --- CSI width / wire format ---
# HT80 on the CYW43455 reports 256 subcarriers. Off-width frames are dropped at the source.
EXPECT_S = 256
# Wire version: 3 = int16 I/Q (keeps absolute amplitude for the weapon feature). The host accepts
# 2 and 3; the ESP nodes still send 2. Use 3 here so both presence AND weapon work.
WIRE_VER = 3
# int16 scale. FIXED (never per-frame): Nexmon CSI is already int16-range, so 1.0 is pass-through
# and keeps amplitude comparable across frames. Do NOT set this to a per-frame auto-scale for weapon.
CSI_SCALE = 1.0


def validate() -> None:
    """Fail loudly if the TODO placeholders were not filled in (called at node startup)."""
    missing = [name for name in ("PC_IP", "AP_BSSID") if globals()[name].startswith("TODO_")]
    if missing:
        raise SystemExit(
            f"[config] set {', '.join(missing)} in firmware/pi/config.py before running "
            f"(PC_IP = Mac LAN IP, AP_BSSID = modem B 5 GHz BSSID)."
        )
