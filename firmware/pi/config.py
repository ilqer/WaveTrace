"""Runtime config for the Pi 5 GHz CSI node. Edit for your deployment, then run pi5_csi_node.py.

This node is additive: it shows up to the host as node 5 alongside the ESP32 mesh (1-4)."""

# --- Backhaul: Pi (eth0) -> Mac. Same UDP port the ESP nodes use. ---
# PC_IP MUST be the Mac's address on the backhaul LAN the Pi's wired eth0 is on.
# A wrong PC_IP (or a macOS firewall blocking UDP 9876) is the #1 cause of an empty
# mesh_verify/health that looks like a capture bug but isn't.
PC_IP = "10.8.1.103"
UDP_PORT = 9876

# --- Node identity ---
NODE_ID = 5  # unique vs ESP nodes 1-4

# --- Sensing link: modem B 5 GHz BSSID = the TRANSMITTER we measure CSI from. ---
# MUST equal the -m argument you passed to makecsiparams. This becomes the per-link
# tx identity on the host: link = (last-two-octets-of-AP_BSSID -> node 5).
AP_BSSID = "aa:bb:cc:dd:ee:ff"

# --- Local: nexmon firmware -> this host (never leaves the Pi). ---
NEXMON_PORT = 5500

# --- CSI width / quantization ---
# Pin one subcarrier width and drop off-width frames at the source (HT40 -> 128).
# Set to None to accept whatever the first frame reports (then it's locked).
EXPECT_S = 128
# int8 wire scale. None = per-frame auto-scale (peak component -> 127). A fixed float
# gives a stable scale across frames if you prefer (calibration normalizes either way).
CSI_SCALE = None
