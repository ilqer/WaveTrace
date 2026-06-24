#!/usr/bin/env bash
# One-time Nexmon CSI firmware setup for the Pi 5 onboard chip (CYW43455), 5 GHz.
# Run ON THE PI over SSH:  bash setup_nexmon.sh
#
# This builds + installs patched firmware, nexutil, and makecsiparams. It is the heavy one-time
# step; per-boot capture params live in start_capture.sh. Re-running is safe (each step guards
# itself). Pi 5 + onboard CSI is bleeding-edge: if a step fails, check the exact branch/kernel in
# nexmon_csi Discussion #395 and the nexmonster/nexmon_csi fork, then re-run.
set -euo pipefail

NEXMON_DIR="${HOME}/nexmon"
CHIP="bcm43455c0"   # CYW43455 family

echo "==> 0/6  Pi 5 kernel pin (Nexmon needs the 4KB-page kernel)"
CONFIG_TXT=/boot/firmware/config.txt
if ! grep -q '^kernel=kernel8.img' "$CONFIG_TXT" 2>/dev/null; then
  echo "kernel=kernel8.img" | sudo tee -a "$CONFIG_TXT" >/dev/null
  echo "    added 'kernel=kernel8.img' -> REBOOT, then re-run this script."
  echo "    (reboot now with: sudo reboot)"
  exit 0
fi
echo "    kernel pin present."

echo "==> 1/6  build dependencies"
sudo apt-get update -y
sudo apt-get install -y raspberrypi-kernel-headers git libgmp3-dev gawk qpdf \
  bison flex make automake texinfo libtool-bin tcpdump

echo "==> 2/6  clone nexmon"
if [ ! -d "$NEXMON_DIR" ]; then
  git clone https://github.com/seemoo-lab/nexmon.git "$NEXMON_DIR"
fi
cd "$NEXMON_DIR"

echo "==> 3/6  build base toolchain (libISL / firmware extraction)"
# shellcheck disable=SC1091
source ./setup_env.sh
make >/dev/null

echo "==> 4/6  clone + build nexmon_csi for ${CHIP}"
PATCH_DIR="$NEXMON_DIR/patches/${CHIP}"
# Pick the kernel-tagged subdir Nexmon created for this chip (one dir under the chip).
KVER_DIR="$(find "$PATCH_DIR" -maxdepth 1 -mindepth 1 -type d | head -n1)"
if [ -z "$KVER_DIR" ]; then echo "ERROR: no patch dir under $PATCH_DIR (chip not extracted)"; exit 1; fi
cd "$KVER_DIR"
if [ ! -d nexmon_csi ]; then
  git clone https://github.com/nexmonster/nexmon_csi.git
fi
cd nexmon_csi
make

echo "==> 5/6  install patched firmware + nexutil + makecsiparams"
make install-firmware
( cd "$NEXMON_DIR/utilities/nexutil" && make && sudo make install )
( cd makecsiparams && make && sudo cp makecsiparams /usr/local/bin/mcp )

echo "==> 6/6  verify"
which nexutil mcp
echo "    firmware + tools installed. Next: bash start_capture.sh  (sets the CSI filter + monitor mode)."
