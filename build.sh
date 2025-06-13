#!/usr/bin/env bash
set -euo pipefail

# 1) Clean old build
rm -rf build/

# 2) Ensure patchelf is installed
if ! command -v patchelf &> /dev/null; then
  echo "Installing patchelf…"
  sudo apt update && sudo apt install -y patchelf
fi

# 3) Run Nuitka
nuitka /home/kunal/Projects/BSS/EdgeComputing/main.py \
  --standalone \
  --remove-output \
  --output-dir=build \
  --nofollow-import-to=tkinter \
  --include-package=ultralytics \
  --include-package=cv2 \
  --include-package=PIL \
  --include-data-files=config.xml=config.xml

echo "✅ Build complete. Self‑contained binary is in /home/kunal/Projects/BSS/EdgeComputing/build/main.dist/main"
