#!/usr/bin/env bash
# Dronez setup for WSL2 Ubuntu / Linux (primary platform, see docs/adr/0001).
# Keep the repo on the Linux filesystem (e.g. ~/projects/dronez), NOT under /mnt/c.
set -euo pipefail

cd "$(dirname "$0")/.."

case "$PWD" in
  /mnt/*) echo "warning: repo is on the Windows filesystem ($PWD); expect slow I/O. Clone into ~/ instead." >&2 ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Offscreen rendering for MuJoCo videos (EGL, OSMesa fallback). Needs sudo once.
if ! ldconfig -p 2>/dev/null | grep -q libEGL.so; then
  echo "note: for MP4 rendering install EGL/OSMesa: sudo apt-get install -y libegl1 libosmesa6 ffmpeg" >&2
fi

uv sync
uv run pre-commit install || true

echo
echo "Data dir (runs, models, videos): ${DRONEZ_DATA_DIR:-$HOME/.local/share/dronez}"
echo "Next: uv run dronez-build-model && uv run pytest -m fast"
