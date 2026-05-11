#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$SCRIPT_DIR}"
WEBOTS_BIN="${WEBOTS_BIN:-$(command -v webots || true)}"
if [[ -z "$WEBOTS_BIN" && -x /usr/local/webots/webots ]]; then
  WEBOTS_BIN="/usr/local/webots/webots"
fi
if [[ -z "$WEBOTS_BIN" && -x /Applications/Webots.app/Contents/MacOS/webots ]]; then
  WEBOTS_BIN="/Applications/Webots.app/Contents/MacOS/webots"
fi
WORLD_FILE="$REPO_DIR/nao_VLM/worlds/nao_VLM.wbt"

cd "$REPO_DIR"
if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "[error] Webots executable not found. Set WEBOTS_BIN=/path/to/webots"
  exit 1
fi
cp .env.debug.example-video .env

export DISPLAY="${DISPLAY:-:0}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"

exec "$WEBOTS_BIN" --mode=realtime "$WORLD_FILE"
