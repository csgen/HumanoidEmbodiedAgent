#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
WEBOTS_BIN="${WEBOTS_BIN:-$(command -v webots || true)}"
if [[ -z "$WEBOTS_BIN" && -x /usr/local/webots/webots ]]; then
  WEBOTS_BIN="/usr/local/webots/webots"
fi
if [[ -z "$WEBOTS_BIN" && -x /Applications/Webots.app/Contents/MacOS/webots ]]; then
  WEBOTS_BIN="/Applications/Webots.app/Contents/MacOS/webots"
fi

WORLD_FILE="$REPO_DIR/nao_VLM/worlds/nao_VLM.wbt"
WEBCAM_SOURCE_VALUE="${WEBCAM_SOURCE:-0}"
VLM_BACKEND_VALUE="${VLM_BACKEND:-openai}"

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "[error] Webots executable not found. Set WEBOTS_BIN=/path/to/webots"
  exit 1
fi

cat > "$REPO_DIR/.env" <<EOF
llm_api_key=${llm_api_key:-${OPENAI_API_KEY:-}}
base_url=${base_url:-}
INPUT_MODE=webcam
RUN_MODE=periodic
WEBCAM_SOURCE=$WEBCAM_SOURCE_VALUE
FRAMEBUFFER_BACKEND=auto
FRAME_BUFFER_SECONDS=${FRAME_BUFFER_SECONDS:-2}
FRAME_BUFFER_FPS=${FRAME_BUFFER_FPS:-10}
FRAMEBUFFER_WIDTH=${FRAMEBUFFER_WIDTH:-1280}
FRAMEBUFFER_HEIGHT=${FRAMEBUFFER_HEIGHT:-720}
VLM_BACKEND=$VLM_BACKEND_VALUE
VLM_MODEL=${VLM_MODEL:-gpt-4o}
VLM_FRAME_COUNT=${VLM_FRAME_COUNT:-5}
VLM_WINDOW_SECONDS=${VLM_WINDOW_SECONDS:-1.5}
VLM_MAX_TOKENS=${VLM_MAX_TOKENS:-700}
LOCAL_VLM_MODEL=${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
LOCAL_VLM_LOAD_IN_4BIT=${LOCAL_VLM_LOAD_IN_4BIT:-1}
LOCAL_VLM_SERVER_URL=${LOCAL_VLM_SERVER_URL:-}
VLM_SCENARIO_HINT=${VLM_SCENARIO_HINT:-}
EOF

echo "[live] wrote $REPO_DIR/.env"
echo "[live] WEBCAM_SOURCE=$WEBCAM_SOURCE_VALUE"
echo "[live] examples:"
echo "       native Linux webcam: WEBCAM_SOURCE=0"
echo "       Mac -> Docker:       WEBCAM_SOURCE=http://host.docker.internal:5000/video_feed"
echo "       SSH tunnel remote:   WEBCAM_SOURCE=http://127.0.0.1:5000/video_feed"

cd "$REPO_DIR"
export DISPLAY="${DISPLAY:-:0}"
export XDG_SESSION_TYPE="${XDG_SESSION_TYPE:-x11}"
exec "$WEBOTS_BIN" --mode=realtime --stdout --stderr "$WORLD_FILE"
