#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/darian/桌面/humanoidRobot"
WEBOTS_BIN="/home/darian/.local/opt/webots/webots"
WORLD_FILE="$REPO_DIR/nao_VLM/worlds/nao_VLM.wbt"
FFPLAY_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/ffplay"
WMCTRL_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/wmctrl"

VIDEO_PATH="${1:-$REPO_DIR/example_video/webcam_20260425_072825.mp4}"
MODEL_ID="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
VIDEO_TITLE="${VIDEO_TITLE:-Demo Source Video}"
VIDEO_GEOM="${VIDEO_GEOM:-0,0,0,820,700}"
WEBOTS_GEOM="${WEBOTS_GEOM:-0,830,0,1050,700}"
FFPLAY_LOG="${FFPLAY_LOG:-/tmp/demo_ffplay.log}"
WEBOTS_LOG="${WEBOTS_LOG:-/tmp/demo_webots.log}"
STARTUP_SLEEP="${STARTUP_SLEEP:-1}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[错误] 视频不存在: $VIDEO_PATH"
  exit 1
fi

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "[错误] Webots 不存在或不可执行: $WEBOTS_BIN"
  exit 1
fi

if [[ ! -x "$FFPLAY_BIN" ]]; then
  echo "[错误] ffplay 不存在: $FFPLAY_BIN"
  exit 1
fi

cat > "$REPO_DIR/.env" <<EOF
llm_api_key=
base_url=
INPUT_MODE=webcam
RUN_MODE=oneshot
WEBCAM_SOURCE=$VIDEO_PATH
FRAMEBUFFER_BACKEND=auto
FRAME_BUFFER_SECONDS=8
FRAME_BUFFER_FPS=10
FRAMEBUFFER_WIDTH=1280
FRAMEBUFFER_HEIGHT=720
VLM_BACKEND=local
LOCAL_VLM_MODEL=$MODEL_ID
LOCAL_VLM_SERVER_URL=${LOCAL_VLM_SERVER_URL:-}
VLM_MODEL=gpt-4o
VLM_FRAME_COUNT=6
VLM_WINDOW_SECONDS=5.0
ONE_SHOT_BUFFER_TIMEOUT=8
ONE_SHOT_VLM_TIMEOUT=120
ONE_SHOT_VIDEO_SETTLE_SECONDS=6.0
VLM_SCENARIO_HINT=
EOF

export DISPLAY=:0
export XDG_SESSION_TYPE=x11

"$FFPLAY_BIN" -window_title "$VIDEO_TITLE" -left 0 -top 0 -x 820 -y 640 -autoexit "$VIDEO_PATH" >"$FFPLAY_LOG" 2>&1 &
FFPLAY_PID=$!

sleep "$STARTUP_SLEEP"
"$WEBOTS_BIN" --mode=realtime "$WORLD_FILE" >"$WEBOTS_LOG" 2>&1 &
WEBOTS_PID=$!

if [[ -x "$WMCTRL_BIN" ]]; then
  for _ in $(seq 1 30); do
    sleep 1
    WEBOTS_WIN_ID="$($WMCTRL_BIN -l | awk 'tolower($0) ~ /webots/ {print $1; exit}')"
    VIDEO_WIN_ID="$($WMCTRL_BIN -l | awk -v title="$VIDEO_TITLE" 'index($0, title) {print $1; exit}')"
    if [[ -n "$VIDEO_WIN_ID" ]]; then
      "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -e "$VIDEO_GEOM" || true
    fi
    if [[ -n "$WEBOTS_WIN_ID" ]]; then
      "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -e "$WEBOTS_GEOM" || true
      break
    fi
  done
fi

echo "已写入 .env 并启动对照演示。"
echo "左侧源视频: $VIDEO_PATH"
echo "右侧 Webots: $WORLD_FILE"
echo "ffplay PID=$FFPLAY_PID"
echo "webots PID=$WEBOTS_PID"
echo "可直接开始录屏。"

if [[ "${PRINT_PIDS_ONLY:-0}" == "1" ]]; then
  printf 'FFPLAY_PID=%s\nWEBOTS_PID=%s\n' "$FFPLAY_PID" "$WEBOTS_PID"
fi
