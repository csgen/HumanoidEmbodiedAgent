#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEFAULT_WEBOTS_BIN="/home/darian/.local/opt/webots/webots"
WEBOTS_BIN="${WEBOTS_BIN:-$(command -v webots || true)}"
if [[ -z "$WEBOTS_BIN" && -x "$DEFAULT_WEBOTS_BIN" ]]; then
  WEBOTS_BIN="$DEFAULT_WEBOTS_BIN"
fi
WORLD_FILE="$REPO_DIR/nao_VLM/worlds/nao_VLM.wbt"
DEFAULT_CONDA_ENV="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian"
DEFAULT_FFPLAY_BIN="$DEFAULT_CONDA_ENV/bin/ffplay"
DEFAULT_WMCTRL_BIN="$DEFAULT_CONDA_ENV/bin/wmctrl"
FFPLAY_BIN="${FFPLAY_BIN:-$(command -v ffplay || true)}"
WMCTRL_BIN="${WMCTRL_BIN:-$(command -v wmctrl || true)}"
if [[ -z "$FFPLAY_BIN" && -x "$DEFAULT_FFPLAY_BIN" ]]; then
  FFPLAY_BIN="$DEFAULT_FFPLAY_BIN"
fi
if [[ -z "$WMCTRL_BIN" && -x "$DEFAULT_WMCTRL_BIN" ]]; then
  WMCTRL_BIN="$DEFAULT_WMCTRL_BIN"
fi
XPROP_BIN="/usr/bin/xprop"

VIDEO_PATH="${1:-$REPO_DIR/example_video/webcam_20260425_072825.mp4}"
MODEL_ID="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
DEMO_MODE="${DEMO_MODE:-oneshot}"
REPLAY_CODE_PATH="${REPLAY_CODE_PATH:-}"
VIDEO_TITLE="${VIDEO_TITLE:-Demo Source Video}"
VIDEO_GEOM="${VIDEO_GEOM:-0,0,0,820,700}"
WEBOTS_GEOM="${WEBOTS_GEOM:-0,830,0,1050,700}"
FFPLAY_LOG="${FFPLAY_LOG:-/tmp/demo_ffplay.log}"
WEBOTS_LOG="${WEBOTS_LOG:-/tmp/demo_webots.log}"
STARTUP_SLEEP="${STARTUP_SLEEP:-1}"
RAISE_DEMO_WINDOWS="${RAISE_DEMO_WINDOWS:-1}"
DEMO_WORKSPACE_INDEX="${DEMO_WORKSPACE_INDEX:-1}"
WINDOW_STABILIZE_SECONDS="${WINDOW_STABILIZE_SECONDS:-16}"
ONE_SHOT_POST_EXECUTION_SECONDS_VALUE="${ONE_SHOT_POST_EXECUTION_SECONDS:-5}"
ONE_SHOT_EXIT_AFTER_EXECUTE_VALUE="${ONE_SHOT_EXIT_AFTER_EXECUTE:-1}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[Error] [错误] Video does not exist: [视频不存在:] $VIDEO_PATH"
  exit 1
fi

if [[ ! -x "$WEBOTS_BIN" ]]; then
  echo "[Error] [错误] Webots missing or not executable: [Webots 不存在或不可执行:] $WEBOTS_BIN"
  exit 1
fi

if [[ ! -x "$FFPLAY_BIN" ]]; then
  echo "[Error] [错误] ffplay does not exist: [ffplay 不存在:] $FFPLAY_BIN"
  exit 1
fi

cat > "$REPO_DIR/.env" <<EOF
llm_api_key=
base_url=
INPUT_MODE=webcam
RUN_MODE=$DEMO_MODE
WEBCAM_SOURCE=$VIDEO_PATH
FRAMEBUFFER_BACKEND=auto
FRAME_BUFFER_SECONDS=8
FRAME_BUFFER_FPS=10
FRAMEBUFFER_WIDTH=1280
FRAMEBUFFER_HEIGHT=720
VLM_BACKEND=local
LOCAL_VLM_MODEL=$MODEL_ID
LOCAL_VLM_SERVER_URL=${LOCAL_VLM_SERVER_URL:-}
LOCAL_VLM_LOAD_IN_4BIT=${LOCAL_VLM_LOAD_IN_4BIT:-1}
LOCAL_VLM_NUM_CANDIDATES=${LOCAL_VLM_NUM_CANDIDATES:-3}
LOCAL_VLM_DEBUG=${LOCAL_VLM_DEBUG:-1}
VLM_MODEL=gpt-4o
VLM_FRAME_COUNT=6
VLM_WINDOW_SECONDS=5.0
VLM_MAX_TOKENS=${VLM_MAX_TOKENS:-700}
ONE_SHOT_BUFFER_TIMEOUT=8
ONE_SHOT_VLM_TIMEOUT=120
ONE_SHOT_VIDEO_SETTLE_SECONDS=6.0
ONE_SHOT_POST_EXECUTION_SECONDS=$ONE_SHOT_POST_EXECUTION_SECONDS_VALUE
ONE_SHOT_EXIT_AFTER_EXECUTE=$ONE_SHOT_EXIT_AFTER_EXECUTE_VALUE
VLM_SCENARIO_HINT=
REPLAY_CODE_PATH=$REPLAY_CODE_PATH
EOF

export DISPLAY=:0
export XDG_SESSION_TYPE=x11

"$FFPLAY_BIN" -window_title "$VIDEO_TITLE" -left 0 -top 0 -x 1280 -y 1400 -loop 0 "$VIDEO_PATH" >"$FFPLAY_LOG" 2>&1 &
FFPLAY_PID=$!

sleep "$STARTUP_SLEEP"
"$WEBOTS_BIN" --mode=realtime --stdout --stderr "$WORLD_FILE" >"$WEBOTS_LOG" 2>&1 &
WEBOTS_PID=$!

if [[ -x "$WMCTRL_BIN" ]]; then
  "$WMCTRL_BIN" -s "$DEMO_WORKSPACE_INDEX" || true
  for _ in $(seq 1 30); do
    sleep 1
    WEBOTS_WIN_ID="$($WMCTRL_BIN -l | awk 'tolower($0) ~ /webots/ {print $1; exit}')"
    VIDEO_WIN_ID="$($WMCTRL_BIN -l | awk -v title="$VIDEO_TITLE" 'index($0, title) {print $1; exit}')"
    if [[ -n "$VIDEO_WIN_ID" ]]; then
      "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -t "$DEMO_WORKSPACE_INDEX" || true
      "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -b remove,maximized_vert,maximized_horz || true
      "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -b add,above || true
      "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -e "$VIDEO_GEOM" || true
      if [[ "$RAISE_DEMO_WINDOWS" == "1" ]]; then
        "$WMCTRL_BIN" -i -a "$VIDEO_WIN_ID" || true
      fi
    fi
    if [[ -n "$WEBOTS_WIN_ID" ]]; then
      "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -t "$DEMO_WORKSPACE_INDEX" || true
      "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -b remove,maximized_vert,maximized_horz || true
      "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -b remove,above || true
      "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -e "$WEBOTS_GEOM" || true
      if [[ "$RAISE_DEMO_WINDOWS" == "1" ]]; then
        "$WMCTRL_BIN" -i -a "$WEBOTS_WIN_ID" || true
      fi
    fi
    if [[ -n "$VIDEO_WIN_ID" && -n "$WEBOTS_WIN_ID" ]]; then
      break
    fi
  done

  (
    end_ts=$((SECONDS + WINDOW_STABILIZE_SECONDS))
    while (( SECONDS < end_ts )); do
      "$WMCTRL_BIN" -s "$DEMO_WORKSPACE_INDEX" || true
      WEBOTS_WIN_ID="$($WMCTRL_BIN -l | awk 'tolower($0) ~ /webots/ {print $1; exit}')"
      VIDEO_WIN_ID="$($WMCTRL_BIN -l | awk -v title="$VIDEO_TITLE" 'index($0, title) {print $1; exit}')"
      if [[ -n "$VIDEO_WIN_ID" ]]; then
        "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -t "$DEMO_WORKSPACE_INDEX" || true
        "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -e "$VIDEO_GEOM" || true
        "$WMCTRL_BIN" -i -r "$VIDEO_WIN_ID" -b add,above || true
      fi
      if [[ -n "$WEBOTS_WIN_ID" ]]; then
        "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -t "$DEMO_WORKSPACE_INDEX" || true
        "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -e "$WEBOTS_GEOM" || true
        "$WMCTRL_BIN" -i -r "$WEBOTS_WIN_ID" -b remove,above || true
      fi
      if [[ "$RAISE_DEMO_WINDOWS" == "1" ]]; then
        [[ -n "$VIDEO_WIN_ID" ]] && "$WMCTRL_BIN" -i -a "$VIDEO_WIN_ID" || true
        [[ -n "$WEBOTS_WIN_ID" ]] && "$WMCTRL_BIN" -i -a "$WEBOTS_WIN_ID" || true
      fi
      sleep 1
    done
  ) >/dev/null 2>&1 &
fi

echo "Written to .env and started comparative demo. [已写入 .env 并启动对照演示。]"
echo "Left source video: [左侧源视频:] $VIDEO_PATH"
echo "Right Webots: [右侧 Webots:] $WORLD_FILE"
echo "ffplay PID=$FFPLAY_PID"
echo "webots PID=$WEBOTS_PID"
echo "Workspace index: [工作区:] $DEMO_WORKSPACE_INDEX"
echo "Ready to start screen recording. [可直接开始录屏。]"

if [[ "${PRINT_PIDS_ONLY:-0}" == "1" ]]; then
  printf 'FFPLAY_PID=%s\nWEBOTS_PID=%s\n' "$FFPLAY_PID" "$WEBOTS_PID"
fi
