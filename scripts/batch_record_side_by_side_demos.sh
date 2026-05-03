#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/darian/桌面/humanoidRobot"
LAUNCH_SCRIPT="$REPO_DIR/scripts/launch_side_by_side_demo.sh"
FFMPEG_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/ffmpeg"
WMCTRL_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/wmctrl"
XWININFO_BIN="/usr/bin/xwininfo"
CONDA_PY="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/python"
SCREEN_RECORD_PY="$REPO_DIR/scripts/screen_record_region.py"
PREVENT_SLEEP_SCRIPT="$REPO_DIR/scripts/prevent_display_sleep.sh"
LOCAL_VLM_SERVER_PY="$REPO_DIR/scripts/local_vlm_codegen_server.py"
LOCAL_VLM_SERVER_HOST="${LOCAL_VLM_SERVER_HOST:-127.0.0.1}"
LOCAL_VLM_SERVER_PORT="${LOCAL_VLM_SERVER_PORT:-8765}"
LOCAL_VLM_SERVER_URL="${LOCAL_VLM_SERVER_URL:-http://${LOCAL_VLM_SERVER_HOST}:${LOCAL_VLM_SERVER_PORT}}"

SAMPLES_DIR="${1:-$REPO_DIR/debug_video_samples}"
OUT_DIR="${2:-$REPO_DIR/artifacts/screen_recordings}"
MODEL_ID="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
LOCAL_VLM_NUM_CANDIDATES_VALUE="${LOCAL_VLM_NUM_CANDIDATES:-1}"
VLM_FRAME_COUNT_VALUE="${VLM_FRAME_COUNT:-4}"
VLM_MAX_TOKENS_VALUE="${VLM_MAX_TOKENS:-320}"
USE_SHORTLIST="${USE_SHORTLIST:-1}"
CAPTURE_WIDTH="${CAPTURE_WIDTH:-}"
CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-}"
CAPTURE_X="${CAPTURE_X:-0}"
CAPTURE_Y="${CAPTURE_Y:-0}"
VIDEO_GEOM="${VIDEO_GEOM:-}"
WEBOTS_GEOM="${WEBOTS_GEOM:-}"
TAIL_SECONDS="${TAIL_SECONDS:-16}"
STARTUP_GRACE="${STARTUP_GRACE:-8}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$CONDA_PY" ]]; then
  echo "[错误] conda python 不存在: $CONDA_PY"
  exit 1
fi

if [[ ! -x "$LAUNCH_SCRIPT" ]]; then
  echo "[错误] 启动脚本不存在: $LAUNCH_SCRIPT"
  exit 1
fi

if [[ ! -f "$SCREEN_RECORD_PY" ]]; then
  echo "[错误] 录屏脚本不存在: $SCREEN_RECORD_PY"
  exit 1
fi

if [[ ! -x "$PREVENT_SLEEP_SCRIPT" ]]; then
  echo "[错误] 防休眠脚本不存在或不可执行: $PREVENT_SLEEP_SCRIPT"
  exit 1
fi

cleanup_run() {
  pkill -f "/home/darian/.local/opt/webots/webots" 2>/dev/null || true
  pkill -f "ffplay.*Demo Source Video" 2>/dev/null || true
  pkill -f "ffplay.*Demo-" 2>/dev/null || true
}

PREVENT_SLEEP_PID=""
LOCAL_VLM_SERVER_PID=""

cleanup_all() {
  cleanup_run
  if [[ -n "$PREVENT_SLEEP_PID" ]]; then
    kill "$PREVENT_SLEEP_PID" 2>/dev/null || true
    wait "$PREVENT_SLEEP_PID" 2>/dev/null || true
  fi
  if [[ -n "$LOCAL_VLM_SERVER_PID" ]]; then
    kill "$LOCAL_VLM_SERVER_PID" 2>/dev/null || true
    wait "$LOCAL_VLM_SERVER_PID" 2>/dev/null || true
  fi
}

trap cleanup_all EXIT INT TERM

DISPLAY=:0 bash "$PREVENT_SLEEP_SCRIPT" >/tmp/prevent_display_sleep.log 2>&1 &
PREVENT_SLEEP_PID=$!
sleep 1

start_local_vlm_server() {
  if [[ -n "$(ps -ef | grep "$LOCAL_VLM_SERVER_PY" | grep -v grep || true)" ]]; then
    return
  fi

  LOCAL_VLM_MODEL="$MODEL_ID" \
  LOCAL_VLM_NUM_CANDIDATES="$LOCAL_VLM_NUM_CANDIDATES_VALUE" \
  VLM_FRAME_COUNT="$VLM_FRAME_COUNT_VALUE" \
  VLM_MAX_TOKENS="$VLM_MAX_TOKENS_VALUE" \
  LOCAL_VLM_SERVER_HOST="$LOCAL_VLM_SERVER_HOST" \
  LOCAL_VLM_SERVER_PORT="$LOCAL_VLM_SERVER_PORT" \
  "$CONDA_PY" "$LOCAL_VLM_SERVER_PY" >/tmp/local_vlm_codegen_server.log 2>&1 &
  LOCAL_VLM_SERVER_PID=$!

  for _ in $(seq 1 180); do
    if curl -fsS "$LOCAL_VLM_SERVER_URL/health" >/dev/null 2>&1; then
      curl -fsS -X POST "$LOCAL_VLM_SERVER_URL/warmup" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"$MODEL_ID\"}" >/dev/null 2>&1 || true
      echo "[信息] 本地 VLM 服务已就绪: $LOCAL_VLM_SERVER_URL"
      return
    fi
    sleep 2
  done

  echo "[错误] 本地 VLM 服务启动超时: $LOCAL_VLM_SERVER_URL"
  exit 1
}

detect_screen_layout() {
  local dims screen_width screen_height left_width right_width margin_x margin_top margin_bottom usable_height

  dims="$(DISPLAY=:0 xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}')"
  if [[ -z "$dims" ]]; then
    dims="$(DISPLAY=:0 xdpyinfo 2>/dev/null | awk '/dimensions:/ {print $2; exit}')"
  fi
  if [[ ! "$dims" =~ ^([0-9]+)x([0-9]+)$ ]]; then
    echo "[错误] 无法检测桌面分辨率。"
    exit 1
  fi

  screen_width="${BASH_REMATCH[1]}"
  screen_height="${BASH_REMATCH[2]}"
  margin_x=12
  margin_top=36
  margin_bottom=12
  usable_height=$((screen_height - margin_top - margin_bottom))
  left_width=$(((screen_width - margin_x * 3) / 2))
  right_width=$((screen_width - left_width - margin_x * 3))

  CAPTURE_WIDTH="${CAPTURE_WIDTH:-$screen_width}"
  CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-$screen_height}"
  VIDEO_GEOM="${VIDEO_GEOM:-0,${margin_x},${margin_top},${left_width},${usable_height}}"
  WEBOTS_GEOM="${WEBOTS_GEOM:-0,$((margin_x * 2 + left_width)),${margin_top},${right_width},${usable_height}}"
}

detect_screen_layout
start_local_vlm_server

build_shortlist() {
  cat <<EOF
$REPO_DIR/debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4
$REPO_DIR/debug_video_samples/finger_no__no_no_finger_wave__82vLYYXukIE.mp4
$REPO_DIR/debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4
$REPO_DIR/debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4
$REPO_DIR/debug_video_samples/no_shake__man_shake_head_no__yZ-351AUZqE.mp4
$REPO_DIR/debug_video_samples/yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4
EOF
}

record_one() {
  local video_path="$1"
  local name stem duration record_seconds title ffplay_log webots_log out_path

  name="$(basename "$video_path")"
  stem="${name%.mp4}"
  title="Demo-$stem"
  ffplay_log="/tmp/${stem}_ffplay.log"
  webots_log="/tmp/${stem}_webots.log"
  out_path="$OUT_DIR/${stem}.mp4"

  duration="$(python3 - <<PY
import cv2
cap = cv2.VideoCapture(r'''$video_path''')
fps = cap.get(cv2.CAP_PROP_FPS) or 0
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
cap.release()
print((frames / fps) if fps > 0 else 10.0)
PY
)"

  record_seconds="$(python3 - <<PY
dur = float(r'''$duration''')
tail = float(r'''$TAIL_SECONDS''')
startup = float(r'''$STARTUP_GRACE''')
print(max(12.0, dur + tail + startup))
PY
)"

  cleanup_run
  sleep 1

  PRINT_PIDS_ONLY=1 \
  VIDEO_TITLE="$title" \
  VIDEO_GEOM="$VIDEO_GEOM" \
  WEBOTS_GEOM="$WEBOTS_GEOM" \
  FFPLAY_LOG="$ffplay_log" \
  WEBOTS_LOG="$webots_log" \
  LOCAL_VLM_MODEL="$MODEL_ID" \
  LOCAL_VLM_NUM_CANDIDATES="$LOCAL_VLM_NUM_CANDIDATES_VALUE" \
  VLM_FRAME_COUNT="$VLM_FRAME_COUNT_VALUE" \
  VLM_MAX_TOKENS="$VLM_MAX_TOKENS_VALUE" \
  LOCAL_VLM_SERVER_URL="$LOCAL_VLM_SERVER_URL" \
  bash "$LAUNCH_SCRIPT" "$video_path" >"/tmp/${stem}_launch.log" 2>&1 &
  local launch_pid=$!

  sleep "$STARTUP_GRACE"

  DISPLAY=:0 "$CONDA_PY" "$SCREEN_RECORD_PY" \
    --x "$CAPTURE_X" \
    --y "$CAPTURE_Y" \
    --width "$CAPTURE_WIDTH" \
    --height "$CAPTURE_HEIGHT" \
    --seconds "$record_seconds" \
    --fps 25 \
    --output "$out_path" >/tmp/${stem}_record.log 2>&1

  wait "$launch_pid" || true
  cleanup_run
  sleep 2

  echo "[完成] $out_path"
}

if [[ "$USE_SHORTLIST" == "1" ]]; then
  while IFS= read -r video_path; do
    [[ -n "$video_path" ]] || continue
    [[ -f "$video_path" ]] || continue
    echo "[开始] $(basename "$video_path")"
    record_one "$video_path"
  done < <(build_shortlist)
else
  shopt -s nullglob
  for video_path in "$SAMPLES_DIR"/*.mp4; do
    echo "[开始] $(basename "$video_path")"
    record_one "$video_path"
  done
fi

echo "全部录制完成，输出目录：$OUT_DIR"
