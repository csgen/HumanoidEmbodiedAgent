#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/darian/桌面/humanoidRobot"
LAUNCH_SCRIPT="$REPO_DIR/scripts/launch_side_by_side_demo.sh"
FFMPEG_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/ffmpeg"
WMCTRL_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/wmctrl"
XWININFO_BIN="/usr/bin/xwininfo"
CONDA_PY="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/python"
SCREEN_RECORD_PY="$REPO_DIR/scripts/screen_record_region.py"

SAMPLES_DIR="${1:-$REPO_DIR/debug_video_samples}"
OUT_DIR="${2:-$REPO_DIR/artifacts/screen_recordings}"
MODEL_ID="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
CAPTURE_WIDTH="${CAPTURE_WIDTH:-1880}"
CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-720}"
CAPTURE_X="${CAPTURE_X:-0}"
CAPTURE_Y="${CAPTURE_Y:-0}"
VIDEO_GEOM="${VIDEO_GEOM:-0,0,0,820,700}"
WEBOTS_GEOM="${WEBOTS_GEOM:-0,830,0,1050,700}"
TAIL_SECONDS="${TAIL_SECONDS:-10}"
STARTUP_GRACE="${STARTUP_GRACE:-6}"

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

cleanup_run() {
  pkill -f "/home/darian/.local/opt/webots/webots" 2>/dev/null || true
  pkill -f "ffplay.*Demo Source Video" 2>/dev/null || true
  pkill -f "ffplay.*Demo-" 2>/dev/null || true
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

shopt -s nullglob
for video_path in "$SAMPLES_DIR"/*.mp4; do
  echo "[开始] $(basename "$video_path")"
  record_one "$video_path"
done

echo "全部录制完成，输出目录：$OUT_DIR"
