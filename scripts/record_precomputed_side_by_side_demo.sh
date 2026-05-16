#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
DEFAULT_CONDA_PY="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/python"
CONDA_PY="${CONDA_PY:-python3}"
if [[ "$CONDA_PY" == "python3" && -x "$DEFAULT_CONDA_PY" ]]; then
  CONDA_PY="$DEFAULT_CONDA_PY"
fi
PRECOMPUTE_PY="$REPO_DIR/scripts/precompute_vlm_code_from_video.py"
LAUNCH_SCRIPT="$REPO_DIR/scripts/launch_side_by_side_demo.sh"
RECORD_PY="$REPO_DIR/scripts/screen_record_region.py"
PREVENT_SLEEP_SCRIPT="$REPO_DIR/scripts/prevent_display_sleep.sh"

VIDEO_PATH="${1:?Please provide video path [请传入视频路径]}"
OUT_MP4="${2:?Please provide output mp4 path [请传入输出 mp4 路径]}"
MODEL_ID="${LOCAL_VLM_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
WORK_DIR="${WORK_DIR:-$REPO_DIR/artifacts/precomputed_demo}"
VLM_FRAME_COUNT_VALUE="${VLM_FRAME_COUNT:-4}"
VLM_MAX_TOKENS_VALUE="${VLM_MAX_TOKENS:-320}"
REPLAY_CODE_OVERRIDE="${REPLAY_CODE_PATH:-}"
DEMO_WORKSPACE_INDEX="${DEMO_WORKSPACE_INDEX:-1}"
ONE_SHOT_POST_EXECUTION_SECONDS_VALUE="${ONE_SHOT_POST_EXECUTION_SECONDS:-6}"
RECORD_SECONDS="${RECORD_SECONDS:-20}"
REPLAY_START_DELAY_VALUE="${REPLAY_START_DELAY:-4}"

mkdir -p "$WORK_DIR"
mkdir -p "$(dirname "$OUT_MP4")"

STEM="$(basename "${VIDEO_PATH%.*}")"
CODE_DIR="$WORK_DIR/$STEM"
CODE_PATH="$CODE_DIR/python_code.py"
VIDEO_TITLE_VALUE="Demo-$STEM"

DISPLAY=:0 bash "$PREVENT_SLEEP_SCRIPT" >/tmp/prevent_display_sleep.log 2>&1 &
SLEEP_PID=$!

cleanup() {
  pkill -f "webots" 2>/dev/null || true
  pkill -f "ffplay.*Demo-" 2>/dev/null || true
  kill "$SLEEP_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ -n "$REPLAY_CODE_OVERRIDE" ]]; then
  CODE_PATH="$REPLAY_CODE_OVERRIDE"
else
  "$CONDA_PY" "$PRECOMPUTE_PY" "$VIDEO_PATH" \
    --output-dir "$CODE_DIR" \
    --model "$MODEL_ID" \
    --frames "$VLM_FRAME_COUNT_VALUE"
fi

if [[ ! -s "$CODE_PATH" ]]; then
  echo "[Error] [错误] Precomputed code is empty: [预计算代码为空:] $CODE_PATH"
  exit 1
fi

SCREEN_DIMS="$(DISPLAY=:0 xrandr --current 2>/dev/null | awk '/\*/ {print $1; exit}')"
WIDTH="${SCREEN_DIMS%x*}"
HEIGHT="${SCREEN_DIMS#*x}"
MARGIN_X=12
MARGIN_TOP=36
MARGIN_BOTTOM=12
USABLE_HEIGHT=$((HEIGHT - MARGIN_TOP - MARGIN_BOTTOM))
LEFT_WIDTH=$(((WIDTH - MARGIN_X * 3) / 2))
RIGHT_WIDTH=$((WIDTH - LEFT_WIDTH - MARGIN_X * 3))
VIDEO_GEOM="0,${MARGIN_X},${MARGIN_TOP},${LEFT_WIDTH},${USABLE_HEIGHT}"
WEBOTS_GEOM="0,$((MARGIN_X * 2 + LEFT_WIDTH)),${MARGIN_TOP},${RIGHT_WIDTH},${USABLE_HEIGHT}"

DEMO_MODE=replay \
REPLAY_CODE_PATH="$CODE_PATH" \
REPLAY_START_DELAY="$REPLAY_START_DELAY_VALUE" \
ONE_SHOT_POST_EXECUTION_SECONDS="$ONE_SHOT_POST_EXECUTION_SECONDS_VALUE" \
ONE_SHOT_EXIT_AFTER_EXECUTE=0 \
DEMO_WORKSPACE_INDEX="$DEMO_WORKSPACE_INDEX" \
VIDEO_TITLE="$VIDEO_TITLE_VALUE" \
VIDEO_GEOM="$VIDEO_GEOM" \
WEBOTS_GEOM="$WEBOTS_GEOM" \
RAISE_DEMO_WINDOWS=1 \
bash "$LAUNCH_SCRIPT" "$VIDEO_PATH" >/tmp/${STEM}_launch.log 2>&1 &
LAUNCH_PID=$!

sleep 2

/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/wmctrl -s "$DEMO_WORKSPACE_INDEX" >/dev/null 2>&1 || true

DISPLAY=:0 "$CONDA_PY" "$RECORD_PY" \
  --x 0 \
  --y 0 \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --seconds "$RECORD_SECONDS" \
  --fps 25 \
  --output "$OUT_MP4"

pkill -f "webots.*nao_VLM.wbt" 2>/dev/null || true
pkill -f "ffplay.*$VIDEO_TITLE_VALUE" 2>/dev/null || true
wait "$LAUNCH_PID" || true

echo "[Complete] [完成] $OUT_MP4"
