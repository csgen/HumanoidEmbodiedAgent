#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/darian/桌面/humanoidRobot"
PYTHON_BIN="/home/darian/miniconda3/envs/humanoid_robot_vlm_darian/bin/python"
RECORD_DEMO_SCRIPT="$REPO_DIR/scripts/record_precomputed_side_by_side_demo.sh"

VIDEO_PATH="${1:?请传入视频路径}"
OUT_MP4="${2:?请传入输出 mp4 路径}"

MATCHED_DIR="$($PYTHON_BIN - "$VIDEO_PATH" <<'PY'
from pathlib import Path
import cv2, numpy as np, sys

repo = Path('/home/darian/桌面/humanoidRobot')
video = Path(sys.argv[1]).resolve()
artifacts = sorted([p for p in (repo/'artifacts/oneshot').iterdir() if p.is_dir()])

def img_fp(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32))
    g = g.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return g

def read_artifact_fp(d):
    out = []
    for jpg in sorted(d.glob('frame_*.jpg'))[:6]:
        img = cv2.imread(str(jpg))
        if img is not None:
            out.append(img_fp(img))
    return out

def sample_video_fp(path, n=6):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    idxs = np.linspace(0, max(0, total - 1), n).astype(int)
    out = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append(img_fp(frame))
    cap.release()
    return out

target = sample_video_fp(video, 6)
best = None
for d in artifacts:
    if not (d / 'python_code.py').exists():
        continue
    art = read_artifact_fp(d)
    if not art or not target:
        continue
    pairs = min(len(art), len(target))
    score = sum(float(np.mean((art[i]-target[i])**2)) for i in range(pairs)) / pairs
    if best is None or score < best[0]:
        best = (score, d)

if best is None:
    raise SystemExit(1)
print(best[1])
PY
)"

if [[ -z "$MATCHED_DIR" || ! -f "$MATCHED_DIR/python_code.py" ]]; then
  echo "[错误] 未找到匹配的历史 VLM 产物。"
  exit 1
fi

WORK_DIR="$REPO_DIR/artifacts/precomputed_demo/$(basename "${VIDEO_PATH%.*}")_matched"
mkdir -p "$WORK_DIR"
cp "$MATCHED_DIR/python_code.py" "$WORK_DIR/python_code.py"
cp "$MATCHED_DIR/summary.txt" "$WORK_DIR/matched_summary.txt" 2>/dev/null || true
cp "$MATCHED_DIR/semantic_context.json" "$WORK_DIR/matched_semantic_context.json" 2>/dev/null || true

DEMO_MODE=replay \
REPLAY_CODE_PATH="$WORK_DIR/python_code.py" \
"$RECORD_DEMO_SCRIPT" "$VIDEO_PATH" "$OUT_MP4"
