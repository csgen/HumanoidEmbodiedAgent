#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/darian/桌面/humanoidRobot"
WEBOTS_BIN="/home/darian/.local/opt/webots/webots"
WORLD_FILE="$REPO_DIR/nao_VLM/worlds/nao_VLM.wbt"

cd "$REPO_DIR"
cp .env.debug.example-video .env

export DISPLAY=:0
export XDG_SESSION_TYPE=x11

exec "$WEBOTS_BIN" --mode=realtime "$WORLD_FILE"
