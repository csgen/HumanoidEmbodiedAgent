#!/usr/bin/env bash
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"

cleanup() {
  xset s on 2>/dev/null || true
  xset +dpms 2>/dev/null || true
}

trap cleanup EXIT INT TERM

xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

while true; do
  xset s reset 2>/dev/null || true
  sleep 20
done
