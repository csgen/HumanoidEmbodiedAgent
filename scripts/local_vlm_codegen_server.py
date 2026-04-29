#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request


REPO_DIR = Path('/home/darian/桌面/humanoidRobot')
CTRL_DIR = REPO_DIR / 'nao_VLM' / 'controllers' / 'nao_vlm_test'
sys.path.insert(0, str(CTRL_DIR))

os.environ.setdefault('VLM_BACKEND', 'local')

import vlm_client


app = Flask(__name__)
_CLIENT = None


def get_client(model: str | None = None):
    global _CLIENT
    target_model = model or os.environ.get('LOCAL_VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')
    if _CLIENT is None or _CLIENT.model != target_model:
        _CLIENT = vlm_client.VLMClient(joint_limits={}, model=target_model)
    return _CLIENT


@app.get('/health')
def health():
    model = os.environ.get('LOCAL_VLM_MODEL', 'Qwen/Qwen2.5-VL-3B-Instruct')
    return jsonify({'ok': True, 'model': model})


@app.post('/generate_from_frames')
def generate_from_frames():
    payload = request.get_json(force=True, silent=False) or {}
    frames_b64 = payload.get('frames_b64') or []
    model = payload.get('model')
    client = get_client(model)
    rsp = client._call_local(frames_b64)
    return jsonify({
        'semantic_context': rsp.semantic_context,
        'python_code': rsp.python_code,
        'raw_text': rsp.raw_text,
        'elapsed_seconds': rsp.elapsed_seconds,
        'ok': rsp.ok,
        'error': rsp.error,
    })


def main():
    host = os.environ.get('LOCAL_VLM_SERVER_HOST', '127.0.0.1')
    port = int(os.environ.get('LOCAL_VLM_SERVER_PORT', '8765'))
    get_client(None)
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
