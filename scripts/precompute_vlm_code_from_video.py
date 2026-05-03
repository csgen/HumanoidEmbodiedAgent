#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path('/home/darian/桌面/humanoidRobot')
CTRL_DIR = REPO_DIR / 'nao_VLM' / 'controllers' / 'nao_vlm_test'
sys.path.insert(0, str(CTRL_DIR))

import config
import vlm_client
from offline_local_vlm_debug import sample_video_frames, build_executor


def main() -> int:
    parser = argparse.ArgumentParser(description='对输入视频运行 VLM，并导出低层控制代码。')
    parser.add_argument('video', help='输入 mp4 路径')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--model', default=os.environ.get('LOCAL_VLM_MODEL', config.LOCAL_VLM_MODEL))
    parser.add_argument('--frames', type=int, default=int(os.environ.get('VLM_FRAME_COUNT', config.VLM_FRAME_COUNT)))
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_absolute():
        video_path = (REPO_DIR / video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f'视频不存在: {video_path}')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_b64 = sample_video_frames(video_path, args.frames)
    client = vlm_client.VLMClient(joint_limits=build_executor()._joint_limits, model=args.model)
    rsp = client.call(frames_b64)

    (out_dir / 'summary.json').write_text(
        json.dumps(
            {
                'video': str(video_path),
                'model': args.model,
                'frames': len(frames_b64),
                'ok': rsp.ok,
                'error': rsp.error,
                'elapsed_seconds': rsp.elapsed_seconds,
                'semantic_context': rsp.semantic_context,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    (out_dir / 'python_code.py').write_text(rsp.python_code or '', encoding='utf-8')
    (out_dir / 'raw_response.txt').write_text(rsp.raw_text or '', encoding='utf-8')

    print(json.dumps({'ok': rsp.ok, 'error': rsp.error, 'elapsed_seconds': rsp.elapsed_seconds, 'output_dir': str(out_dir)}, ensure_ascii=False))
    return 0 if rsp.ok and rsp.python_code.strip() else 1


if __name__ == '__main__':
    raise SystemExit(main())
