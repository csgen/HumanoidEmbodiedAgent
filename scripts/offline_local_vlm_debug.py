#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import cv2


REPO_DIR = Path('/home/darian/桌面/humanoidRobot')
CTRL_DIR = REPO_DIR / 'nao_VLM' / 'controllers' / 'nao_vlm_test'
sys.path.insert(0, str(CTRL_DIR))

import config
import vlm_client
from sandbox_exec import SandboxExecutor


def sample_video_frames(path: Path, n: int) -> list[str]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f'无法打开视频: {path}')

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    else:
        indices = [round(i * max(0, total - 1) / max(1, n - 1)) for i in range(n)]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
    cap.release()

    out = []
    for frame in frames[:n]:
        ok, encoded = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        out.append(base64.b64encode(encoded.tobytes()).decode('utf-8'))
    return out


def build_executor() -> SandboxExecutor:
    executor = SandboxExecutor()
    executor.register_many({
        'move_joint': lambda *a, **k: None,
        'move_joints': lambda *a, **k: None,
        'move_arm_ik': lambda *a, **k: None,
        'set_hand': lambda *a, **k: None,
        'oscillate_joint': lambda *a, **k: None,
        'hold': lambda *a, **k: None,
        'idle': lambda *a, **k: None,
        'speak': lambda *a, **k: None,
        'navigate_to': lambda *a, **k: None,
    })
    executor.set_joint_limits({
        'HeadYaw': (-2.0857, 2.0857),
        'HeadPitch': (-0.6720, 0.5149),
        'LShoulderPitch': (-2.0857, 2.0857),
        'LShoulderRoll': (-0.3142, 1.3265),
        'LElbowYaw': (-2.0857, 2.0857),
        'LElbowRoll': (-1.5446, -0.0349),
        'LWristYaw': (-1.8238, 1.8238),
        'RShoulderPitch': (-2.0857, 2.0857),
        'RShoulderRoll': (-1.3265, 0.3142),
        'RElbowYaw': (-2.0857, 2.0857),
        'RElbowRoll': (0.0349, 1.5446),
        'RWristYaw': (-1.8238, 1.8238),
    })
    return executor


def build_runtime_signature_executor() -> SandboxExecutor:
    executor = SandboxExecutor()

    def move_joint(name: str, angle: float, duration: float, trajectory: str = 'cubic'):
        return None

    def move_joints(joint_angles: dict, duration: float, trajectory: str = 'cubic'):
        return None

    def move_arm_ik(side: str, xyz, duration: float, orientation=None):
        return None

    def move_head(yaw: float, pitch: float, duration: float = 0.2, trajectory: str = 'min_jerk'):
        return None

    def set_hand(side: str, openness: float, duration: float, trajectory: str = 'cubic'):
        return None

    def oscillate_joint(name: str, center: float, amplitude: float, frequency: float, duration: float, decay: float = 0.0):
        return None

    def hold(duration: float):
        return None

    def idle(duration: float):
        return None

    executor.register_many({
        'move_joint': move_joint,
        'move_joints': move_joints,
        'move_arm_ik': move_arm_ik,
        'move_head': move_head,
        'set_hand': set_hand,
        'oscillate_joint': oscillate_joint,
        'hold': hold,
        'idle': idle,
    })
    executor.set_joint_limits(build_executor()._joint_limits)
    return executor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='+', help='一个或多个 mp4 路径')
    parser.add_argument('--model', default=os.environ.get('LOCAL_VLM_MODEL', config.LOCAL_VLM_MODEL))
    parser.add_argument('--frames', type=int, default=int(os.environ.get('VLM_FRAME_COUNT', config.VLM_FRAME_COUNT)))
    args = parser.parse_args()

    client = vlm_client.VLMClient(joint_limits=build_executor()._joint_limits, model=args.model)
    validator = build_executor()
    runtime_executor = build_runtime_signature_executor()

    for video in args.videos:
        path = Path(video)
        if not path.is_absolute():
            path = (REPO_DIR / path).resolve()
        print(f'\n===== {path.name} =====')
        frames_b64 = sample_video_frames(path, args.frames)
        print(f'采样帧数: {len(frames_b64)}')
        summary = vlm_client._infer_visual_motion_summary(frames_b64)
        print('motion_summary=')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        rsp = client.call(frames_b64)
        print(f'ok={rsp.ok} error={rsp.error} elapsed={rsp.elapsed_seconds:.2f}s')
        print('semantic_context=')
        print(json.dumps(rsp.semantic_context, ensure_ascii=False, indent=2))
        print('python_code=')
        print(rsp.python_code)
        validation = validator.validate(rsp.python_code)
        print(f'validator_ok={validation.ok} validator_error={validation.error}')
        runtime_result = runtime_executor.run(rsp.python_code) if rsp.python_code else None
        if runtime_result is not None:
            print(f'runtime_ok={runtime_result.ok} runtime_error={runtime_result.error}')
        if rsp.python_code:
            penalty = vlm_client._naturalness_penalty(summary, rsp.python_code)
            generic = vlm_client._looks_too_generic_for_clip(summary, rsp.semantic_context, rsp.python_code)
            print(f'naturalness_penalty={penalty:.2f} generic_like={generic}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
