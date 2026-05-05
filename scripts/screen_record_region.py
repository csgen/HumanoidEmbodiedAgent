#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mss
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='录制桌面区域为 mp4')
    parser.add_argument('--x', type=int, required=True)
    parser.add_argument('--y', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--fps', type=float, default=25.0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--black-mean-threshold', type=float, default=4.0)
    parser.add_argument('--black-std-threshold', type=float, default=3.0)
    parser.add_argument('--max-consecutive-black-frames', type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        float(args.fps),
        (int(args.width), int(args.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f'无法写入视频: {args.output}')

    region = {
        'left': int(args.x),
        'top': int(args.y),
        'width': int(args.width),
        'height': int(args.height),
    }
    frame_interval = 1.0 / max(1.0, float(args.fps))
    deadline = time.time() + max(0.1, float(args.seconds))
    consecutive_black_frames = 0

    try:
        with mss.mss() as sct:
            while time.time() < deadline:
                loop_start = time.time()
                shot = np.array(sct.grab(region), copy=False)
                frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_luma = float(gray.mean())
                std_luma = float(gray.std())

                if (
                    mean_luma <= float(args.black_mean_threshold)
                    and std_luma <= float(args.black_std_threshold)
                ):
                    consecutive_black_frames += 1
                else:
                    consecutive_black_frames = 0

                if consecutive_black_frames >= int(args.max_consecutive_black_frames):
                    raise RuntimeError(
                        '检测到连续黑屏帧，桌面可能已休眠、锁屏或显示输出关闭，停止录制。'
                    )

                writer.write(frame)
                sleep_for = frame_interval - (time.time() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except Exception:
        writer.release()
        try:
            args.output.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    writer.release()


if __name__ == '__main__':
    main()
