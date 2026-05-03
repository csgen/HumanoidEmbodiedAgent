#!/usr/bin/env python3
"""本机摄像头预览与录制脚本。

默认输出目录：/home/darian/桌面/humanoidRobot/example_video

用法：
  python3 scripts/record_webcam.py

按键：
  r  开始/停止录制
  q  退出
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


DEFAULT_OUTPUT_DIR = Path("/home/darian/桌面/humanoidRobot/example_video")
DEFAULT_LINUX_RESOLUTIONS = [
    (1280, 720),
    (640, 480),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="打开本机摄像头并录制示例视频")
    parser.add_argument("--device", type=int, default=0, help="摄像头设备编号，默认 0")
    parser.add_argument("--fps", type=float, default=20.0, help="录制帧率，默认 20")
    parser.add_argument("--width", type=int, default=1280, help="期望宽度，默认 1280")
    parser.add_argument("--height", type=int, default=720, help="期望高度，默认 720")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="启动后立刻开始录制，无需按 r",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "v4l2", "gstreamer", "any"],
        default="auto",
        help="视频后端，默认 auto；Linux 下会优先尝试 v4l2",
    )
    return parser


def create_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{output_path}")
    return writer


def _resolve_backend_names(requested: str) -> list[tuple[str, int]]:
    backends = {
        "any": cv2.CAP_ANY,
        "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
        "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
    }
    if requested != "auto":
        return [(requested, backends[requested])]
    if platform.system().lower() == "linux":
        return [
            ("v4l2", backends["v4l2"]),
            ("any", backends["any"]),
            ("gstreamer", backends["gstreamer"]),
        ]
    return [("any", backends["any"])]


def _probe_frame(cap: cv2.VideoCapture, warmup_reads: int = 5):
    frame = None
    for _ in range(warmup_reads):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
    return False, frame


def open_camera(device: int, width: int, height: int, fps: float, backend_name: str):
    errors: list[str] = []
    requested_resolutions = [(width, height)]
    if platform.system().lower() == "linux":
        for fallback in DEFAULT_LINUX_RESOLUTIONS:
            if fallback not in requested_resolutions:
                requested_resolutions.append(fallback)

    for current_backend_name, current_backend in _resolve_backend_names(backend_name):
        for current_width, current_height in requested_resolutions:
            cap = cv2.VideoCapture(device, current_backend)
            if not cap.isOpened():
                errors.append(f"backend={current_backend_name} 无法打开设备 {device}")
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_height)
            cap.set(cv2.CAP_PROP_FPS, fps)

            ok, frame = _probe_frame(cap)
            if ok:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or current_width
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or current_height
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                if actual_fps <= 1e-6:
                    actual_fps = fps
                return cap, frame, current_backend_name, actual_width, actual_height, actual_fps

            errors.append(
                f"backend={current_backend_name} 分辨率={current_width}x{current_height} 可打开但无法读取画面"
            )
            cap.release()

    detail = "；".join(errors) if errors else f"无法打开摄像头设备 {device}"
    raise RuntimeError(detail)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap, first_frame, backend_used, actual_width, actual_height, actual_fps = open_camera(
            args.device,
            args.width,
            args.height,
            args.fps,
            args.backend,
        )
    except Exception as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    writer = None
    output_path = None
    recording = False
    started_at = None

    if args.auto_start:
        filename = datetime.now().strftime("webcam_%Y%m%d_%H%M%S.mp4")
        output_path = args.output_dir / filename
        writer = create_writer(output_path, actual_fps, actual_width, actual_height)
        recording = True
        started_at = time.time()

    print("摄像头已打开。")
    print(f"后端：{backend_used}")
    print(f"分辨率：{actual_width}x{actual_height}，FPS：{actual_fps:.2f}")
    print(f"输出目录：{args.output_dir}")
    print("按 r 开始/停止录制，按 q 退出。")

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                ok = True
            else:
                ok, frame = cap.read()

            if not ok or frame is None:
                print("[错误] 读取摄像头画面失败。", file=sys.stderr)
                return 1

            status = "REC" if recording else "PREVIEW"
            text = f"{status} | r: start/stop | q: quit"
            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255) if recording else (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if recording and started_at is not None:
                elapsed = time.time() - started_at
                timer_text = f"{elapsed:.1f}s"
                cv2.putText(
                    frame,
                    timer_text,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)

            cv2.imshow("Webcam Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                if recording:
                    recording = False
                    started_at = None
                    writer.release()
                    writer = None
                    print(f"录制完成：{output_path}")
                else:
                    filename = datetime.now().strftime("webcam_%Y%m%d_%H%M%S.mp4")
                    output_path = args.output_dir / filename
                    writer = create_writer(output_path, actual_fps, actual_width, actual_height)
                    recording = True
                    started_at = time.time()
                    print(f"开始录制：{output_path}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            if output_path is not None:
                print(f"录制完成：{output_path}")
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
