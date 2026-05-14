"""
Phase-5 metrics recorder for Webots controller runs.

The recorder is intentionally passive: nothing is created unless
METRICS_RUN_ID is set. Controller code can call it freely without changing
normal demo behavior.

result.json schema (the dict passed to write_result()): see
evaluation/RESULT_SCHEMA.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import pinocchio as pin
except Exception:
    pin = None

import config


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


class MetricsRecorder:
    def __init__(self, run_id: str, output_dir: Optional[Path] = None) -> None:
        self.run_id = str(run_id).strip()
        if not self.run_id:
            raise ValueError('run_id must be non-empty')
        if output_dir is None:
            if config.METRICS_OUTPUT_DIR:
                output_dir = Path(config.METRICS_OUTPUT_DIR).expanduser()
            else:
                output_dir = config.ARTIFACTS_DIR / self.run_id
        self.run_dir = Path(output_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.joint_log_path = self.run_dir / 'joint_states.jsonl'
        self.sandbox_log_path = self.run_dir / 'sandbox_events.jsonl'
        self.result_path = self.run_dir / 'result.json'
        self.screenshot_path = self.run_dir / 'robot_response.png'
        self._step_records = 0
        self._sandbox_records = 0
        # Robot motion capture: a throttled sequence of screenshots taken
        # during the VLM-code execution window (armed via begin/end).
        self._motion_capture_armed = False
        self._motion_frame_paths: list = []
        self._last_motion_capture_sim_time: Optional[float] = None

    @classmethod
    def from_env(cls) -> Optional['MetricsRecorder']:
        if not config.METRICS_RUN_ID:
            return None
        return cls(config.METRICS_RUN_ID)

    def _append_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, default=_json_default, sort_keys=True) + '\n')

    def record_step(self, *, sim_time: float, q_current, sensors: Dict[str, Any], model, data) -> None:
        joints = {}
        for name, sensor in (sensors or {}).items():
            try:
                joints[name] = float(sensor.getValue())
            except Exception:
                continue

        com_xyz = None
        if pin is not None:
            try:
                pin.centerOfMass(model, data, q_current)
                com_xyz = np.asarray(data.com[0], dtype=float).reshape(3).tolist()
            except Exception:
                com_xyz = None

        self._step_records += 1
        self._append_jsonl(
            self.joint_log_path,
            {
                'wall_time': time.time(),
                'sim_time': float(sim_time),
                'step_index': self._step_records,
                'joints': joints,
                'q': np.asarray(q_current, dtype=float).tolist(),
                'com_xyz': com_xyz,
            },
        )

    def record_sandbox_event(
        self,
        event: str,
        *,
        error: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
        code: str = '',
    ) -> None:
        self._sandbox_records += 1
        code_hash = ''
        if code:
            code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]
        self._append_jsonl(
            self.sandbox_log_path,
            {
                'wall_time': time.time(),
                'event_index': self._sandbox_records,
                'event': event,
                'error': error,
                'elapsed_seconds': elapsed_seconds,
                'code_hash': code_hash,
            },
        )

    def export_screenshot(self, robot) -> Optional[Path]:
        try:
            robot.exportImage(str(self.screenshot_path), 90)
        except Exception as exc:
            (self.run_dir / 'robot_response.error.txt').write_text(str(exc), encoding='utf-8')
            return None
        return self.screenshot_path

    # ------------------------------------------------------------------ motion capture
    # A throttled sequence of robot screenshots taken while the VLM-generated
    # code executes, so a continuous motion is reviewable / judgeable as a
    # sequence rather than a single (possibly unrepresentative) frame.

    def begin_motion_capture(self) -> None:
        """Arm motion capture. Call right before executing VLM code."""
        self._motion_capture_armed = True
        self._motion_frame_paths = []
        self._last_motion_capture_sim_time = None

    def end_motion_capture(self) -> list:
        """Disarm motion capture and return the captured frame paths (as str)."""
        self._motion_capture_armed = False
        return list(self._motion_frame_paths)

    @property
    def motion_frame_paths(self) -> list:
        return list(self._motion_frame_paths)

    def maybe_capture_motion_frame(self, robot, sim_time: float) -> Optional[Path]:
        """
        Throttled per-step screenshot. No-op unless armed. Captures at most
        one frame per MOTION_FRAME_INTERVAL_S of sim time, up to
        MOTION_FRAME_MAX frames total. Safe to call every simulation step.
        """
        if not self._motion_capture_armed:
            return None
        if len(self._motion_frame_paths) >= config.MOTION_FRAME_MAX:
            return None
        last = self._last_motion_capture_sim_time
        if last is not None and (float(sim_time) - last) < config.MOTION_FRAME_INTERVAL_S:
            return None

        index = len(self._motion_frame_paths) + 1
        frame_path = self.run_dir / f'robot_frame_{index:02d}.jpg'
        try:
            robot.exportImage(str(frame_path), 90)
        except Exception as exc:
            (self.run_dir / f'robot_frame_{index:02d}.error.txt').write_text(
                str(exc), encoding='utf-8')
            return None
        self._motion_frame_paths.append(str(frame_path))
        self._last_motion_capture_sim_time = float(sim_time)
        return frame_path

    def write_result(self, payload: Dict[str, Any]) -> Path:
        payload = dict(payload or {})
        payload.setdefault('run_id', self.run_id)
        payload.setdefault('scenario_id', config.EVAL_SCENARIO_ID)
        payload.setdefault('method', config.EVAL_METHOD)
        payload.setdefault('artifacts', {})
        payload['artifacts'].update({
            'run_dir': str(self.run_dir),
            'joint_states': str(self.joint_log_path),
            'sandbox_events': str(self.sandbox_log_path),
            'robot_screenshot': str(self.screenshot_path) if self.screenshot_path.exists() else '',
            'result_json': str(self.result_path),
        })
        self.result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + os.linesep,
            encoding='utf-8',
        )
        return self.result_path

    def paths(self) -> Dict[str, str]:
        return {
            'run_dir': str(self.run_dir),
            'joint_states': str(self.joint_log_path),
            'sandbox_events': str(self.sandbox_log_path),
            'robot_screenshot': str(self.screenshot_path) if self.screenshot_path.exists() else '',
            'result_json': str(self.result_path),
        }
