"""
NAO VLM Webots controller — Phase 0 integration.

Wires together:
  - Pinocchio IK / dynamics (existing)
  - FrameBuffer background capture (new, see frame_buffer.py)
  - VLMClient multi-image GPT-4o call (new, see vlm_client.py)
  - SandboxExecutor for VLM-generated code (new, see sandbox_exec.py)
  - NaoVlmAPI with both legacy methods AND Phase-0 motion primitives

Flow per Webots step:
  1. sync Webots sensors into Pinocchio q
  2. if no VLM call in flight AND interval elapsed, kick a new VLM call
     in a daemon thread (frames sampled from FrameBuffer)
  3. drain result queue; if a VLMResponse arrived, run its .python_code
     through the sandbox (which internally calls blocking primitives that
     themselves call robot.step() - physics advances inside those loops)

Trajectory shapes available via the `trajectory=` parameter on primitives:
  'linear', 'cubic' (smoothstep, default), 'cosine', 'min_jerk' (quintic).
See `_ease()` for definitions.
"""
from __future__ import annotations

import base64
import difflib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Load .env before reading config
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Try repo root first, then fall back to cwd (Webots controller cwd is this dir)
    _here = Path(__file__).resolve().parent
    for env_candidate in (_here.parent.parent.parent / '.env', _here / '.env'):
        if env_candidate.exists():
            load_dotenv(env_candidate)
            print(f'[init] loaded env from {env_candidate}')
            break
except ImportError:
    pass  # python-dotenv is optional; OS env vars still work

# ---------------------------------------------------------------------------
# Core deps
# ---------------------------------------------------------------------------
import cv2
import numpy as np
import pinocchio as pin
from controller import Supervisor

import config
from frame_buffer import FrameBuffer
from idle_animator import IdleAnimator
from fallback import FallbackPolicy
from sandbox_exec import SandboxExecutor
from vlm_trigger import VLMTrigger

# VLMClient import is guarded; controller still runs without OpenAI SDK
try:
    from vlm_client import VLMClient
    _VLM_IMPORT_OK = True
except Exception as _e:
    print(f'[init] VLMClient unavailable: {_e}')
    VLMClient = None  # type: ignore
    _VLM_IMPORT_OK = False


# ===========================================================================
# Easing helpers
# ===========================================================================

def _ease(t: float, style: str = 'cubic') -> float:
    """Return eased progress in [0,1]. Phase-0 basic implementations."""
    t = max(0.0, min(1.0, t))
    if style == 'linear':
        return t
    if style == 'cubic':
        # smoothstep (Hermite); Phase 1 may swap to true cubic spline w/ v0,v1
        return t * t * (3.0 - 2.0 * t)
    if style == 'cosine':
        return 0.5 * (1.0 - math.cos(math.pi * t))
    if style == 'min_jerk':
        # 5th-order polynomial with zero v,a at both ends (classic min-jerk)
        return t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t)
    return t


# ===========================================================================
# NaoVlmAPI - legacy methods + Phase-0 motion primitives
# ===========================================================================

class NaoVlmAPI:
    def __init__(
        self,
        robot: Supervisor,
        pin_model,
        pin_data,
        motors: Dict[str, object],
        sensors: Dict[str, object],
        cameras: Dict[str, object],
        timestep: int,
        joint_q_idx_map: Dict[str, int],
        finger_motors: Dict[str, List[object]],
    ):
        self.robot = robot
        self.model = pin_model
        self.data = pin_data
        self.motors = motors
        self.sensors = sensors
        self.cameras = cameras
        self.timestep = timestep
        self.joint_q_idx_map = joint_q_idx_map
        self.finger_motors = finger_motors

        self.hand_frames = {
            'left': (self.model.getFrameId('l_wrist') if self.model.existFrame('l_wrist')
                     else self.model.getFrameId('LHand')),
            'right': (self.model.getFrameId('r_wrist') if self.model.existFrame('r_wrist')
                      else self.model.getFrameId('RHand')),
        }
        self.q_current = pin.neutral(self.model)
        self.current_posture = 'stand'
        self._aborted = False   # set to True when robot.step returns -1

    # ------------------------------------------------------------------ internals

    def _sync_sensors(self) -> None:
        """Mirror Webots joint sensors -> Pinocchio q."""
        for name, sensor in self.sensors.items():
            idx = self.joint_q_idx_map.get(name)
            if idx is not None:
                self.q_current[idx] = sensor.getValue()

    def _step(self) -> bool:
        """Advance simulation by one timestep, sync sensors. Returns False if sim ended."""
        if self.robot.step(self.timestep) == -1:
            self._aborted = True
            return False
        self._sync_sensors()
        return True

    # Tiny margin keeps us off the exact boundary so Webots's strict `>` float
    # comparison doesn't spam "too big requested position: X > X" warnings.
    _LIMIT_MARGIN_RAD = 1e-4   # ~0.006 deg, negligible motion-wise

    def _clip_to_motor_limits(self, motor, angle: float) -> float:
        mn = motor.getMinPosition()
        mx = motor.getMaxPosition()
        if mn != mx:
            return float(np.clip(angle,
                                 mn + self._LIMIT_MARGIN_RAD,
                                 mx - self._LIMIT_MARGIN_RAD))
        return float(angle)

    def _canonicalize_joint_name(self, name: str) -> Optional[str]:
        if name in self.motors:
            return name
        if not isinstance(name, str):
            return None

        normalized = ''.join(ch for ch in name if ch.isalnum()).lower()
        exact_normalized = {}
        for joint_name in self.motors:
            key = ''.join(ch for ch in joint_name if ch.isalnum()).lower()
            exact_normalized[key] = joint_name

        if normalized in exact_normalized:
            return exact_normalized[normalized]

        candidates = difflib.get_close_matches(
            normalized,
            list(exact_normalized.keys()),
            n=1,
            cutoff=0.72,
        )
        if not candidates:
            return None
        resolved = exact_normalized[candidates[0]]
        print(f'[NaoVlmAPI] joint alias resolved: {name!r} -> {resolved!r}')
        return resolved

    # ------------------------------------------------------------------ introspection

    def get_joint_limits(self) -> Dict[str, tuple]:
        """Return {joint_name: (min_rad, max_rad)} for non-fixed joints."""
        out = {}
        for name, motor in self.motors.items():
            mn, mx = motor.getMinPosition(), motor.getMaxPosition()
            if mn != mx:
                out[name] = (float(mn), float(mx))
        return out

    def get_robot_state(self) -> Dict[str, float]:
        """Return a small dict of joint positions + posture."""
        state = {
            'posture': self.current_posture,
            'head_pitch': (self.motors['HeadPitch'].getTargetPosition()
                           if 'HeadPitch' in self.motors else 0.0),
            'head_yaw': (self.motors['HeadYaw'].getTargetPosition()
                         if 'HeadYaw' in self.motors else 0.0),
        }
        return state

    def _side_arm_joint_names(self, side: str) -> List[str]:
        prefix = 'L' if side == 'left' else 'R'
        return [
            f'{prefix}ShoulderPitch',
            f'{prefix}ShoulderRoll',
            f'{prefix}ElbowYaw',
            f'{prefix}ElbowRoll',
            f'{prefix}WristYaw',
        ]

    def _joint_velocity_indices(self, joint_name: str) -> List[int]:
        if joint_name not in self.model.names:
            return []
        jid = self.model.getJointId(joint_name)
        joint = self.model.joints[jid]
        return list(range(joint.idx_v, joint.idx_v + joint.nv))

    def capture_camera_image(self, save_path: str = 'vlm_view.jpg') -> str:
        if 'CameraTop' not in self.cameras:
            return 'ERROR: Top camera not found.'
        cam = self.cameras['CameraTop']
        image_data = cam.getImage()
        if not image_data:
            return 'ERROR: Failed to capture image.'
        w, h = cam.getWidth(), cam.getHeight()
        arr = np.frombuffer(image_data, np.uint8).reshape((h, w, 4))
        cv2.imwrite(save_path, arr[:, :, :3])
        return f'SUCCESS: Image saved to {save_path}'

    def speak(self, text: str) -> str:
        print(f'\n[NAO speak] {text}')
        return 'OK speak'

    # ------------------------------------------------------------------ Phase-0 motion primitives
    #   These are the API the VLM prompt describes. Phase 1 will refine the
    #   trajectory math (cubic splines, proper min-jerk) without API changes.

    def move_joint(self, name: str, angle: float, duration: float,
                   trajectory: str = 'cubic') -> str:
        resolved_name = self._canonicalize_joint_name(name)
        motor = self.motors.get(resolved_name) if resolved_name else None
        if motor is None:
            return f'ERROR: unknown joint {name!r}'
        target = self._clip_to_motor_limits(motor, float(angle))
        start = motor.getTargetPosition()
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        for i in range(1, steps + 1):
            s = _ease(i / steps, trajectory)
            motor.setPosition(start + (target - start) * s)
            if not self._step():
                return 'ABORTED'
        return f'OK move_joint {resolved_name}->{target:.3f}'

    def move_joints(self, joint_angles: Dict[str, float], duration: float,
                    trajectory: str = 'cubic') -> str:
        segments = {}
        for name, angle in joint_angles.items():
            resolved_name = self._canonicalize_joint_name(name)
            motor = self.motors.get(resolved_name) if resolved_name else None
            if motor is None:
                continue
            start = motor.getTargetPosition()
            target = self._clip_to_motor_limits(motor, float(angle))
            segments[resolved_name] = (motor, start, target)
        if not segments:
            return 'ERROR: no known joints in move_joints'
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        for i in range(1, steps + 1):
            s = _ease(i / steps, trajectory)
            for name, (motor, start, target) in segments.items():
                motor.setPosition(start + (target - start) * s)
            if not self._step():
                return 'ABORTED'
        return f'OK move_joints n={len(segments)}'

    def move_arm_ik(self, side: str, xyz, duration: float,
                    orientation=None) -> str:
        if side not in ('left', 'right'):
            return f'ERROR: invalid side {side!r}'
        try:
            target_pos = np.asarray(xyz, dtype=float).reshape(3)
        except Exception:
            return f'ERROR: xyz must be a 3-vector, got {xyz!r}'

        frame_id = self.hand_frames[side]
        q = self.q_current.copy()
        allowed_joint_names = [
            name for name in self._side_arm_joint_names(side)
            if name in self.motors and name in self.model.names
        ]
        if not allowed_joint_names:
            return f'ERROR: no controllable arm joints for side={side!r}'
        allowed_velocity_indices = []
        for joint_name in allowed_joint_names:
            allowed_velocity_indices.extend(self._joint_velocity_indices(joint_name))

        # IK: only allow the selected arm chain to move.
        reached = False
        for _ in range(30):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            cur = self.data.oMf[frame_id].translation
            err = target_pos - cur
            if np.linalg.norm(err) < 0.005:
                reached = True
                break
            J = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED
            )[:3, :]
            enabled = np.zeros(J.shape[1], dtype=bool)
            for idx in allowed_velocity_indices:
                if 0 <= idx < J.shape[1]:
                    enabled[idx] = True
            J[:, ~enabled] = 0.0
            damping = 1e-4
            JJt = J @ J.T + damping * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            q = pin.integrate(self.model, q, dq * 0.5)

        # Extract target angles only for the selected arm so IK never drags
        # unrelated joints into the motion.
        target_angles = {}
        for motor_name in allowed_joint_names:
            if motor_name in self.model.names and motor_name in self.motors:
                jid = self.model.getJointId(motor_name)
                qi = self.model.joints[jid].idx_q
                target_angles[motor_name] = float(q[qi])
        if not target_angles:
            return 'ERROR: no motors match IK solution'

        result = self.move_joints(target_angles, duration, trajectory='cubic')
        suffix = 'converged' if reached else 'not-converged'
        return f'{result} ({suffix})'

    def oscillate_joint(self, name: str, center: float, amplitude: float,
                        frequency: float, duration: float,
                        decay: float = 0.0) -> str:
        resolved_name = self._canonicalize_joint_name(name)
        motor = self.motors.get(resolved_name) if resolved_name else None
        if motor is None:
            return f'ERROR: unknown joint {name!r}'
        c = float(center)
        a0 = float(amplitude)
        f = float(frequency)
        d = max(0.0, float(decay))
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        for i in range(1, steps + 1):
            t = (i / steps) * dur
            # Amplitude decay: envelope factor in (0,1] over the duration
            env = math.exp(-d * t / dur) if d > 0 else 1.0
            angle = c + a0 * env * math.sin(2.0 * math.pi * f * t)
            motor.setPosition(self._clip_to_motor_limits(motor, angle))
            if not self._step():
                return 'ABORTED'
        return f'OK oscillate_joint {resolved_name}'

    def set_hand(self, side: str, openness: float, duration: float,
                 trajectory: str = 'cubic') -> str:
        if side not in ('left', 'right'):
            return f'ERROR: invalid side {side!r}'
        prefix = 'L' if side == 'left' else 'R'
        motors = list(self.finger_motors.get(prefix, []))
        if not motors:
            return f'ERROR: no finger motors for side={side!r}'
        target = float(np.clip(openness, 0.0, 1.0))
        starts = [motor.getTargetPosition() for motor in motors]
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        for i in range(1, steps + 1):
            s = _ease(i / steps, trajectory)
            for motor, start in zip(motors, starts):
                motor.setPosition(start + (target - start) * s)
            if not self._step():
                return 'ABORTED'
        return f'OK set_hand {side} openness={target:.2f}'

    def hold(self, duration: float) -> str:
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        for _ in range(steps):
            if not self._step():
                return 'ABORTED'
        return f'OK hold {duration:.2f}s'

    def idle(self, duration: float) -> str:
        """Subtle breathing motion + small head scan. Phase-0 basic version."""
        dur = max(self.timestep / 1000.0, float(duration))
        steps = max(1, int(round(dur * 1000.0 / self.timestep)))
        hy = self.motors.get('HeadYaw')
        lsp = self.motors.get('LShoulderPitch')
        rsp = self.motors.get('RShoulderPitch')
        for i in range(1, steps + 1):
            t = (i / steps) * dur
            if hy is not None:
                hy.setPosition(self._clip_to_motor_limits(hy, 0.08 * math.sin(0.3 * t)))
            # Anti-phase subtle breathing on shoulders
            if lsp is not None:
                lsp.setPosition(self._clip_to_motor_limits(
                    lsp, 1.5 + 0.03 * math.sin(0.5 * t)))
            if rsp is not None:
                rsp.setPosition(self._clip_to_motor_limits(
                    rsp, 1.5 + 0.03 * math.sin(0.5 * t + math.pi)))
            if not self._step():
                return 'ABORTED'
        return f'OK idle {duration:.2f}s'

    # ------------------------------------------------------------------ legacy API (backward compat)
    #   Kept verbatim in behavior so the existing automated test sequence still works.

    def look_at(self, yaw_angle: float, pitch_angle: float) -> str:
        yaw = float(np.clip(yaw_angle, -2.0, 2.0))
        pitch = float(np.clip(pitch_angle, -0.6, 0.5))
        if 'HeadYaw' in self.motors:
            self.motors['HeadYaw'].setPosition(yaw)
        if 'HeadPitch' in self.motors:
            self.motors['HeadPitch'].setPosition(pitch)
        return f'SUCCESS: Look at yaw={yaw:.2f}, pitch={pitch:.2f}'

    def move_arm(self, side, x, y, z):
        """Legacy non-blocking IK. Retained for backward compat; prefer move_arm_ik."""
        return self.move_arm_ik(side, [x, y, z], duration=0.6)

    def operate_gripper(self, side: str, action: str) -> str:
        prefix = 'L' if side == 'left' else 'R'
        target_angle = 0.0 if action == 'close' else 1.0
        count = 0
        for motor in self.finger_motors.get(prefix, []):
            motor.setPosition(target_angle)
            count += 1
        if count == 0:
            return 'ERROR: finger motors not found'
        return f'SUCCESS: {side} hand {action}ed ({count} phalanges)'

    def set_posture(self, posture_name: str) -> str:
        self.current_posture = posture_name
        val_knee = 1.0 if posture_name == 'squat' else 0.0
        val_hip_ankle = -0.5 if posture_name == 'squat' else 0.0
        for leg in ('L', 'R'):
            for suffix, v in (('KneePitch', val_knee),
                              ('HipPitch', val_hip_ankle),
                              ('AnklePitch', val_hip_ankle)):
                name = f'{leg}{suffix}'
                if name in self.motors:
                    self.motors[name].setPosition(v)
        return f'SUCCESS: posture={posture_name}'

    def navigate_to(self, delta_x: float, delta_y: float, delta_theta: float) -> str:
        """Supervisor-mode macroscopic translation (biped workaround)."""
        node = self.robot.getSelf()
        if node is None:
            return 'ERROR: supervisor mode required'
        trans_field = node.getField('translation')
        current = trans_field.getSFVec3f()
        frames = 50
        for i in range(1, frames + 1):
            interp_x = current[0] + (delta_x * i / frames)
            interp_y = current[1] + (delta_y * i / frames)
            trans_field.setSFVec3f([interp_x, interp_y, current[2]])
            if not self._step():
                return 'ABORTED'
        return f'SUCCESS: navigated ({delta_x:+.2f}, {delta_y:+.2f})'


# ===========================================================================
# VLMWorker - off-thread VLM call wrapper
# ===========================================================================

class VLMWorker:
    """Launches a VLM call on a daemon thread; main thread polls result_queue."""

    def __init__(self, client, buffer: FrameBuffer):
        self.client = client
        self.buffer = buffer
        self.result_queue: queue.Queue = queue.Queue()
        self._in_flight = threading.Event()
        self.total_calls = 0

    @property
    def in_flight(self) -> bool:
        return self._in_flight.is_set()

    def kick(self) -> bool:
        """Try to start a new VLM call. Returns False if one is already in flight
        or the buffer hasn't filled yet."""
        if self._in_flight.is_set():
            return False
        frames = self.buffer.sample_recent(config.VLM_FRAME_COUNT)
        if not frames:
            return False
        self._in_flight.set()
        self.total_calls += 1
        t = threading.Thread(
            target=self._run, args=(frames,), name='VLMWorker', daemon=True
        )
        t.start()
        return True

    def _run(self, frames: List[str]) -> None:
        try:
            rsp = self.client.call(frames)
            self.result_queue.put(rsp)
        except Exception as e:
            # Should not happen: VLMClient.call catches its own errors.
            print(f'[VLMWorker] unexpected: {e}')
        finally:
            self._in_flight.clear()

    def poll(self):
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None


# ===========================================================================
# Main controller
# ===========================================================================

def _build_pinocchio():
    urdf_path = config.find_urdf_path()
    if urdf_path is None:
        print('[ERROR] URDF not found. Candidates checked:')
        for c in config.URDF_CANDIDATES:
            print(f'          {c}')
        return None, None, None
    print(f'[init] URDF: {urdf_path}')
    try:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    except Exception as e:
        print(f'[ERROR] Pinocchio failed to load URDF: {e}')
        return None, None, None
    data = model.createData()
    return urdf_path, model, data


def _bind_devices(robot: Supervisor, model, timestep: int):
    cameras: Dict[str, object] = {}
    cam_top = robot.getDevice('CameraTop')
    if cam_top:
        cam_top.enable(timestep)
        cameras['CameraTop'] = cam_top

    motors: Dict[str, object] = {}
    sensors: Dict[str, object] = {}
    joint_q_idx_map: Dict[str, int] = {}

    for name in config.TRACKED_JOINTS:
        m = robot.getDevice(name)
        if m:
            motors[name] = m
        s = robot.getDevice(name + 'S')
        if s:
            s.enable(timestep)
            sensors[name] = s
        if name in model.names:
            joint_q_idx_map[name] = model.joints[model.getJointId(name)].idx_q

    finger_motors: Dict[str, List[object]] = {'L': [], 'R': []}
    for side in ('L', 'R'):
        for i in range(1, 9):
            m = robot.getDevice(f'{side}Phalanx{i}')
            if m:
                finger_motors[side].append(m)

    return cameras, motors, sensors, joint_q_idx_map, finger_motors


def _wait_for_frame_buffer(
    robot: Supervisor,
    vlm_api: NaoVlmAPI,
    buffer: FrameBuffer,
    timestep: int,
    timeout_s: float,
) -> bool:
    deadline = time.time() + max(0.1, timeout_s)
    while time.time() < deadline:
        if robot.step(timestep) == -1:
            return False
        vlm_api._sync_sensors()
        if len(buffer) >= config.VLM_FRAME_COUNT:
            return True
    return False


def _is_video_file_source(source) -> bool:
    if isinstance(source, int):
        return False
    if not isinstance(source, str):
        return False
    if '://' in source:
        return False
    suffix = Path(source).suffix.lower()
    return suffix in {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def _wait_for_video_progress(
    robot: Supervisor,
    vlm_api: NaoVlmAPI,
    buffer: FrameBuffer,
    timestep: int,
    settle_s: float,
) -> bool:
    deadline = time.time() + max(0.0, settle_s)
    while time.time() < deadline:
        if robot.step(timestep) == -1:
            return False
        vlm_api._sync_sensors()
        if not buffer.is_alive and len(buffer) >= config.VLM_FRAME_COUNT:
            return True
    return True


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _sample_frames_from_video_file(path: str, n: int) -> List[str]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
    finally:
        cap.release()

    if len(frames) < max(1, n):
        return []

    indices = np.linspace(0, len(frames) - 1, n).astype(int)
    out: List[str] = []
    for idx in indices:
        ok, buf = cv2.imencode('.jpg', frames[int(idx)], [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        out.append(base64.b64encode(buf.tobytes()).decode('ascii'))
    return out


def _save_oneshot_artifacts(frames_b64: List[str], rsp, exec_result=None) -> Path:
    out_dir = config.ARTIFACTS_DIR / time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, img_b64 in enumerate(frames_b64, start=1):
        try:
            (out_dir / f'frame_{index:02d}.jpg').write_bytes(base64.b64decode(img_b64))
        except Exception as e:
            _write_text(out_dir / f'frame_{index:02d}.error.txt', str(e))

    semantic_json = json.dumps(rsp.semantic_context or {}, ensure_ascii=False, indent=2)
    _write_text(out_dir / 'semantic_context.json', semantic_json)
    _write_text(out_dir / 'python_code.py', rsp.python_code or '')
    _write_text(out_dir / 'raw_response.txt', rsp.raw_text or '')

    summary = [
        f'ok={rsp.ok}',
        f'elapsed_seconds={rsp.elapsed_seconds:.3f}',
        f'error={rsp.error}',
    ]
    if exec_result is not None:
        summary.extend([
            f'exec_ok={exec_result.ok}',
            f'exec_elapsed_seconds={exec_result.elapsed_seconds:.3f}',
            f'exec_error={exec_result.error}',
        ])
        if exec_result.traceback:
            _write_text(out_dir / 'execution_traceback.txt', exec_result.traceback)
    _write_text(out_dir / 'summary.txt', '\n'.join(summary) + '\n')
    return out_dir


def _run_oneshot_demo(
    robot: Supervisor,
    timestep: int,
    vlm_api: NaoVlmAPI,
    buffer: Optional[FrameBuffer],
    client,
    executor: SandboxExecutor,
) -> None:
    if buffer is None:
        print('[oneshot] ERROR: oneshot mode requires INPUT_MODE=webcam/video source.')
        return
    if client is None:
        print('[oneshot] ERROR: VLM client unavailable; check SDK and llm_api_key.')
        return

    print(f'[oneshot] waiting for at least {config.VLM_FRAME_COUNT} frames...')
    ready = _wait_for_frame_buffer(
        robot=robot,
        vlm_api=vlm_api,
        buffer=buffer,
        timestep=timestep,
        timeout_s=config.ONE_SHOT_BUFFER_TIMEOUT,
    )
    if not ready:
        print('[oneshot] ERROR: frame buffer did not fill before timeout or simulation ended.')
        return

    if _is_video_file_source(config.WEBCAM_SOURCE):
        settle_s = config.ONE_SHOT_VIDEO_SETTLE_SECONDS
        if settle_s <= 0.0:
            settle_s = max(config.VLM_WINDOW_SECONDS, config.FRAME_BUFFER_SECONDS)
        print(f'[oneshot] video source detected; allowing playback to progress for {settle_s:.2f}s before sampling...')
        if not _wait_for_video_progress(
            robot=robot,
            vlm_api=vlm_api,
            buffer=buffer,
            timestep=timestep,
            settle_s=settle_s,
        ):
            print('[oneshot] ERROR: simulation ended while waiting for video progress.')
            return

        direct_frames = _sample_frames_from_video_file(config.WEBCAM_SOURCE, config.VLM_FRAME_COUNT)
        if direct_frames:
            frames = direct_frames
            print(f'[oneshot] sampled {len(frames)} frame(s) directly from source video {config.WEBCAM_SOURCE!r}')
        else:
            frames = buffer.sample_recent(config.VLM_FRAME_COUNT)
            print(f'[oneshot] fallback to buffer sampling: {len(frames)} frame(s) from {config.WEBCAM_SOURCE!r}')
    else:
        frames = buffer.sample_recent(config.VLM_FRAME_COUNT)

    if not frames:
        print('[oneshot] ERROR: failed to sample frames from buffer.')
        return

    if not _is_video_file_source(config.WEBCAM_SOURCE):
        print(f'[oneshot] sampled {len(frames)} frame(s) from {config.WEBCAM_SOURCE!r}')
    print('[oneshot] sending frames to VLM...')
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def _call_vlm() -> None:
        try:
            result_queue.put(client.call(frames))
        except Exception as e:
            result_queue.put(e)

    threading.Thread(target=_call_vlm, name='OneShotVLM', daemon=True).start()

    rsp = None
    deadline = time.time() + max(1.0, config.ONE_SHOT_VLM_TIMEOUT)
    while time.time() < deadline:
        try:
            rsp = result_queue.get_nowait()
            break
        except queue.Empty:
            pass

        if robot.step(timestep) == -1:
            print('[oneshot] simulation ended while waiting for VLM.')
            return
        vlm_api._sync_sensors()

    if rsp is None:
        print(f'[oneshot] ERROR: VLM call timed out after {config.ONE_SHOT_VLM_TIMEOUT:.1f}s')
        return
    if isinstance(rsp, Exception):
        print(f'[oneshot] ERROR: unexpected VLM exception: {rsp}')
        return

    if rsp.ok:
        print(f'[oneshot] VLM done in {rsp.elapsed_seconds:.2f}s')
        print(f'[oneshot] semantic context: {rsp.semantic_context}')
        print(f'[oneshot] code:\n{rsp.python_code}\n')
        exec_result = executor.run(rsp.python_code)
        if (not exec_result.ok) and hasattr(client, 'repair'):
            print(f'[oneshot] first execution failed, asking local VLM to repair: {exec_result.error}')
            repaired_rsp = client.repair(
                frames,
                rsp.semantic_context,
                rsp.python_code,
                exec_result.error or 'execution_failed',
            )
            if repaired_rsp.ok:
                print(f'[oneshot] repaired semantic context: {repaired_rsp.semantic_context}')
                print(f'[oneshot] repaired code:\n{repaired_rsp.python_code}\n')
                repaired_exec_result = executor.run(repaired_rsp.python_code)
                if repaired_exec_result.ok:
                    rsp = repaired_rsp
                    exec_result = repaired_exec_result
                else:
                    print(f'[oneshot] repaired execution still failed: {repaired_exec_result.error}')
        artifact_dir = _save_oneshot_artifacts(frames, rsp, exec_result)
        if exec_result.ok:
            print(f'[oneshot] execution OK in {exec_result.elapsed_seconds:.2f}s')
        else:
            print(f'[oneshot] execution FAILED: {exec_result.error}')
            if exec_result.traceback:
                print(exec_result.traceback)
        print(f'[oneshot] artifacts saved to: {artifact_dir}')
    else:
        artifact_dir = _save_oneshot_artifacts(frames, rsp)
        print(f'[oneshot] VLM FAILED: {rsp.error}')
        print(f'[oneshot] artifacts saved to: {artifact_dir}')

    if config.ONE_SHOT_EXIT_AFTER_EXECUTE:
        print('[oneshot] done; exiting controller.')


def main():
    print('\n' + '=' * 60)
    print(' NAO VLM Embodied Controller')
    print('=' * 60)

    # 1. Pinocchio
    urdf_path, model, data = _build_pinocchio()
    if model is None:
        return

    # 2. Webots
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    cameras, motors, sensors, joint_q_idx_map, finger_motors = _bind_devices(
        robot, model, timestep
    )
    print(f'[init] bound {len(motors)} motors, {len(sensors)} sensors, '
          f'{len(finger_motors["L"])+len(finger_motors["R"])} finger phalanges, '
          f'{len(cameras)} camera(s)')

    # 3. NaoVlmAPI
    vlm_api = NaoVlmAPI(
        robot=robot, pin_model=model, pin_data=data,
        motors=motors, sensors=sensors, cameras=cameras,
        timestep=timestep, joint_q_idx_map=joint_q_idx_map,
        finger_motors=finger_motors,
    )

    # 4. FrameBuffer
    buffer: Optional[FrameBuffer] = None
    if config.INPUT_MODE == 'webcam':
        print(f'[init] FrameBuffer source={config.WEBCAM_SOURCE!r}  '
              f'fps={config.FRAME_BUFFER_FPS}  window={config.FRAME_BUFFER_SECONDS}s')
        buffer = FrameBuffer(
            source=config.WEBCAM_SOURCE,
            buffer_seconds=config.FRAME_BUFFER_SECONDS,
            fps=config.FRAME_BUFFER_FPS,
            backend=config.FRAMEBUFFER_BACKEND,
            frame_width=config.FRAMEBUFFER_WIDTH,
            frame_height=config.FRAMEBUFFER_HEIGHT,
        ).start()
    else:
        print(f'[init] INPUT_MODE={config.INPUT_MODE} - FrameBuffer disabled')

    # 5. VLM client + worker
    client = None
    worker: Optional[VLMWorker] = None
    if _VLM_IMPORT_OK and buffer is not None:
        try:
            client = VLMClient(joint_limits=vlm_api.get_joint_limits())
            print(f'[init] VLMClient ready, model={client.model}')
        except Exception as e:
            print(f'[init] VLMClient disabled: {e}')
            client = None
            worker = None
    if client is not None and buffer is not None and config.RUN_MODE != 'oneshot':
        worker = VLMWorker(client, buffer)

    # 6. Sandbox executor
    executor = SandboxExecutor()
    executor.set_joint_limits(vlm_api.get_joint_limits())
    executor.register_many({
        'move_joint': vlm_api.move_joint,
        'move_joints': vlm_api.move_joints,
        'move_arm_ik': vlm_api.move_arm_ik,
        'set_hand': vlm_api.set_hand,
        'oscillate_joint': vlm_api.oscillate_joint,
        'hold': vlm_api.hold,
        'idle': vlm_api.idle,
    })
    print(f'[init] sandbox exposes: {executor.registered_names}')

    fallback = FallbackPolicy(idle_fn=vlm_api.idle)

    if config.RUN_MODE == 'oneshot':
        print('[init] run mode: oneshot')
        try:
            _run_oneshot_demo(
                robot=robot,
                timestep=timestep,
                vlm_api=vlm_api,
                buffer=buffer,
                client=client,
                executor=executor,
            )
        finally:
            if buffer is not None:
                buffer.stop()
        return

    # 7. State-aware trigger + idle animator
    trigger: Optional[VLMTrigger] = None
    if buffer is not None:
        trigger = VLMTrigger(
            buffer,
            motion_threshold=config.MOTION_THRESHOLD,
            post_action_delay=config.POST_ACTION_DELAY,
            idle_safety_timeout=config.IDLE_SAFETY_TIMEOUT,
        )
        print(f'[init] VLMTrigger: motion>{config.MOTION_THRESHOLD}  '
              f'post_action={config.POST_ACTION_DELAY}s  '
              f'safety={config.IDLE_SAFETY_TIMEOUT}s')

    idle_animator = IdleAnimator(motors, vlm_api._clip_to_motor_limits)
    print(f'[init] IdleAnimator overlays: {idle_animator.stats()["joints"]}')

    # 8. Main loop
    step_count = 0
    print(f'[init] main loop entering. State-aware trigger active. '
          f'Press Ctrl-C or stop Webots to exit.\n')

    last_logged_state = None
    while robot.step(timestep) != -1:
        step_count += 1
        vlm_api._sync_sensors()

        # Idle overlay ticks every main-loop step; primitives take over the
        # loop when they run, so idle naturally pauses during VLM execution.
        idle_animator.tick(robot.getTime())

        # State-aware trigger decides whether to kick the VLM
        if trigger is not None and worker is not None:
            if trigger.state != last_logged_state:
                print(f'[step {step_count}] state -> {trigger.state}')
                last_logged_state = trigger.state

            if not worker.in_flight:
                reason = trigger.consider_trigger()
                if reason is not None and worker.kick():
                    trigger.confirm_fire(reason)
                    trigger.mark_executing()
                    buf_stats = buffer.stats() if buffer else {}
                    tstats = trigger.stats()
                    print(f'[step {step_count}] VLM kick #{worker.total_calls} ({reason})  '
                          f'motion={buf_stats.get("last_motion_score", 0):.2f}  '
                          f'fires: motion={tstats["motion_fires"]} '
                          f'post={tstats["postaction_fires"]} '
                          f'safety={tstats["safety_fires"]}')

            # Drain results
            rsp = worker.poll()
            if rsp is not None:
                if rsp.ok:
                    print(f'[VLM] {rsp.elapsed_seconds:.2f}s  ctx={rsp.semantic_context}')
                    print(f'[VLM] code:\n{rsp.python_code}\n')
                    result = executor.run(rsp.python_code)
                    if result.ok:
                        print(f'[VLM] exec OK in {result.elapsed_seconds:.2f}s')
                        fallback.record_success(rsp)
                    else:
                        print(f'[VLM] exec FAILED: {result.error}')
                        if result.traceback:
                            print(result.traceback)
                        fallback.handle_failure(f'exec_failed:{result.error}')
                    # Whether exec succeeded or failed, open the post-action
                    # observation window so we see the human's reaction.
                    trigger.mark_action_done()
                else:
                    print(f'[VLM] call failed: {rsp.error}  (raw={rsp.raw_text[:200]!r})')
                    fallback.handle_failure(f'call_failed:{rsp.error}')
                    trigger.mark_idle()

    # Shutdown
    if buffer is not None:
        buffer.stop()
    print('[shutdown] main loop exited cleanly.')


if __name__ == '__main__':
    main()
