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
from typing import Any, Dict, List, Optional

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
from metrics_recorder import MetricsRecorder
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
        self._aborted = False
        self.metrics_recorder = None

    def _normalize_arm_side(self, side: str) -> str:
        if not isinstance(side, str):
            return side
        normalized = side.strip().lower()
        alias_map = {
            'l': 'left',
            'left': 'left',
            'left_arm': 'left',
            'left_hand': 'left',
            'r': 'right',
            'right': 'right',
            'right_arm': 'right',
            'right_hand': 'right',
        }
        return alias_map.get(normalized, side)

    # ------------------------------------------------------------------ internals

    def _sync_sensors(self) -> None:
        """Mirror Webots joint sensors -> Pinocchio q."""
        for name, sensor in self.sensors.items():
            idx = self.joint_q_idx_map.get(name)
            if idx is not None:
                self.q_current[idx] = sensor.getValue()

    def set_metrics_recorder(self, recorder) -> None:
        self.metrics_recorder = recorder

    def _record_metrics_step(self) -> None:
        if self.metrics_recorder is None:
            return
        sim_time = self.robot.getTime()
        try:
            self.metrics_recorder.record_step(
                sim_time=sim_time,
                q_current=self.q_current,
                sensors=self.sensors,
                model=self.model,
                data=self.data,
            )
        except Exception as exc:
            print(f'[metrics] joint-state logging failed: {exc}')
        # Throttled robot-motion screenshot (no-op unless capture is armed).
        try:
            self.metrics_recorder.maybe_capture_motion_frame(self.robot, sim_time)
        except Exception as exc:
            print(f'[metrics] motion-frame capture failed: {exc}')

    def _step(self) -> bool:
        """Advance simulation by one timestep, sync sensors. Returns False if sim ended."""
        if self.robot.step(self.timestep) == -1:
            self._aborted = True
            return False
        self._sync_sensors()
        self._record_metrics_step()
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
        # WristYaw is intentionally excluded: it is a pure rotation about the
        # wrist axis and contributes ~0 to the hand position objective, so the
        # position-only DLS solver in move_arm_ik cannot drive it meaningfully
        # but the integration step would still leave it at a random-walk value.
        # The VLM can still command it directly via move_joint / oscillate_joint.
        prefix = 'L' if side == 'left' else 'R'
        return [
            f'{prefix}ShoulderPitch',
            f'{prefix}ShoulderRoll',
            f'{prefix}ElbowYaw',
            f'{prefix}ElbowRoll',
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
        side = self._normalize_arm_side(side)
        if side not in ('left', 'right'):
            return f'ERROR: invalid side {side!r}'
        try:
            target_pos = np.asarray(xyz, dtype=float).reshape(3)
        except Exception:
            return f'ERROR: xyz must be a 3-vector, got {xyz!r}'

        frame_id = self.hand_frames[side]
        q = self.q_current.copy()

        # Reset the wrist roll to neutral before the IK solve so any drift
        # accumulated in q_current from previous primitives does not survive
        # into the commanded readback. WristYaw is not in the IK chain (see
        # _side_arm_joint_names), so the solver leaves this value alone.
        wrist_name = f"{'L' if side == 'left' else 'R'}WristYaw"
        if wrist_name in self.model.names:
            wjid = self.model.getJointId(wrist_name)
            wqi = self.model.joints[wjid].idx_q
            q[wqi] = 0.0

        # Seed ElbowYaw to its anatomically-natural branch before the IK solve.
        # NAO's ElbowYaw has ~±2.09 rad range and exposes a redundant rotational
        # DoF the human elbow lacks, so the position-only IK admits two valid
        # solutions per target: forearm natural (~+1.2 on R, ~-1.2 on L) and
        # forearm flipped 180° around the upper arm (opposite sign). The solver
        # converges to whichever branch is closer to the seed. By overriding the
        # seed to NEUTRAL_POSE the solver is locked onto the natural branch and
        # cannot drift across the zero barrier during the 30 damped steps.
        elbow_yaw_name = f"{'L' if side == 'left' else 'R'}ElbowYaw"
        if elbow_yaw_name in self.model.names:
            ejid = self.model.getJointId(elbow_yaw_name)
            eqi = self.model.joints[ejid].idx_q
            q[eqi] = config.NEUTRAL_POSE.get(elbow_yaw_name, q[eqi])

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
        side = self._normalize_arm_side(side)
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
        hp = self.motors.get('HeadPitch')
        lsp = self.motors.get('LShoulderPitch')
        rsp = self.motors.get('RShoulderPitch')
        for i in range(1, steps + 1):
            t = (i / steps) * dur
            if hy is not None:
                hy.setPosition(self._clip_to_motor_limits(hy, 0.04 * math.sin(0.35 * t)))
            if hp is not None:
                hp.setPosition(self._clip_to_motor_limits(hp, 0.03 * math.sin(0.28 * t)))
            # Anti-phase subtle breathing on shoulders
            if lsp is not None:
                lsp.setPosition(self._clip_to_motor_limits(
                    lsp, 1.45 + 0.015 * math.sin(0.45 * t)))
            if rsp is not None:
                rsp.setPosition(self._clip_to_motor_limits(
                    rsp, 1.45 + 0.015 * math.sin(0.45 * t + math.pi)))
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

    def kick_with_frames(self, frames: List[str]) -> bool:
        if self._in_flight.is_set():
            return False
        if not frames:
            return False
        self._in_flight.set()
        self.total_calls += 1
        t = threading.Thread(
            target=self._run, args=(list(frames),), name='VLMWorker', daemon=True
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
        vlm_api._record_metrics_step()
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
        vlm_api._record_metrics_step()
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


def _response_payload(rsp) -> dict:
    if rsp is None:
        return {}
    return {
        'ok': bool(getattr(rsp, 'ok', False)),
        'elapsed_seconds': float(getattr(rsp, 'elapsed_seconds', 0.0) or 0.0),
        'error': getattr(rsp, 'error', None),
        'semantic_context': getattr(rsp, 'semantic_context', {}) or {},
        'python_code': getattr(rsp, 'python_code', '') or '',
        'raw_text': getattr(rsp, 'raw_text', '') or '',
    }


def _exec_payload(exec_result) -> dict:
    if exec_result is None:
        return {'ok': False, 'elapsed_seconds': 0.0, 'error': 'not_executed'}
    return {
        'ok': bool(getattr(exec_result, 'ok', False)),
        'elapsed_seconds': float(getattr(exec_result, 'elapsed_seconds', 0.0) or 0.0),
        'error': getattr(exec_result, 'error', None),
        'traceback': getattr(exec_result, 'traceback', None),
    }


def _decode_frame_b64(img_b64: str) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame if frame is not None else None
    except Exception:
        return None


def _resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 0 or h <= 0:
        return frame
    new_h = max(1, int(round(h * float(width) / float(w))))
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def _put_label(frame: np.ndarray, label: str) -> np.ndarray:
    # Prepend a separate label band ABOVE the frame rather than painting over
    # its top pixels, so no image content (e.g. the robot's head) is hidden.
    band_h = 30
    band = np.full((band_h, frame.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    cv2.putText(
        band,
        label,
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([band, frame])


def _save_input_contact_sheet(frames_b64: List[str], out_dir: Path) -> Optional[Path]:
    frames = []
    for index, img_b64 in enumerate(frames_b64, start=1):
        frame = _decode_frame_b64(img_b64)
        if frame is None:
            continue
        frame = _resize_to_width(frame, 240)
        frames.append(_put_label(frame, f'Input frame {index}'))
    if not frames:
        return None

    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            pad = np.full((max_h - frame.shape[0], frame.shape[1], 3), 245, dtype=np.uint8)
            frame = np.vstack([frame, pad])
        padded.append(frame)
    sheet = np.hstack(padded)
    path = out_dir / 'input_contact_sheet.jpg'
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _save_robot_motion_contact_sheet(frame_paths: List[str], out_dir: Path) -> Optional[Path]:
    """Horizontal labeled strip of the throttled robot-motion screenshots.

    Mirrors `_save_input_contact_sheet`, but reads frames from disk (the
    metrics recorder writes them as numbered JPEGs) instead of base64.
    """
    frames = []
    for index, fp in enumerate(frame_paths, start=1):
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = _resize_to_width(img, 240)
        frames.append(_put_label(img, f'Robot motion {index}'))
    if not frames:
        return None

    max_h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < max_h:
            pad = np.full((max_h - frame.shape[0], frame.shape[1], 3), 245, dtype=np.uint8)
            frame = np.vstack([frame, pad])
        padded.append(frame)
    sheet = np.hstack(padded)
    path = out_dir / 'robot_motion_contact_sheet.jpg'
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _save_demo_summary(artifact_dir: Path, screenshot_path: Optional[Path] = None) -> Optional[Path]:
    contact_path = artifact_dir / 'input_contact_sheet.jpg'
    # Prefer the multi-frame motion sheet; fall back to the single screenshot.
    motion_sheet_path = artifact_dir / 'robot_motion_contact_sheet.jpg'
    if motion_sheet_path.exists():
        robot_path: Path = motion_sheet_path
        robot_label = 'Webots robot motion sequence'
    else:
        robot_path = Path(screenshot_path or (artifact_dir / 'robot_response.png'))
        robot_label = 'Final Webots robot response'
    if not contact_path.exists() or not robot_path.exists():
        return None

    input_img = cv2.imread(str(contact_path), cv2.IMREAD_COLOR)
    robot_img = cv2.imread(str(robot_path), cv2.IMREAD_COLOR)
    if input_img is None or robot_img is None:
        return None

    target_w = max(640, input_img.shape[1])
    input_img = _resize_to_width(input_img, target_w)
    robot_img = _resize_to_width(robot_img, target_w)
    input_img = _put_label(input_img, 'Human input frames sampled for VLM')
    robot_img = _put_label(robot_img, robot_label)

    spacer = np.full((24, target_w, 3), 255, dtype=np.uint8)
    summary = np.vstack([input_img, spacer, robot_img])
    path = artifact_dir / 'demo_summary.jpg'
    cv2.imwrite(str(path), summary, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


def _hold_simulation(
    robot: Supervisor,
    timestep: int,
    vlm_api: NaoVlmAPI,
    seconds: float,
    reason: str,
) -> None:
    seconds = max(0.0, float(seconds))
    if seconds <= 0.0:
        return
    print(f'[sim] holding for {seconds:.2f}s ({reason}); close Webots to end earlier.')
    deadline = time.time() + seconds
    while time.time() < deadline:
        if robot.step(timestep) == -1:
            print('[sim] Webots closed during hold.')
            return
        vlm_api._sync_sensors()
        vlm_api._record_metrics_step()


def _save_oneshot_artifacts(frames_b64: List[str], rsp, exec_result=None,
                            out_dir: Optional[Path] = None,
                            timeline: Optional[List[Dict[str, Any]]] = None) -> Path:
    out_dir = out_dir or (config.ARTIFACTS_DIR / time.strftime('%Y%m%d_%H%M%S'))
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, img_b64 in enumerate(frames_b64, start=1):
        try:
            (out_dir / f'frame_{index:02d}.jpg').write_bytes(base64.b64decode(img_b64))
        except Exception as e:
            _write_text(out_dir / f'frame_{index:02d}.error.txt', str(e))
    _save_input_contact_sheet(frames_b64, out_dir)

    semantic_json = json.dumps(rsp.semantic_context or {}, ensure_ascii=False, indent=2)
    _write_text(out_dir / 'semantic_context.json', semantic_json)
    _write_text(out_dir / 'python_code.py', rsp.python_code or '')
    _write_text(out_dir / 'raw_response.txt', rsp.raw_text or '')
    if timeline is not None:
        _write_text(out_dir / 'timeline.json', json.dumps(timeline, ensure_ascii=False, indent=2) + '\n')

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
    fallback: FallbackPolicy,
    metrics_recorder: Optional[MetricsRecorder] = None,
) -> None:
    if buffer is None:
        print('[oneshot] ERROR: oneshot mode requires INPUT_MODE=webcam/video source.')
        return
    if client is None:
        print('[oneshot] ERROR: VLM client unavailable; check SDK and llm_api_key.')
        return

    t0 = time.time()
    timeline: List[Dict[str, Any]] = []

    def stage(name: str, detail: str = '') -> None:
        elapsed = time.time() - t0
        timeline.append({'elapsed_seconds': elapsed, 'stage': name, 'detail': detail})
        suffix = f' - {detail}' if detail else ''
        print(f'[oneshot +{elapsed:06.2f}s] {name}{suffix}')

    stage('waiting_for_frames', f'target={config.VLM_FRAME_COUNT} source={config.WEBCAM_SOURCE!r}')
    ready = _wait_for_frame_buffer(
        robot=robot,
        vlm_api=vlm_api,
        buffer=buffer,
        timestep=timestep,
        timeout_s=config.ONE_SHOT_BUFFER_TIMEOUT,
    )
    if not ready:
        stage('error', 'frame buffer did not fill before timeout or simulation ended')
        return

    if _is_video_file_source(config.WEBCAM_SOURCE):
        settle_s = config.ONE_SHOT_VIDEO_SETTLE_SECONDS
        if settle_s <= 0.0:
            settle_s = max(config.VLM_WINDOW_SECONDS, config.FRAME_BUFFER_SECONDS)
        stage('video_playback_settle', f'{settle_s:.2f}s before sampling')
        if not _wait_for_video_progress(
            robot=robot,
            vlm_api=vlm_api,
            buffer=buffer,
            timestep=timestep,
            settle_s=settle_s,
        ):
            stage('error', 'simulation ended while waiting for video progress')
            return

        direct_frames = _sample_frames_from_video_file(config.WEBCAM_SOURCE, config.VLM_FRAME_COUNT)
        if direct_frames:
            frames = direct_frames
            stage('sampled_input_frames', f'{len(frames)} direct frames from source video')
        else:
            frames = buffer.sample_recent(config.VLM_FRAME_COUNT)
            stage('sampled_input_frames', f'{len(frames)} buffered frames from source video')
    else:
        frames = buffer.sample_recent(config.VLM_FRAME_COUNT)

    if not frames:
        stage('error', 'failed to sample frames')
        return

    if not _is_video_file_source(config.WEBCAM_SOURCE):
        stage('sampled_input_frames', f'{len(frames)} frames from {config.WEBCAM_SOURCE!r}')
    stage('vlm_request_start', f'backend={config.VLM_BACKEND} model={getattr(client, "model", "")}')
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
        vlm_api._record_metrics_step()

    if rsp is None:
        stage('vlm_timeout', f'after {config.ONE_SHOT_VLM_TIMEOUT:.1f}s')
        if metrics_recorder is not None:
            metrics_recorder.write_result({
                'status': 'vlm_timeout',
                'input': {'mode': config.INPUT_MODE, 'source': str(config.WEBCAM_SOURCE)},
                'frames_count': len(frames),
                'vlm_response': {},
                'exec_outcome': _exec_payload(None),
                'fallback_stats': fallback.stats(),
                'timeline': timeline,
            })
        return
    if isinstance(rsp, Exception):
        stage('vlm_exception', str(rsp))
        if metrics_recorder is not None:
            metrics_recorder.write_result({
                'status': 'vlm_exception',
                'input': {'mode': config.INPUT_MODE, 'source': str(config.WEBCAM_SOURCE)},
                'frames_count': len(frames),
                'vlm_response': {'error': str(rsp)},
                'exec_outcome': _exec_payload(None),
                'fallback_stats': fallback.stats(),
                'timeline': timeline,
            })
        return

    exec_result = None
    motion_frames: List[str] = []
    motion_sheet_path: Optional[Path] = None
    if rsp.ok:
        stage('vlm_response_received', f'{rsp.elapsed_seconds:.2f}s')
        print(f'[oneshot] semantic context: {rsp.semantic_context}')
        print(f'[oneshot] code:\n{rsp.python_code}\n')
        stage('robot_execution_start')
        # Arm throttled robot-motion screenshots for the execution window.
        if metrics_recorder is not None:
            metrics_recorder.begin_motion_capture()
        exec_result = executor.run(rsp.python_code)
        if (not exec_result.ok) and hasattr(client, 'repair'):
            stage('robot_execution_failed_repairing', exec_result.error or '')
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
        # Disarm capture and build the labeled motion contact sheet.
        if metrics_recorder is not None:
            motion_frames = metrics_recorder.end_motion_capture()
            motion_sheet_path = _save_robot_motion_contact_sheet(
                motion_frames, metrics_recorder.run_dir)
            stage('robot_motion_frames_captured', f'{len(motion_frames)} frames')
        artifact_dir = _save_oneshot_artifacts(
            frames,
            rsp,
            exec_result,
            out_dir=(metrics_recorder.run_dir if metrics_recorder is not None else None),
            timeline=timeline,
        )
        if exec_result.ok:
            stage('robot_execution_ok', f'{exec_result.elapsed_seconds:.2f}s')
            fallback.record_success(rsp)
        else:
            stage('robot_execution_failed', exec_result.error or '')
            if exec_result.traceback:
                print(exec_result.traceback)
        stage('artifacts_saved', str(artifact_dir))
    else:
        artifact_dir = _save_oneshot_artifacts(
            frames,
            rsp,
            out_dir=(metrics_recorder.run_dir if metrics_recorder is not None else None),
            timeline=timeline,
        )
        stage('vlm_failed', rsp.error or '')
        stage('artifacts_saved', str(artifact_dir))

    if metrics_recorder is not None:
        screenshot_path = metrics_recorder.export_screenshot(robot)
        demo_summary_path = _save_demo_summary(artifact_dir, screenshot_path)
        stage('robot_screenshot_saved', str(screenshot_path or ''))
        _write_text(artifact_dir / 'timeline.json', json.dumps(timeline, ensure_ascii=False, indent=2) + '\n')
        metrics_recorder.write_result({
            'status': 'ok' if (rsp.ok and exec_result is not None and exec_result.ok) else 'failed',
            'input': {'mode': config.INPUT_MODE, 'source': str(config.WEBCAM_SOURCE)},
            'frames_count': len(frames),
            'vlm_response': _response_payload(rsp),
            'exec_outcome': _exec_payload(exec_result),
            'fallback_stats': fallback.stats(),
            'artifact_dir': str(artifact_dir),
            'timeline': timeline,
            'artifacts': {
                'input_contact_sheet': str(artifact_dir / 'input_contact_sheet.jpg'),
                'demo_summary': str(demo_summary_path or ''),
                'timeline': str(artifact_dir / 'timeline.json'),
                'robot_motion_frames': motion_frames,
                'robot_motion_contact_sheet': str(motion_sheet_path or ''),
            },
        })

    if config.ONE_SHOT_EXIT_AFTER_EXECUTE:
        stage('done', 'controller will exit after optional post-execution hold')
        if 'artifact_dir' in locals():
            _write_text(artifact_dir / 'timeline.json', json.dumps(timeline, ensure_ascii=False, indent=2) + '\n')


def _run_replay_demo(
    robot: Supervisor,
    timestep: int,
    vlm_api: NaoVlmAPI,
    executor: SandboxExecutor,
    metrics_recorder: Optional[MetricsRecorder] = None,
) -> None:
    code_path = Path(config.REPLAY_CODE_PATH).expanduser()
    if not code_path.is_absolute():
        code_path = (Path(__file__).resolve().parent.parent.parent.parent / code_path).resolve()
    if not code_path.exists():
        print(f'[replay] ERROR: code file not found: {code_path}')
        return

    code = code_path.read_text(encoding='utf-8')
    if not code.strip():
        print(f'[replay] ERROR: code file is empty: {code_path}')
        return

    if config.REPLAY_START_DELAY > 0.0:
        print(f'[replay] waiting {config.REPLAY_START_DELAY:.2f}s before execution...')
        steps = max(1, int(round(config.REPLAY_START_DELAY * 1000.0 / timestep)))
        for _ in range(steps):
            if robot.step(timestep) == -1:
                print('[replay] simulation ended during start delay.')
                return
            vlm_api._sync_sensors()
            vlm_api._record_metrics_step()

    print(f'[replay] executing precomputed code from: {code_path}')
    print(f'[replay] code:\n{code}\n')
    result = executor.run(code)
    if result.ok:
        print(f'[replay] execution OK in {result.elapsed_seconds:.2f}s')
    else:
        print(f'[replay] execution FAILED: {result.error}')
        if result.traceback:
            print(result.traceback)
    if metrics_recorder is not None:
        metrics_recorder.export_screenshot(robot)
        metrics_recorder.write_result({
            'status': 'ok' if result.ok else 'failed',
            'input': {'mode': 'replay', 'source': str(code_path)},
            'frames_count': 0,
            'vlm_response': {
                'ok': True,
                'semantic_context': {},
                'python_code': code,
                'raw_text': '',
                'error': None,
                'elapsed_seconds': 0.0,
            },
            'exec_outcome': _exec_payload(result),
            'fallback_stats': {},
            'artifact_dir': str(metrics_recorder.run_dir),
        })


def _request_simulation_quit(robot: Supervisor, status: int = 0) -> None:
    try:
        print(f'[sim] requesting Webots quit with status={status}')
        robot.simulationQuit(status)
    except Exception as exc:
        print(f'[sim] WARNING: simulationQuit failed: {exc}')


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

    metrics_recorder = MetricsRecorder.from_env()
    if metrics_recorder is not None:
        vlm_api.set_metrics_recorder(metrics_recorder)
        print(f'[init] metrics enabled: {metrics_recorder.run_dir}')

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
            if config.EVAL_METHOD == 'rule_baseline' or config.VLM_BACKEND == 'rule_baseline':
                repo_root = str(config.REPO_ROOT)
                if repo_root not in sys.path:
                    sys.path.insert(0, repo_root)
                from evaluation.rule_baseline import RuleBaselineClient
                client = RuleBaselineClient(joint_limits=vlm_api.get_joint_limits())
            else:
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
    executor.set_metrics_recorder(metrics_recorder)

    def move_head(yaw: float, pitch: float, duration: float = 0.2, trajectory: str = 'min_jerk') -> str:
        return vlm_api.move_joints(
            {
                'HeadYaw': float(yaw),
                'HeadPitch': float(pitch),
            },
            duration=float(duration),
            trajectory=trajectory,
        )

    # Active primitive set (matches the VLM prompt). Legacy duplicates
    # (operate_gripper -> set_hand, look_at -> move_head, move_arm -> move_arm_ik,
    # set_posture -> upper-body composition) are intentionally NOT registered so
    # the VLM cannot accidentally pick deprecated paths or bypass the
    # forbidden-lower-body invariant enforced by SandboxExecutor.validate().
    executor.register_many({
        'move_joint': vlm_api.move_joint,
        'move_joints': vlm_api.move_joints,
        'move_arm_ik': vlm_api.move_arm_ik,
        'move_head': move_head,
        'set_hand': vlm_api.set_hand,
        'oscillate_joint': vlm_api.oscillate_joint,
        'hold': vlm_api.hold,
        'idle': vlm_api.idle,
    })
    print(f'[init] sandbox exposes: {executor.registered_names}')

    fallback = FallbackPolicy(idle_fn=vlm_api.idle)

    if config.RUN_MODE == 'replay':
        print('[init] run mode: replay')
        try:
            _run_replay_demo(robot, timestep, vlm_api, executor, metrics_recorder=metrics_recorder)
        finally:
            _hold_simulation(
                robot,
                timestep,
                vlm_api,
                config.ONE_SHOT_POST_EXECUTION_SECONDS,
                'replay complete',
            )
            if buffer is not None:
                buffer.stop()
            if config.ONE_SHOT_EXIT_AFTER_EXECUTE:
                _request_simulation_quit(robot, 0)
        return

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
                fallback=fallback,
                metrics_recorder=metrics_recorder,
            )
        finally:
            _hold_simulation(
                robot,
                timestep,
                vlm_api,
                config.ONE_SHOT_POST_EXECUTION_SECONDS,
                'oneshot complete',
            )
            if buffer is not None:
                buffer.stop()
            if config.ONE_SHOT_EXIT_AFTER_EXECUTE:
                _request_simulation_quit(robot, 0)
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
    last_trigger_frames: List[str] = []
    print(f'[init] main loop entering. State-aware trigger active. '
          f'Press Ctrl-C or stop Webots to exit.\n')

    last_logged_state = None
    while robot.step(timestep) != -1:
        step_count += 1
        vlm_api._sync_sensors()
        vlm_api._record_metrics_step()

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
                    last_trigger_frames = buffer.sample_recent(config.VLM_FRAME_COUNT) if buffer is not None else []
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
                    ctx = rsp.semantic_context or {}
                    md = ctx.get('motion_dynamics', '?')
                    aff = ctx.get('affect', '?')
                    dist = ctx.get('social_distance', '?')
                    conf = ctx.get('confidence', '?')
                    human_intent = ctx.get('intent', '(not set)')
                    robot_intent = ctx.get('robot_intent', '(not set)')
                    print(f'[VLM] {rsp.elapsed_seconds:.2f}s  '
                          f'{md}/{aff}/{dist}  conf={conf}')
                    print(f'[VLM]   human: {human_intent}')
                    print(f'[VLM]   robot: {robot_intent}')
                    print(f'[VLM] code:\n{rsp.python_code}\n')
                    result = executor.run(rsp.python_code)
                    if result.ok:
                        print(f'[VLM] exec OK in {result.elapsed_seconds:.2f}s')
                        fallback.record_success(rsp)
                    else:
                        print(f'[VLM] exec FAILED: {result.error}')
                        if result.traceback:
                            print(result.traceback)
                        decision = fallback.handle_failure(f'exec_failed:{result.error}')
                        if decision.action == 'retry' and worker is not None and last_trigger_frames:
                            print(f'[fallback] tier A retry after exec failure: {decision.reason}')
                            if worker.kick_with_frames(last_trigger_frames):
                                trigger.mark_executing()
                                continue
                        elif decision.action == 'replay':
                            print(f'[fallback] tier B replay after exec failure: {decision.reason}')
                            replay_result = executor.run(decision.python_code)
                            if replay_result.ok:
                                print(f'[fallback] replay exec OK in {replay_result.elapsed_seconds:.2f}s')
                            else:
                                print(f'[fallback] replay exec FAILED: {replay_result.error}')
                        # decision.action == 'idle' was already handled by
                        # handle_failure (it ran idle internally).
                    # Whether exec succeeded or failed, open the post-action
                    # observation window so we see the human's reaction.
                    trigger.mark_action_done()
                else:
                    print(f'[VLM] call failed: {rsp.error}  (raw={rsp.raw_text[:200]!r})')
                    decision = fallback.handle_failure(f'call_failed:{rsp.error}')
                    if decision.action == 'retry' and worker is not None and last_trigger_frames:
                        print(f'[fallback] tier A retry after call failure: {decision.reason}')
                        if worker.kick_with_frames(last_trigger_frames):
                            trigger.mark_executing()
                            continue
                    elif decision.action == 'replay':
                        print(f'[fallback] tier B replay after call failure: {decision.reason}')
                        replay_result = executor.run(decision.python_code)
                        if replay_result.ok:
                            print(f'[fallback] replay exec OK in {replay_result.elapsed_seconds:.2f}s')
                        else:
                            print(f'[fallback] replay exec FAILED: {replay_result.error}')
                        # After a replay, open the post-action window so we
                        # observe the human's reaction to the cached response.
                        trigger.mark_action_done()
                        continue
                    trigger.mark_idle()

    # Shutdown
    if buffer is not None:
        buffer.stop()
    print('[shutdown] main loop exited cleanly.')


if __name__ == '__main__':
    main()
