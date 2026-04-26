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

import math
import os
import queue
import sys
import threading
import time
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Load .env before reading config
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    # Try repo root first, then fall back to cwd (Webots controller cwd is this dir)
    from pathlib import Path
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
        motor = self.motors.get(name)
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
        return f'OK move_joint {name}->{target:.3f}'

    def move_joints(self, joint_angles: Dict[str, float], duration: float,
                    trajectory: str = 'cubic') -> str:
        segments = {}
        for name, angle in joint_angles.items():
            motor = self.motors.get(name)
            if motor is None:
                continue
            start = motor.getTargetPosition()
            target = self._clip_to_motor_limits(motor, float(angle))
            segments[name] = (motor, start, target)
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

        # IK: iterate to convergence (keep existing bugfix: lock floating base).
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
            J[:, :6] = 0.0   # lock floating base so the torso doesn't "absorb" the delta
            dq = np.linalg.pinv(J) @ err
            q = pin.integrate(self.model, q, dq * 0.5)

        # Extract target angles for the Webots motors we control
        target_angles = {}
        for motor_name in self.motors:
            if motor_name in self.model.names:
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
        motor = self.motors.get(name)
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
        return f'OK oscillate_joint {name}'

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
        ).start()
    else:
        print(f'[init] INPUT_MODE={config.INPUT_MODE} - FrameBuffer disabled')

    # 5. VLM client + worker
    worker: Optional[VLMWorker] = None
    if _VLM_IMPORT_OK and buffer is not None:
        try:
            client = VLMClient(joint_limits=vlm_api.get_joint_limits())
            worker = VLMWorker(client, buffer)
            print(f'[init] VLMClient ready, model={client.model}')
        except Exception as e:
            print(f'[init] VLMClient disabled: {e}')
            worker = None

    # 6. Sandbox executor
    executor = SandboxExecutor()
    executor.register_many({
        'move_joint': vlm_api.move_joint,
        'move_joints': vlm_api.move_joints,
        'move_arm_ik': vlm_api.move_arm_ik,
        'oscillate_joint': vlm_api.oscillate_joint,
        'hold': vlm_api.hold,
        'idle': vlm_api.idle,
        'speak': vlm_api.speak,
        # Legacy aliases so existing prompts still work:
        'look_at': vlm_api.look_at,
        'move_arm': vlm_api.move_arm,
        'operate_gripper': vlm_api.operate_gripper,
        'set_posture': vlm_api.set_posture,
    })
    print(f'[init] sandbox exposes: {executor.registered_names}')

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
                    else:
                        print(f'[VLM] exec FAILED: {result.error}')
                        if result.traceback:
                            print(result.traceback)
                    # Whether exec succeeded or failed, open the post-action
                    # observation window so we see the human's reaction.
                    trigger.mark_action_done()
                else:
                    print(f'[VLM] call failed: {rsp.error}  (raw={rsp.raw_text[:200]!r})')
                    # Call failed before any motion happened - back to IDLE
                    # so we can retry on next motion event.
                    trigger.mark_idle()

    # Shutdown
    if buffer is not None:
        buffer.stop()
    print('[shutdown] main loop exited cleanly.')


if __name__ == '__main__':
    main()
