"""
IdleAnimator — subtle "the robot is alive" overlay for the main loop.

Called once per main-loop step. Writes small sinusoidal offsets to HeadYaw
and ShoulderPitches so the robot does not look frozen while the VLM is
inferring or no action is running.

Threading note
--------------
Webots motor writes must happen on a single thread. So this is NOT a
separate thread — it's a function the main loop calls each step. Whenever a
motion primitive is executing, it owns the main loop (runs its own inner
robot.step() calls), so IdleAnimator simply does not get called during that
window. That's the correct behavior: the primitive's commanded motion wins,
idle resumes automatically once the primitive returns.

State continuity
----------------
After a primitive ends, the target joints will be wherever the primitive
left them. Rather than snapping to the idle sine wave (visible jitter),
IdleAnimator low-pass-filters between the motor's current commanded position
and its idle target. Over ~30 ticks the joints ease back to the idle pattern.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Set

import config


class IdleAnimator:
    def __init__(
        self,
        motors: Dict[str, object],
        clip_fn: Callable[[object, float], float],
        blend_rate: float = 0.10,
        relax_blend_rate: float = 0.05,
    ) -> None:
        """
        Parameters
        ----------
        motors
            Dict of {joint_name: Webots motor device}, same as NaoVlmAPI.motors
        clip_fn
            Callable (motor, angle) -> safe angle. Typically
            NaoVlmAPI._clip_to_motor_limits.
        blend_rate
            Fraction of remaining distance to cover per tick for OVERLAY
            joints (the ones with active sinusoidal "alive" motion).
            0.10 ≈ 200 ms time constant at 20 ms step.
        relax_blend_rate
            Slower blend rate used for PASSIVE-RELAXATION joints (joints
            that should ease back to NEUTRAL_POSE between primitives but
            don't get an active oscillation). 0.05 ≈ 400 ms time constant
            so the post-primitive pose remains readable for a moment
            before unwinding to rest.
        """
        self.motors = motors
        self.clip_fn = clip_fn
        self.blend_rate = float(blend_rate)
        self.relax_blend_rate = float(relax_blend_rate)
        self._enabled = True
        self._tick_count = 0

        # Active overlay formulas: joint -> function(t_seconds) -> angle_radians.
        # These produce visible "alive" motion — slow head scan + chest breathing.
        # Amplitudes are picked to be visually perceptible at camera distance
        # but well within motor limits and not disruptive of balance.
        # (We do NOT touch knees/ankles by default — disturbing them risks
        # bipedal balance. See Step 6 Part D in the plan for an opt-in path.)
        self._formulas: Dict[str, Callable[[float], float]] = {
            # Slow horizontal head scan ~11.5° peak. Period ~10 s — "looking
            # around" not "twitchy".
            'HeadYaw':        lambda t: 0.20 * math.sin(0.60 * t),
            # Subtle head nod in counter-phase with the chest (head sinks
            # slightly as chest rises).
            'HeadPitch':      lambda t: 0.06 * math.sin(1.20 * t + math.pi),
            # Shoulder breathing — visible chest rise/fall, L/R opposite
            # phase gives a natural mild sway.
            'LShoulderPitch': lambda t: 1.50 + 0.08 * math.sin(1.20 * t),
            'RShoulderPitch': lambda t: 1.50 + 0.08 * math.sin(1.20 * t + math.pi),
            # Step 6 Part D: bilateral knee bounce, paired in phase
            # (symmetric squat). Amplitude ±0.12 rad (~7°) → hip rises and
            # falls by ~14 mm, clearly visible from a front view. Center
            # 0.10 rad is a slight standing crouch; the robot settles
            # there within ~1 s of startup. Ankles move in counter-phase
            # at half-amplitude to keep the feet flat, so the CoM stays
            # roughly over the feet and paired knee motion does not
            # topple the robot. Frequency 0.80 rad/s (period ~8 s) matches
            # the shoulder breathing rhythm above, so the chest rise/fall
            # and knee bounce read as one coherent breathing motion. If
            # the robot ever rocks or loses balance, drop the frequency
            # back to 0.40 (period ~16 s) or comment out these four lines.
            'LKneePitch':  lambda t: 0.10 + 0.12 * math.sin(1.20 * t),
            'RKneePitch':  lambda t: 0.10 + 0.12 * math.sin(1.20 * t),
            'LAnklePitch': lambda t: -0.05 - 0.06 * math.sin(1.20 * t),
            'RAnklePitch': lambda t: -0.05 - 0.06 * math.sin(1.20 * t),
        }
        self._overlay_joints: Set[str] = set(self._formulas.keys())

        # Passive relaxation: every NEUTRAL_POSE joint that is NOT already
        # an active overlay target gets a constant-target formula. This
        # makes the robot gradually unwind to rest after a primitive ends
        # at a non-rest pose (e.g. elbow bent, hand raised). The slower
        # relax_blend_rate preserves the post-primitive pose for ~0.4 s
        # before unwinding begins to dominate. Without this, joints like
        # ElbowRoll / ShoulderRoll / etc. stay frozen wherever the last
        # primitive left them, looking like the robot is "stuck" between
        # VLM kicks.
        for jname, rest_value in config.NEUTRAL_POSE.items():
            if jname in self._formulas:
                continue  # already covered by an active overlay
            # The (lambda r: lambda _t: r)(...) capture-by-value pattern
            # avoids Python's late-binding-closure trap.
            self._formulas[jname] = (lambda r: lambda _t: r)(float(rest_value))

    # ------------------------------------------------------------------ control

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ------------------------------------------------------------------ main-loop hook

    def tick(self, sim_time_s: float) -> None:
        """Emit one step of idle motion. Called once per main-loop iteration."""
        if not self._enabled:
            return
        self._tick_count += 1
        for name, formula in self._formulas.items():
            motor = self.motors.get(name)
            if motor is None:
                continue
            target = formula(sim_time_s)
            # Use the active blend rate for overlay joints (faster, follows
            # the sinusoidal pattern) and the slower relax rate for passive
            # joints (so post-primitive poses remain readable briefly).
            rate = (self.blend_rate if name in self._overlay_joints
                    else self.relax_blend_rate)
            current = motor.getTargetPosition()
            blended = current + (target - current) * rate
            motor.setPosition(self.clip_fn(motor, blended))

    # ------------------------------------------------------------------ debug

    def stats(self) -> dict:
        return {
            'enabled': self._enabled,
            'ticks': self._tick_count,
            'joints': list(self._formulas.keys()),
        }
