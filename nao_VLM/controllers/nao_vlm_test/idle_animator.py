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
from typing import Callable, Dict


class IdleAnimator:
    def __init__(
        self,
        motors: Dict[str, object],
        clip_fn: Callable[[object, float], float],
        blend_rate: float = 0.10,
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
            Fraction of remaining distance to cover per tick when easing
            from post-primitive position back to the idle pattern.
            0.10 ≈ 200 ms time constant at 20 ms step.
        """
        self.motors = motors
        self.clip_fn = clip_fn
        self.blend_rate = float(blend_rate)
        self._enabled = True
        self._tick_count = 0

        # Idle formulas: joint -> function(t_seconds) -> angle_radians
        #
        # Keep amplitudes SMALL and only touch upper-body joints that primitives
        # rarely end on a stable target — so the overlay doesn't fight them.
        # (We avoid knees/ankles entirely: disturbing those would risk balance.)
        self._formulas: Dict[str, Callable[[float], float]] = {
            'HeadYaw':        lambda t: 0.08 * math.sin(0.30 * t),
            'LShoulderPitch': lambda t: 1.50 + 0.03 * math.sin(0.50 * t),
            'RShoulderPitch': lambda t: 1.50 + 0.03 * math.sin(0.50 * t + math.pi),
        }

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
            # Low-pass blend from current commanded pos toward idle target
            current = motor.getTargetPosition()
            blended = current + (target - current) * self.blend_rate
            motor.setPosition(self.clip_fn(motor, blended))

    # ------------------------------------------------------------------ debug

    def stats(self) -> dict:
        return {
            'enabled': self._enabled,
            'ticks': self._tick_count,
            'joints': list(self._formulas.keys()),
        }
