"""
State-aware VLM trigger (see plan §11.3).

Three states:
  IDLE          - robot standing by, no action in flight
                  FIRES on motion_score > threshold, OR long-silence safety timeout
  EXECUTING     - VLM inference in flight and/or primitive code executing
                  NEVER fires (don't queue up requests)
  POST_ACTION   - action just finished; window to observe human reaction
                  FIRES once after post_action_delay seconds

The executor (main loop) drives state transitions via mark_* methods.
"""
from __future__ import annotations

import time
from typing import Optional


class VLMTrigger:
    STATE_IDLE = 'idle'
    STATE_EXECUTING = 'executing'
    STATE_POST_ACTION = 'post_action'

    def __init__(
        self,
        buffer,
        motion_threshold: float = 5.0,
        post_action_delay: float = 2.0,
        idle_safety_timeout: float = 30.0,
        motion_cooldown: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        buffer
            A FrameBuffer (or anything exposing `last_motion_score: float`).
        motion_threshold
            Mean-absolute-frame-diff above which we consider "someone moved".
        post_action_delay
            Seconds to wait after action completes before the POST_ACTION
            window's observation trigger fires.
        idle_safety_timeout
            Backstop: after this many seconds in IDLE with no motion trigger,
            fire anyway (robustness against a broken motion sensor).
        motion_cooldown
            After a motion-triggered fire, ignore motion re-triggers for this
            long. Prevents one prolonged wave from triggering repeatedly
            while the VLM is still booting the EXECUTING state.
        """
        self.buffer = buffer
        self.motion_threshold = float(motion_threshold)
        self.post_action_delay = float(post_action_delay)
        self.idle_safety_timeout = float(idle_safety_timeout)
        self.motion_cooldown = float(motion_cooldown)

        self.state = self.STATE_IDLE
        self.state_enter_time = time.time()
        self._last_motion_fire_time = 0.0

        # Debugging counters
        self.n_motion_fires = 0
        self.n_postaction_fires = 0
        self.n_safety_fires = 0

    # ------------------------------------------------------------------ state transitions

    def mark_executing(self) -> None:
        self.state = self.STATE_EXECUTING
        self.state_enter_time = time.time()

    def mark_action_done(self) -> None:
        self.state = self.STATE_POST_ACTION
        self.state_enter_time = time.time()

    def mark_idle(self) -> None:
        self.state = self.STATE_IDLE
        self.state_enter_time = time.time()

    # ------------------------------------------------------------------ trigger query / confirm

    def consider_trigger(self) -> Optional[str]:
        """
        Pure query: would the trigger fire right now?
        Returns the trigger reason ('motion' | 'post_action' | 'safety') if
        conditions are met, else None. Does NOT mutate state — caller must
        call confirm_fire(reason) after the VLM kick actually succeeds.

        This split avoids inflating fire counters when a kick is rejected
        upstream (e.g. FrameBuffer not yet populated with enough frames).
        """
        now = time.time()
        elapsed = now - self.state_enter_time

        if self.state == self.STATE_EXECUTING:
            return None

        if self.state == self.STATE_POST_ACTION:
            if elapsed >= self.post_action_delay:
                return 'post_action'
            return None

        # STATE_IDLE
        motion = getattr(self.buffer, 'last_motion_score', 0.0)
        if motion > self.motion_threshold and (now - self._last_motion_fire_time) >= self.motion_cooldown:
            return 'motion'
        if elapsed > self.idle_safety_timeout:
            return 'safety'
        return None

    def confirm_fire(self, reason: str) -> None:
        """
        Called by the main loop AFTER worker.kick() succeeded for `reason`.
        Updates fire counters and timestamps so cooldowns / windows reset.
        State transition to EXECUTING is the caller's responsibility (via
        mark_executing()), kept separate so this stays a pure bookkeeping op.
        """
        now = time.time()
        if reason == 'motion':
            self.n_motion_fires += 1
            self._last_motion_fire_time = now
            self.mark_idle()   # reset state_enter_time so safety timer restarts
        elif reason == 'post_action':
            self.n_postaction_fires += 1
            self.mark_idle()   # consume the post-action window
        elif reason == 'safety':
            self.n_safety_fires += 1
            self._last_motion_fire_time = now
            self.mark_idle()
        else:
            raise ValueError(f'unknown fire reason: {reason!r}')

    # Backwards-compatible convenience: combined query + auto-confirm.
    # Tests use this; production main loop should prefer consider/confirm.
    def should_trigger(self) -> bool:
        reason = self.consider_trigger()
        if reason is None:
            return False
        self.confirm_fire(reason)
        return True

    # ------------------------------------------------------------------ introspection

    def stats(self) -> dict:
        return {
            'state': self.state,
            'elapsed_in_state': time.time() - self.state_enter_time,
            'motion_score': getattr(self.buffer, 'last_motion_score', 0.0),
            'motion_fires': self.n_motion_fires,
            'postaction_fires': self.n_postaction_fires,
            'safety_fires': self.n_safety_fires,
        }
