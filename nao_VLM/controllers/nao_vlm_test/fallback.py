"""
Three-tier fallback policy for VLM call / sandbox execution failures.

Tier A: retry once (within retry_budget) with the same evidence.
Tier B: replay the most recent successful action (from a small history).
Tier C: idle — call vlm_api.idle(2.0) to keep the robot alive-looking.

The main loop drives the policy:
    fallback.record_success(rsp)              after a successful exec
    decision = fallback.handle_failure(reason)
    if decision.action == 'retry':            re-kick VLM with same frames
    elif decision.action == 'replay':         executor.run(decision.python_code)
    elif decision.action == 'idle':           already played by handle_failure
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Optional


@dataclass
class FallbackDecision:
    action: str            # 'retry' | 'replay' | 'idle'
    python_code: str = ''  # only set for 'replay'
    reason: str = ''


class FallbackPolicy:
    def __init__(
        self,
        idle_fn,
        retry_budget: int = 1,
        history_size: int = 3,
    ) -> None:
        self._idle_fn = idle_fn
        self._retry_budget_default = max(0, int(retry_budget))
        self._retry_budget_remaining = self._retry_budget_default

        # Action history: most-recent-last. Each entry is the VLMResponse-like
        # object stored at record_success time. We only need .python_code on
        # replay, but keep the whole object so future logic (e.g. semantic
        # similarity match) can use it.
        self._history: Deque[Any] = deque(maxlen=max(1, int(history_size)))

        # One-shot guard: once Tier B has fired in the current failure cycle,
        # subsequent failures should not replay again — they fall through to
        # Tier C idle. Reset on record_success (a fresh good action means the
        # next failure cycle starts clean).
        self._tier_b_used_this_cycle = False

        # Counters for evaluation / Fallback Activation Rate metric
        self.n_tier_a = 0
        self.n_tier_b = 0
        self.n_tier_c = 0

    def reset_cycle(self) -> None:
        """Reset retry budget and Tier-B usage flag. Called after a successful
        action AND after Tier-C idle so a fresh failure cycle starts clean."""
        self._retry_budget_remaining = self._retry_budget_default
        self._tier_b_used_this_cycle = False

    def record_success(self, rsp) -> None:
        """Cache a successful VLMResponse and reset the retry budget."""
        # Only cache responses that actually have python_code (a successful
        # response with empty code would be useless to replay).
        code = getattr(rsp, 'python_code', '') or ''
        if code.strip():
            self._history.append(rsp)
        self.reset_cycle()

    def handle_failure(self, reason: str) -> FallbackDecision:
        """
        Return the next action. State machine:
          - retry budget left -> Tier A (retry)
          - history non-empty -> Tier B (replay most recent)
          - else               -> Tier C (idle)
        """
        if self._retry_budget_remaining > 0:
            self._retry_budget_remaining -= 1
            self.n_tier_a += 1
            return FallbackDecision(action='retry', reason=reason)

        if self._history and not self._tier_b_used_this_cycle:
            cached = self._history[-1]
            code = getattr(cached, 'python_code', '') or ''
            if code.strip():
                self.n_tier_b += 1
                self._tier_b_used_this_cycle = True
                # Don't reset the retry budget yet — if the replay also fails,
                # we want to fall to Tier C, not loop on Tier-A retries or
                # repeat the same replay endlessly.
                return FallbackDecision(action='replay', python_code=code, reason=reason)

        # Tier C: idle and reset cycle
        self._idle_fn(2.0)
        self.n_tier_c += 1
        self.reset_cycle()
        return FallbackDecision(action='idle', reason=reason)

    # ------------------------------------------------------------------ introspection

    def stats(self) -> dict:
        return {
            'retry_budget_remaining': self._retry_budget_remaining,
            'history_size': len(self._history),
            'tier_a_fires': self.n_tier_a,
            'tier_b_fires': self.n_tier_b,
            'tier_c_fires': self.n_tier_c,
        }
