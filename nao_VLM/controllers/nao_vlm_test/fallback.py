from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FallbackDecision:
    action: str
    python_code: str = ''
    reason: str = ''


class FallbackPolicy:
    def __init__(self, idle_fn, retry_budget: int = 1) -> None:
        self._idle_fn = idle_fn
        self._retry_budget_default = max(0, int(retry_budget))
        self._retry_budget_remaining = self._retry_budget_default

    def reset_cycle(self) -> None:
        self._retry_budget_remaining = self._retry_budget_default

    def record_success(self, rsp) -> None:
        self.reset_cycle()

    def handle_failure(self, reason: str) -> FallbackDecision:
        if self._retry_budget_remaining > 0:
            self._retry_budget_remaining -= 1
            return FallbackDecision(action='retry', reason=reason)

        self._idle_fn(2.0)
        self.reset_cycle()
        return FallbackDecision(action='idle', reason=reason)
