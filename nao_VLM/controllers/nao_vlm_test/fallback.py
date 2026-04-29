from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


@dataclass
class FallbackDecision:
    action: str
    python_code: str = ''
    reason: str = ''


class FallbackPolicy:
    def __init__(self, idle_fn, retry_budget: int = 1, history_size: int = 3) -> None:
        self._idle_fn = idle_fn
        self._retry_budget_default = max(0, int(retry_budget))
        self._retry_budget_remaining = self._retry_budget_default
        self._history: Deque[Dict[str, object]] = deque(maxlen=max(1, int(history_size)))

    def reset_cycle(self) -> None:
        self._retry_budget_remaining = self._retry_budget_default

    def record_success(self, rsp) -> None:
        if getattr(rsp, 'ok', False) and getattr(rsp, 'python_code', ''):
            self._history.append({
                'python_code': rsp.python_code,
                'semantic_context': dict(getattr(rsp, 'semantic_context', {}) or {}),
            })
        self.reset_cycle()

    def handle_failure(self, reason: str) -> FallbackDecision:
        if self._retry_budget_remaining > 0:
            self._retry_budget_remaining -= 1
            return FallbackDecision(action='retry', reason=reason)

        if self._history:
            last = self._history[-1]
            return FallbackDecision(
                action='replay',
                python_code=str(last.get('python_code') or ''),
                reason=reason,
            )

        self._idle_fn(2.0)
        self.reset_cycle()
        return FallbackDecision(action='idle', reason=reason)

