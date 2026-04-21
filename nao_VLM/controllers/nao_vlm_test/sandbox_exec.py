"""
Sandbox executor — runs VLM-generated primitive compositions in a restricted
Python environment, using bound methods from NaoVlmAPI as the whitelisted API.

Phase 0: restrict builtins, wire bound methods, catch exceptions.
Phase 4 (later): add joint-limit pre-check and static AST validation.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


# Minimal builtins that VLM-generated code may reasonably want. Everything
# else (open, import, eval, exec, __import__, etc.) is blocked.
_SAFE_BUILTINS: Dict[str, Any] = {
    'abs': abs, 'min': min, 'max': max, 'round': round,
    'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
    'sum': sum, 'sorted': sorted, 'list': list, 'dict': dict, 'tuple': tuple,
    'int': int, 'float': float, 'str': str, 'bool': bool,
    'True': True, 'False': False, 'None': None,
    'print': print,
}


@dataclass
class SandboxResult:
    ok: bool
    elapsed_seconds: float
    error: Optional[str] = None
    traceback: Optional[str] = None


class SandboxExecutor:
    """
    Executes VLM code with a whitelist of API callables.

    Usage:
        executor = SandboxExecutor()
        executor.register('move_joint', vlm_api.move_joint)
        executor.register('move_arm_ik', vlm_api.move_arm_ik)
        ...
        result = executor.run(vlm_code_str)
    """

    def __init__(self) -> None:
        self._api: Dict[str, Callable[..., Any]] = {}
        # time is whitelisted via a thin shim so VLM can do time.sleep(0.2)
        # if it wants; primitives themselves are blocking so this is rare.
        self._extras: Dict[str, Any] = {'time': time}

    # ------------------------------------------------------------------ API registration

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        """Expose `fn` to the sandbox under `name`."""
        if not callable(fn):
            raise TypeError(f"Sandbox API {name!r} must be callable, got {type(fn)}")
        self._api[name] = fn

    def register_many(self, mapping: Dict[str, Callable[..., Any]]) -> None:
        for name, fn in mapping.items():
            self.register(name, fn)

    def expose(self, name: str, value: Any) -> None:
        """Expose a non-callable (e.g. a constant dict) to the sandbox."""
        self._extras[name] = value

    # ------------------------------------------------------------------ execution

    def _build_globals(self) -> Dict[str, Any]:
        g: Dict[str, Any] = {'__builtins__': dict(_SAFE_BUILTINS)}
        g.update(self._extras)
        g.update(self._api)
        return g

    def run(self, code_str: str) -> SandboxResult:
        """
        Execute `code_str`. Returns a SandboxResult; never raises.

        The code runs synchronously in the calling thread. If primitives are
        blocking (they are, in Phase 1 onward), this call is long-running.
        """
        if not code_str or not code_str.strip():
            return SandboxResult(False, 0.0, error='empty code', traceback=None)

        safe_globals = self._build_globals()
        safe_locals: Dict[str, Any] = {}

        t0 = time.time()
        try:
            exec(code_str, safe_globals, safe_locals)  # noqa: S102 (intentional)
        except Exception as e:
            tb = traceback.format_exc()
            return SandboxResult(False, time.time() - t0, error=str(e), traceback=tb)

        return SandboxResult(True, time.time() - t0)

    # ------------------------------------------------------------------ introspection

    @property
    def registered_names(self):
        return sorted(self._api.keys())
