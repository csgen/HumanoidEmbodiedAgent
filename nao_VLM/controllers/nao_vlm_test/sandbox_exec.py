"""
Sandbox executor — runs VLM-generated primitive compositions in a restricted
Python environment, using bound methods from NaoVlmAPI as the whitelisted API.

Phase 0: restrict builtins, wire bound methods, catch exceptions.
Phase 4 (later): add joint-limit pre-check and static AST validation.
"""
from __future__ import annotations

import ast
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


@dataclass
class ValidationResult:
    ok: bool
    error: Optional[str] = None


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
        self._joint_limits: Dict[str, tuple] = {}
        self._metrics_recorder = None

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

    def set_joint_limits(self, joint_limits: Dict[str, tuple]) -> None:
        self._joint_limits = dict(joint_limits or {})

    def set_metrics_recorder(self, recorder) -> None:
        self._metrics_recorder = recorder

    def _record_event(self, event: str, *, error: Optional[str] = None,
                      elapsed_seconds: Optional[float] = None,
                      code: str = '') -> None:
        if self._metrics_recorder is None:
            return
        try:
            self._metrics_recorder.record_sandbox_event(
                event,
                error=error,
                elapsed_seconds=elapsed_seconds,
                code=code,
            )
        except Exception as exc:
            print(f'[metrics] sandbox event logging failed: {exc}')

    def _literal(self, node):
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    def _keyword_literal(self, node: ast.Call, name: str):
        for kw in node.keywords:
            if kw.arg == name:
                return self._literal(kw.value)
        return None

    def _normalize_arm_side(self, side: Any):
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

    def validate(self, code_str: str) -> ValidationResult:
        try:
            tree = ast.parse(code_str)
        except SyntaxError as exc:
            return ValidationResult(False, error=f'syntax_error: {exc}')

        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            value = tree.body[0].value
            if isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Constant)):
                return ValidationResult(False, error='non_executable_python_block')

        oscillation_calls = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            fn_name = None
            if isinstance(node.func, ast.Name):
                fn_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr

            if fn_name and fn_name not in self._api and fn_name not in {'sleep'}:
                return ValidationResult(False, error=f'unknown_primitive: {fn_name}')

            if fn_name == 'move_joint':
                allowed_keywords = {'name', 'angle', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_joint_kwargs: {unknown_keywords[0]}')
                joint_name = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'name')
                angle = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'angle')
                duration = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'duration')
                if angle is not None and not isinstance(angle, (int, float)):
                    return ValidationResult(False, error=f'non_scalar_joint_angle: {joint_name}')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_joint')

            if fn_name == 'move_joints':
                allowed_keywords = {'joint_angles', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_joints_kwargs: {unknown_keywords[0]}')
                mapping = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'joint_angles')
                duration = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'duration')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_joints')
                if isinstance(mapping, dict):
                    for joint_name, angle in mapping.items():
                        if angle is not None and not isinstance(angle, (int, float)):
                            return ValidationResult(False, error=f'non_scalar_joint_angle: {joint_name}')

            if fn_name == 'move_head':
                allowed_keywords = {'yaw', 'pitch', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_head_kwargs: {unknown_keywords[0]}')
                yaw = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'yaw')
                pitch = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'pitch')
                duration = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'duration')
                if yaw is not None and not isinstance(yaw, (int, float)):
                    return ValidationResult(False, error='non_scalar_yaw: move_head')
                if pitch is not None and not isinstance(pitch, (int, float)):
                    return ValidationResult(False, error='non_scalar_pitch: move_head')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_head')

            if fn_name == 'oscillate_joint':
                oscillation_calls += 1
                allowed_keywords = {'name', 'center', 'amplitude', 'frequency', 'duration', 'decay'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_oscillate_joint_kwargs: {unknown_keywords[0]}')
                joint_name = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'name')
                center = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'center')
                amplitude = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'amplitude')
                frequency = self._literal(node.args[3]) if len(node.args) >= 4 else self._keyword_literal(node, 'frequency')
                duration = self._literal(node.args[4]) if len(node.args) >= 5 else self._keyword_literal(node, 'duration')

            if fn_name == 'move_arm_ik':
                allowed_keywords = {'side', 'xyz', 'duration', 'orientation'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_arm_ik_kwargs: {unknown_keywords[0]}')
                side = self._normalize_arm_side(self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'side'))
                xyz = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'xyz')
                duration = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'duration')
                if isinstance(side, str) and side not in {'left', 'right'}:
                    return ValidationResult(False, error=f'invalid_arm_side: {side}')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_arm_ik')

            if fn_name == 'set_hand':
                allowed_keywords = {'side', 'openness', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_set_hand_kwargs: {unknown_keywords[0]}')
                side = self._normalize_arm_side(self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'side'))
                openness = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'openness')
                duration = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'duration')
                if isinstance(side, str) and side not in {'left', 'right'}:
                    return ValidationResult(False, error=f'invalid_hand_side: {side}')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: set_hand')

            if fn_name in {'hold', 'idle'}:
                allowed_keywords = {'duration'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_{fn_name}_kwargs: {unknown_keywords[0]}')
                duration = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'duration')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error=f'non_scalar_duration: {fn_name}')

        return ValidationResult(True)

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

        validation = self.validate(code_str)
        if not validation.ok:
            self._record_event('validate_fail', error=validation.error, code=code_str)
            return SandboxResult(False, 0.0, error=validation.error, traceback=None)
        self._record_event('validate_pass', code=code_str)

        safe_globals = self._build_globals()
        safe_locals: Dict[str, Any] = {}

        t0 = time.time()
        try:
            exec(code_str, safe_globals, safe_locals)  # noqa: S102 (intentional)
        except Exception as e:
            tb = traceback.format_exc()
            elapsed = time.time() - t0
            self._record_event('exec_error', error=str(e), elapsed_seconds=elapsed, code=code_str)
            return SandboxResult(False, elapsed, error=str(e), traceback=tb)

        elapsed = time.time() - t0
        self._record_event('exec_ok', elapsed_seconds=elapsed, code=code_str)
        return SandboxResult(True, elapsed)

    # ------------------------------------------------------------------ introspection

    @property
    def registered_names(self):
        return sorted(self._api.keys())
