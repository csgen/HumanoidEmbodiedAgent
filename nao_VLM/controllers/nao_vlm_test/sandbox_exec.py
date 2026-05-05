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
        self._walking_forbidden = True
        self._forbidden_joints = {
            'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
            'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
        }

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

            if self._walking_forbidden and fn_name == 'navigate_to':
                return ValidationResult(False, error='navigate_to_forbidden_for_demo')

            if fn_name == 'move_joint':
                allowed_keywords = {'name', 'angle', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_joint_kwargs: {unknown_keywords[0]}')
                joint_name = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'name')
                angle = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'angle')
                duration = self._literal(node.args[2]) if len(node.args) >= 3 else self._keyword_literal(node, 'duration')
                if isinstance(joint_name, str) and joint_name in self._forbidden_joints:
                    return ValidationResult(False, error=f'lower_body_joint_forbidden: {joint_name}')
                if isinstance(joint_name, str) and self._joint_limits and joint_name not in self._joint_limits:
                    return ValidationResult(False, error=f'unknown_joint: {joint_name}')
                if angle is not None and not isinstance(angle, (int, float)):
                    return ValidationResult(False, error=f'non_scalar_joint_angle: {joint_name}')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_joint')
                if isinstance(duration, (int, float)) and not (0.05 <= float(duration) <= 3.0):
                    return ValidationResult(False, error=f'invalid_duration: move_joint={duration}')
                if isinstance(joint_name, str) and isinstance(angle, (int, float)):
                    limits = self._joint_limits.get(joint_name)
                    if limits is not None and limits[0] != limits[1]:
                        if not (limits[0] <= float(angle) <= limits[1]):
                            return ValidationResult(False, error=f'joint_limit_violation: {joint_name}={angle}')

            if fn_name == 'move_joints':
                allowed_keywords = {'joint_angles', 'duration', 'trajectory'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_move_joints_kwargs: {unknown_keywords[0]}')
                mapping = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'joint_angles')
                duration = self._literal(node.args[1]) if len(node.args) >= 2 else self._keyword_literal(node, 'duration')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: move_joints')
                if isinstance(duration, (int, float)) and not (0.05 <= float(duration) <= 3.0):
                    return ValidationResult(False, error=f'invalid_duration: move_joints={duration}')
                if isinstance(mapping, dict):
                    for joint_name, angle in mapping.items():
                        if isinstance(joint_name, str) and joint_name in self._forbidden_joints:
                            return ValidationResult(False, error=f'lower_body_joint_forbidden: {joint_name}')
                        if isinstance(joint_name, str) and self._joint_limits and joint_name not in self._joint_limits:
                            return ValidationResult(False, error=f'unknown_joint: {joint_name}')
                        if angle is not None and not isinstance(angle, (int, float)):
                            return ValidationResult(False, error=f'non_scalar_joint_angle: {joint_name}')
                        limits = self._joint_limits.get(joint_name)
                        if limits is not None and isinstance(angle, (int, float)) and limits[0] != limits[1]:
                            if not (limits[0] <= float(angle) <= limits[1]):
                                return ValidationResult(False, error=f'joint_limit_violation: {joint_name}={angle}')

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
                if isinstance(joint_name, str) and joint_name in self._forbidden_joints:
                    return ValidationResult(False, error=f'lower_body_joint_forbidden: {joint_name}')
                if isinstance(joint_name, str) and self._joint_limits and joint_name not in self._joint_limits:
                    return ValidationResult(False, error=f'unknown_joint: {joint_name}')
                limits = self._joint_limits.get(joint_name) if isinstance(joint_name, str) else None
                if limits is not None and isinstance(center, (int, float)) and isinstance(amplitude, (int, float)):
                    lo = float(center) - abs(float(amplitude))
                    hi = float(center) + abs(float(amplitude))
                    if not (limits[0] <= lo <= limits[1] and limits[0] <= hi <= limits[1]):
                        return ValidationResult(False, error=f'oscillation_limit_violation: {joint_name}')
                if isinstance(amplitude, (int, float)) and abs(float(amplitude)) > 0.7:
                    return ValidationResult(False, error=f'oscillation_amplitude_too_large: {joint_name}')
                if isinstance(frequency, (int, float)) and float(frequency) > 2.5:
                    return ValidationResult(False, error=f'oscillation_frequency_too_high: {joint_name}')
                if isinstance(duration, (int, float)) and float(duration) > 3.0:
                    return ValidationResult(False, error=f'oscillation_duration_too_long: {joint_name}')

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
                if isinstance(duration, (int, float)) and not (0.05 <= float(duration) <= 3.0):
                    return ValidationResult(False, error=f'invalid_duration: move_arm_ik={duration}')
                if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
                    try:
                        radius = sum(float(v) * float(v) for v in xyz) ** 0.5
                    except Exception:
                        radius = 0.0
                    if radius < 0.05:
                        return ValidationResult(False, error=f'ik_target_too_small: {xyz}')
                    if radius > 0.6:
                        return ValidationResult(False, error=f'ik_target_too_far: {xyz}')

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
                if isinstance(openness, (int, float)) and not (0.0 <= float(openness) <= 1.0):
                    return ValidationResult(False, error=f'hand_openness_out_of_range: {openness}')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error='non_scalar_duration: set_hand')
                if isinstance(duration, (int, float)) and not (0.05 <= float(duration) <= 3.0):
                    return ValidationResult(False, error=f'invalid_duration: set_hand={duration}')

            if fn_name in {'hold', 'idle'}:
                allowed_keywords = {'duration'}
                unknown_keywords = [kw.arg for kw in node.keywords if kw.arg and kw.arg not in allowed_keywords]
                if unknown_keywords:
                    return ValidationResult(False, error=f'unknown_{fn_name}_kwargs: {unknown_keywords[0]}')
                duration = self._literal(node.args[0]) if len(node.args) >= 1 else self._keyword_literal(node, 'duration')
                if duration is not None and not isinstance(duration, (int, float)):
                    return ValidationResult(False, error=f'non_scalar_duration: {fn_name}')
                if isinstance(duration, (int, float)) and not (0.05 <= float(duration) <= 3.0):
                    return ValidationResult(False, error=f'invalid_duration: {fn_name}={duration}')

        if oscillation_calls >= 4:
            return ValidationResult(False, error='too_many_oscillations')

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
            return SandboxResult(False, 0.0, error=validation.error, traceback=None)

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
