"""
VLM client — supports both remote OpenAI-compatible backends and a local
multi-image backend.

Both backends follow the same contract:
  - input: chronological video frames
  - output: semantic JSON + Python primitive code

The local backend does NOT route through hard-coded gesture labels or a fixed
response library. It asks the local VLM to directly generate the robot control
program from the observed video frames.
"""
from __future__ import annotations

import ast
import base64
import difflib
import io
import json
import math
import re
import textwrap
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import config
from sandbox_exec import SandboxExecutor


_BASE_SYSTEM_PROMPT = """\
You are the embodied-intelligence core of a NAO H25 humanoid robot running
inside a Webots physics simulator.

You will receive a SHORT CHRONOLOGICAL SEQUENCE of {frame_count} frames captured
over approximately {window_seconds:.1f} seconds from a camera observing a
human. Your job is to:

  1. Interpret the human's motion AND intent across the full sequence
     (not just the final frame).
  2. Emit a structured semantic context object.
  3. Emit Python control code that composes low-level motion primitives
     to produce a socially-appropriate physical response.

# Important framing
- DO NOT identify, recognise, or describe any specific individual. Treat the
  human as anonymous. Refer to them only as "the human" or "they".
- Focus EXCLUSIVELY on motion patterns, body language, gestures, and
  approximate distance — never on identity, demographic attributes, or
  appearance details.
- The camera framing is a robot-eye-view of a human interacting with the
  robot. It is NOT a selfie / phone-recording / video-call context, even if
  the framing looks like one. A close, front-facing view means the human is
  near the robot — choose social distance accordingly.

# Motion grammar (physical primitives)
You construct behavior, you do NOT select named actions. There is NO "wave()"
or "greet()" function. You must COMPOSE motion from these primitives:

    move_joint(name: str, angle: float, duration: float,
               trajectory: str = 'cubic')
        # Smoothly move a single joint to `angle` radians over `duration` s.
        # trajectory shapes:
        #   'cubic'    - smoothstep, default
        #   'min_jerk' - quintic, more elegant / natural
        #   'linear'   - more mechanical
        #   'cosine'   - soft ease-in-out

    move_joints(joint_angles: dict, duration: float,
                trajectory: str = 'cubic')
        # Move multiple joints in parallel.

    move_arm_ik(side: str, xyz: list, duration: float)

    set_hand(side: str, openness: float, duration: float,
             trajectory: str = 'cubic')

    oscillate_joint(name: str, center: float, amplitude: float,
                    frequency: float, duration: float, decay: float = 0.0)

    hold(duration: float)

    idle(duration: float)

# Demo constraints
- For this project stage, DO NOT walk and DO NOT use locomotion.
- Prefer upper-body behavior only: head, shoulders, elbows, wrists, hands.
- Avoid lower-body joints unless absolutely necessary; in this stage they are not needed.
- Keep motions short, smooth, physically plausible, and pet-like.
- React naturally rather than mimic exactly; the response should feel socially appropriate.
- Avoid exaggerated repeated oscillations unless the video strongly supports them.
- Prefer nonverbal motion only in this demo stage.

# NAO coordinate convention
- Torso-centred, right-hand frame. Units are METRES.
- x forward, y LEFT is positive, z up from torso.
- Right arm targets usually use negative y. Left arm targets usually use positive y.
- Neutral right hand ≈ [0.02, -0.10, -0.20]. Neutral left hand ≈ [0.02, +0.10, -0.20].
- A natural raised right hand is often around [0.15, -0.15, 0.10].
- A natural raised left hand is often around [0.15, +0.15, 0.10].

# Joint reference (radian limits)
{joint_limit_block}

# Diversity and grounding requirements
- Different input clips should generally produce DIFFERENT control code.
- Reusing the same wrist-only oscillation for unrelated clips is incorrect.
- Base side choice, dominant joints, timing, and stillness-vs-motion decisions on
  the actual frame sequence.
- Prefer coordinated head + arm + wrist composition over a single isolated joint
  when the frames support a richer upper-body response.
- Use very small motion when the clip evidence is weak; use larger but still smooth
  motion only when the sequence clearly supports it.
- If motion is concentrated on one side of the image, prefer one dominant arm
  or a head-led response rather than moving both arms.
- For low-motion or static clips, do not output a zero-angle no-op posture;
  prefer a small but readable head attention response plus a brief hold or idle.

# Composition guidance
- Prefer short sequences of 2-5 primitive calls.
- Use `move_joints(...)` when a coordinated upper-body posture is needed.
- Use `set_hand(...)` for subtle open/close emphasis instead of large arm swings when appropriate.
- Use `hold(...)` when the response should pause, freeze, or attend.
- Use `oscillate_joint(...)` only when the observed motion itself appears repeated
  across multiple frames.
- Return smoothly toward a calm upper-body posture after a dynamic motion.
- Every primitive call should have a visible purpose; avoid repeating the exact
  same call twice in a row.
- Avoid mirrored both-arm responses unless the frames clearly show a bilateral
  interaction. Most natural pet-like responses use head plus one dominant arm.
- If you use `move_joints(...)`, prefer coordinated shoulder/elbow/head changes
  that form a readable pose rather than tiny no-op joint changes.
- A response that only returns one elbow to its mechanical limit is usually too
  generic unless the frames strongly justify it.
- Avoid driving shoulder pitch, shoulder roll, elbow roll, or head pitch to the
  exact joint extremes; leave visible comfort margin.
- For pet-like responses, prefer one readable gesture with a short settle rather
  than several large disjoint posture changes.

# Output format — MANDATORY
First a JSON block, THEN a Python block, and nothing else:

```json
{{
  "intent": "<short free-form description of what the human is doing>",
  "social_distance": "close" | "medium" | "far",
  "affect": "<happy|neutral|hostile|sad|curious|etc.>",
  "confidence": <float 0.0-1.0>,
  "motion_dynamics": "oscillatory" | "approaching" | "retreating" |
                     "raising" | "lowering" | "static"
}}
```

```python
# your primitive composition here
```

Do NOT output any prose outside the two fenced blocks.
"""

_USER_PROMPT = (
    "Analyse the motion across the frames below and generate the required "
    "JSON context and Python primitive composition."
)

_SCENARIO_HINT_TEMPLATE = """\

# Scenario hint
The current run is a single-turn demo. Focus on the most salient human action
in the clip and produce exactly one short response sequence.

Extra task hint from operator:
{scenario_hint}
"""

_LOCAL_USER_PROMPT = (
    "Analyse the chronological frames below and directly output the required "
    "JSON context plus Python robot-control program. Do not classify into any "
    "intermediate human-gesture labels. Do not use any named canned action. "
    "Directly compose the robot behaviour from the allowed low-level motion "
    "primitives and joint movements. Use the exact primitive names provided in "
    "the system instructions. Do not prefix calls with objects like `nao.` or "
    "`robot.`. Output only the required fenced JSON block and fenced Python "
    "block. Do not use lower-body motion or walking in this demo stage. Prefer "
    "head, arm, wrist, and hand motion only. Do not default to the same motion "
    "pattern for every clip. Infer laterality, tempo, activity level, and "
    "dominant direction from the frames, then let those observations change the "
    "generated control program. If the visible motion is lateralized, prefer one "
    "dominant arm or a head-led response, and keep the opposite arm mostly quiet "
    "unless the frames clearly justify bilateral motion."
)

_LOCAL_REPAIR_PROMPT_TEMPLATE = """The previous robot-control program was not acceptable.

Use the SAME video frames and produce a corrected result.

Rules:
- Keep the robot fully controlled by VLM-generated code.
- Do not use any intermediate gesture labels.
- Do not use canned named actions.
- Use only these primitives: move_joint, move_joints, move_arm_ik, set_hand, oscillate_joint, hold, idle.
- Do NOT use lower-body joints.
- Prefer smooth upper-body motion only: head, shoulders, elbows, wrists, hands.
- Keep the motion short, natural, pet-like, and physically plausible.
- Use exact primitive names with no object prefixes.

Previous semantic JSON:
{semantic_json}

Previous Python code:
```python
{python_code}
```

Problem to fix:
{error_text}

Return ONLY a corrected JSON block and corrected Python block.
"""

_LOCAL_REFINEMENT_PROMPT_TEMPLATE = """The previous candidate is too generic or not well grounded in the specific clip.

Use the SAME video frames and produce a BETTER grounded result.

Rules:
- Keep the robot fully controlled by VLM-generated code.
- Do not use intermediate gesture labels or canned named actions.
- Use only these primitives: move_joint, move_joints, move_arm_ik, set_hand, oscillate_joint, hold, idle.
- Do NOT use lower-body joints.
- The new code must differ materially from the previous code if the previous code was too generic.
- Avoid a wrist-only oscillation unless the video truly supports it.
- Prefer a short coordinated combination such as head+arm, head+hand, or arm+hand over a single-joint response when the frames support it.
- If the clip evidence is weak, stay small and calm rather than inventing a dramatic response.
- For low-motion or static clips, avoid zero-angle no-op poses; use a small readable head attention cue instead.
- Do not repeat the exact same primitive call twice in a row.
- Avoid symmetric both-arm poses unless the video clearly supports bilateral motion.
- If the visible motion is lateralized, keep the non-dominant arm mostly quiet instead of inventing a support pose on the other side.
- Avoid near-zero or no-op joint targets that create motion without readable intent.
- Avoid joint-extreme postures that look tense or mechanically forced.
- For low-motion or static clips, prefer head-only attention cues; avoid inventing arm poses unless the frames clearly justify them.
- End the response with a short `hold(...)` or `idle(...)` after the main motion so the behavior settles naturally.

Previous semantic JSON:
{semantic_json}

Previous Python code:
```python
{python_code}
```

Reason for refinement:
{error_text}

Return ONLY a corrected JSON block and corrected Python block.
"""

_LOCAL_SELECTION_PROMPT_TEMPLATE = """You must choose the BEST candidate robot response for the SAME video clip.

Selection rules:
- Choose the candidate that is most grounded in the actual video frames.
- Prefer the candidate that feels most natural, short, smooth, and pet-like.
- Prefer upper-body only behavior.
- Reject candidates that look generic, repetitive, or unrelated to the observed motion.
- Reject candidates that use walking, lower-body joints, or obviously unsafe motion.
- If more than one candidate is plausible, choose the one with the clearest social appropriateness.
- Reject candidates that repeat the exact same call twice in a row.
- Reject candidates that move both arms symmetrically without clear evidence.
- Reject candidates whose motion is technically valid but visually unreadable or no-op.
- If the motion evidence is laterally asymmetric, prefer candidates with one
  dominant arm instead of bilateral responses.
- Reject candidates that add non-dominant-arm support poses without clear evidence.

Cheap motion summary from the clip:
{motion_summary}

Candidates:
{candidate_blocks}

Return ONLY the chosen candidate number in this exact form:

```text
CHOSEN_CANDIDATE: <integer>
```

Do not explain your choice. Do not output JSON or Python here.
"""

_LOCAL_MODEL_LOCK = Lock()
_LOCAL_MODEL_ID: Optional[str] = None
_LOCAL_PROCESSOR = None
_LOCAL_MODEL = None
_LOCAL_MODEL_KIND: Optional[str] = None


def build_system_prompt(joint_limits: Dict[str, Tuple[float, float]]) -> str:
    rows = []
    for name in sorted(joint_limits):
        lo, hi = joint_limits[name]
        if lo == hi:
            continue
        rows.append(f"  - {name}: [{lo:+.3f}, {hi:+.3f}]")
    block = "\n".join(rows) if rows else "  (no joint limits provided)"

    prompt = _BASE_SYSTEM_PROMPT.format(
        frame_count=config.VLM_FRAME_COUNT,
        window_seconds=config.VLM_WINDOW_SECONDS,
        joint_limit_block=block,
    )
    if config.VLM_SCENARIO_HINT:
        prompt += _SCENARIO_HINT_TEMPLATE.format(
            scenario_hint=config.VLM_SCENARIO_HINT,
        )
    return prompt


@dataclass
class VLMResponse:
    semantic_context: Dict[str, Any]
    python_code: str
    raw_text: str
    elapsed_seconds: float
    ok: bool
    error: Optional[str] = None


def _decode_frames_to_pil(frames_b64: Sequence[str]):
    from PIL import Image

    images = []
    for img_b64 in frames_b64:
        payload = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(payload)).convert('RGB')
        images.append(image)
    return images


def _infer_visual_motion_summary(frames_b64: Sequence[str]) -> Dict[str, Any]:
    if len(frames_b64) < 2:
        return {
            'frame_count': len(frames_b64),
            'motion_energy': 0.0,
            'dominant_axis': 'static',
            'lateral_bias': 'center',
            'vertical_bias': 'center',
            'activity_level': 'low',
        }

    frames = []
    for img_b64 in frames_b64:
        payload = base64.b64decode(img_b64)
        arr = np.frombuffer(payload, dtype=np.uint8)
        try:
            import cv2
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = None
        if img is not None:
            frames.append(img.astype(np.float32) / 255.0)

    if len(frames) < 2:
        return {
            'frame_count': len(frames_b64),
            'motion_energy': 0.0,
            'dominant_axis': 'static',
            'lateral_bias': 'center',
            'vertical_bias': 'center',
            'activity_level': 'low',
        }

    total_energy = 0.0
    centroid_dx = 0.0
    centroid_dy = 0.0
    centroid_samples = 0
    left_energy_total = 0.0
    right_energy_total = 0.0

    prev_cx = None
    prev_cy = None
    for prev, cur in zip(frames[:-1], frames[1:]):
        diff = np.abs(cur - prev)
        energy = float(diff.mean())
        total_energy += energy

        h, w = diff.shape[:2]
        half_w = max(1, w // 2)
        left_energy_total += float(diff[:, :half_w].mean())
        right_energy_total += float(diff[:, half_w:].mean())
        ys, xs = np.mgrid[0:h, 0:w]
        weight = diff + 1e-6
        mass = float(weight.sum())
        cx = float((xs * weight).sum() / mass) / max(1.0, float(w - 1))
        cy = float((ys * weight).sum() / mass) / max(1.0, float(h - 1))

        if prev_cx is not None and prev_cy is not None:
            centroid_dx += cx - prev_cx
            centroid_dy += cy - prev_cy
            centroid_samples += 1
        prev_cx, prev_cy = cx, cy

    mean_energy = total_energy / max(1, len(frames) - 1)
    mean_dx = centroid_dx / max(1, centroid_samples)
    mean_dy = centroid_dy / max(1, centroid_samples)

    abs_dx = abs(mean_dx)
    abs_dy = abs(mean_dy)
    if mean_energy < 0.015:
        dominant_axis = 'static'
    elif abs_dx > abs_dy * 1.25:
        dominant_axis = 'horizontal'
    elif abs_dy > abs_dx * 1.25:
        dominant_axis = 'vertical'
    else:
        dominant_axis = 'mixed'

    if mean_dx > 0.02:
        lateral_bias = 'rightward_in_image'
    elif mean_dx < -0.02:
        lateral_bias = 'leftward_in_image'
    else:
        lateral_bias = 'center'

    if mean_dy > 0.02:
        vertical_bias = 'downward_in_image'
    elif mean_dy < -0.02:
        vertical_bias = 'upward_in_image'
    else:
        vertical_bias = 'center'

    if mean_energy < 0.02:
        activity = 'low'
    elif mean_energy < 0.05:
        activity = 'medium'
    else:
        activity = 'high'

    left_energy = left_energy_total / max(1, len(frames) - 1)
    right_energy = right_energy_total / max(1, len(frames) - 1)
    if left_energy > right_energy * 1.18:
        active_side = 'left_image'
    elif right_energy > left_energy * 1.18:
        active_side = 'right_image'
    else:
        active_side = 'balanced'

    return {
        'frame_count': len(frames_b64),
        'motion_energy': round(mean_energy, 4),
        'dominant_axis': dominant_axis,
        'lateral_bias': lateral_bias,
        'vertical_bias': vertical_bias,
        'activity_level': activity,
        'active_side': active_side,
        'left_energy': round(left_energy, 4),
        'right_energy': round(right_energy, 4),
    }


def _build_local_user_prompt(frames_b64: Sequence[str]) -> str:
    summary = _infer_visual_motion_summary(frames_b64)
    return (
        _LOCAL_USER_PROMPT
        + "\n\n# Cheap motion summary extracted from the frame sequence\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\nUse this only as weak grounding. The frames remain the main evidence."
    )


def _parse_code_calls(code: str) -> List[Tuple[str, int]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    calls: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.append((node.func.id, getattr(node, 'lineno', 0)))
        elif isinstance(node.func, ast.Attribute):
            calls.append((node.func.attr, getattr(node, 'lineno', 0)))
    return calls


def _normalize_arm_side_token(side: Any) -> Optional[str]:
    if not isinstance(side, str):
        return None
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
    return alias_map.get(normalized)


def _build_static_validator(joint_limits: Dict[str, Tuple[float, float]]) -> SandboxExecutor:
    executor = SandboxExecutor()

    def move_joint(name: str, angle: float, duration: float, trajectory: str = 'cubic'):
        return None

    def move_joints(joint_angles: dict, duration: float, trajectory: str = 'cubic'):
        return None

    def move_arm_ik(side: str, xyz, duration: float, orientation=None):
        return None

    def move_head(yaw: float, pitch: float, duration: float = 0.2, trajectory: str = 'min_jerk'):
        return None

    def set_hand(side: str, openness: float, duration: float, trajectory: str = 'cubic'):
        return None

    def oscillate_joint(name: str, center: float, amplitude: float, frequency: float, duration: float, decay: float = 0.0):
        return None

    def hold(duration: float):
        return None

    def idle(duration: float):
        return None

    executor.register_many({
        'move_joint': move_joint,
        'move_joints': move_joints,
        'move_arm_ik': move_arm_ik,
        'move_head': move_head,
        'set_hand': set_hand,
        'oscillate_joint': oscillate_joint,
        'hold': hold,
        'idle': idle,
    })
    executor.set_joint_limits(joint_limits)
    return executor


def _runtime_signature_ok(executor: SandboxExecutor, code: str) -> bool:
    if not code or not code.strip():
        return False
    result = executor.run(code)
    return bool(result.ok)


def _candidate_has_minimal_structure(semantic: Dict[str, Any], code: str) -> bool:
    if not semantic or not code:
        return False
    calls = _parse_code_calls(code)
    return bool(calls)


def _non_comment_code_lines(code: str) -> List[str]:
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith('#')
    ]


def _literal_or_none(node: Optional[ast.AST]):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _parse_top_level_calls(code: str) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    calls: List[Dict[str, Any]] = []
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        call = stmt.value
        if isinstance(call.func, ast.Name):
            fn_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            fn_name = call.func.attr
        else:
            continue
        calls.append({
            'name': fn_name,
            'args': [_literal_or_none(arg) for arg in call.args],
            'kwargs': {kw.arg: _literal_or_none(kw.value) for kw in call.keywords if kw.arg},
        })
    return calls


def _has_consecutive_duplicate_calls(code: str) -> bool:
    lines = _non_comment_code_lines(code)
    return any(prev == cur for prev, cur in zip(lines, lines[1:]))


def _call_duration(call: Dict[str, Any]) -> Optional[float]:
    if not isinstance(call, dict):
        return None
    args = call.get('args') or []
    kwargs = call.get('kwargs') or {}
    fn_name = call.get('name')
    duration = None
    if fn_name in {'move_joint', 'move_joints', 'move_arm_ik', 'set_hand'}:
        if len(args) >= 3 and isinstance(args[2], (int, float)):
            duration = float(args[2])
        elif isinstance(kwargs.get('duration'), (int, float)):
            duration = float(kwargs.get('duration'))
        elif fn_name == 'move_joints' and len(args) >= 2 and isinstance(args[1], (int, float)):
            duration = float(args[1])
        elif fn_name == 'set_hand' and len(args) >= 2 and isinstance(args[1], (int, float)):
            duration = float(args[1])
    elif fn_name in {'hold', 'idle'}:
        if args and isinstance(args[0], (int, float)):
            duration = float(args[0])
        elif isinstance(kwargs.get('duration'), (int, float)):
            duration = float(kwargs.get('duration'))
    elif fn_name == 'oscillate_joint':
        if len(args) >= 5 and isinstance(args[4], (int, float)):
            duration = float(args[4])
        elif isinstance(kwargs.get('duration'), (int, float)):
            duration = float(kwargs.get('duration'))
    return duration


def _has_too_many_micro_steps(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    micro = 0
    for call in calls:
        duration = _call_duration(call)
        if duration is not None and duration <= 0.15:
            micro += 1
    return micro >= 3


def _has_static_clip_with_arm_motion(summary: Dict[str, Any], code: str) -> bool:
    activity = str((summary or {}).get('activity_level') or 'low').lower()
    dominant_axis = str((summary or {}).get('dominant_axis') or 'static').lower()
    if activity != 'low':
        return False
    calls = _parse_top_level_calls(code)
    for call in calls:
        fn_name = call.get('name')
        if fn_name in {'move_arm_ik', 'set_hand', 'oscillate_joint'}:
            return True
        if fn_name not in {'move_joint', 'move_joints'}:
            continue
        joint_names: List[str] = []
        if fn_name == 'move_joint':
            joint_name = call['args'][0] if call['args'] else call['kwargs'].get('name')
            if isinstance(joint_name, str):
                joint_names.append(joint_name)
        else:
            joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
            if isinstance(joint_angles, dict):
                joint_names.extend(name for name in joint_angles if isinstance(name, str))
        if any((name.startswith('L') or name.startswith('R')) and not name.startswith('Head') for name in joint_names):
            return True
    return False


def _has_no_settle_after_dynamic_motion(summary: Dict[str, Any], code: str) -> bool:
    activity = str((summary or {}).get('activity_level') or 'low').lower()
    if activity not in {'medium', 'high'}:
        return False
    calls = _parse_top_level_calls(code)
    if len(calls) < 2:
        return False
    tail = calls[-1].get('name')
    if tail in {'hold', 'idle'}:
        return False
    return any(call.get('name') in {'move_arm_ik', 'move_joint', 'move_joints', 'set_hand'} for call in calls)


def _is_mirrored_arm_target(xyz_a: Any, xyz_b: Any) -> bool:
    if not isinstance(xyz_a, (list, tuple)) or not isinstance(xyz_b, (list, tuple)):
        return False
    if len(xyz_a) != 3 or len(xyz_b) != 3:
        return False
    try:
        ax, ay, az = [float(v) for v in xyz_a]
        bx, by, bz = [float(v) for v in xyz_b]
    except Exception:
        return False
    return (
        abs(ax - bx) <= 0.06
        and abs(az - bz) <= 0.06
        and abs(ay + by) <= 0.08
        and abs(abs(ay) - abs(by)) <= 0.06
    )


def _has_symmetric_dual_arm_ik(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    move_arm_calls = [call for call in calls if call.get('name') == 'move_arm_ik']
    if len(move_arm_calls) < 2:
        return False

    for first, second in zip(move_arm_calls, move_arm_calls[1:]):
        side_a = first['args'][0] if len(first['args']) >= 1 else first['kwargs'].get('side')
        side_b = second['args'][0] if len(second['args']) >= 1 else second['kwargs'].get('side')
        xyz_a = first['args'][1] if len(first['args']) >= 2 else first['kwargs'].get('xyz')
        xyz_b = second['args'][1] if len(second['args']) >= 2 else second['kwargs'].get('xyz')
        dur_a = first['args'][2] if len(first['args']) >= 3 else first['kwargs'].get('duration')
        dur_b = second['args'][2] if len(second['args']) >= 3 else second['kwargs'].get('duration')

        if {side_a, side_b} != {'left', 'right'}:
            continue
        if not _is_mirrored_arm_target(xyz_a, xyz_b):
            continue
        if isinstance(dur_a, (int, float)) and isinstance(dur_b, (int, float)) and abs(float(dur_a) - float(dur_b)) > 0.25:
            continue
        return True
    return False


def _has_mirrored_dual_arm_posture(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    for call in calls:
        if call.get('name') != 'move_joints':
            continue
        joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
        if not isinstance(joint_angles, dict) or not joint_angles:
            continue
        joint_names = [name for name in joint_angles if isinstance(name, str)]
        if not joint_names or any(name.startswith('Head') for name in joint_names):
            continue

        left = {name[1:]: value for name, value in joint_angles.items() if isinstance(name, str) and name.startswith('L')}
        right = {name[1:]: value for name, value in joint_angles.items() if isinstance(name, str) and name.startswith('R')}
        shared = sorted(set(left) & set(right))
        if not shared:
            continue

        mirrored_pairs = 0
        for suffix in shared:
            lv = left[suffix]
            rv = right[suffix]
            if not isinstance(lv, (int, float)) or not isinstance(rv, (int, float)):
                continue
            if abs(abs(float(lv)) - abs(float(rv))) <= 0.12:
                mirrored_pairs += 1

        if mirrored_pairs >= 1 and len(joint_names) <= 4:
            return True
    return False


def _uses_only_distal_arm_joints(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    joint_names: List[str] = []
    for call in calls:
        if call.get('name') == 'move_joint':
            joint_name = call['args'][0] if call['args'] else call['kwargs'].get('name')
            if isinstance(joint_name, str):
                joint_names.append(joint_name)
        elif call.get('name') == 'move_joints':
            joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
            if isinstance(joint_angles, dict):
                joint_names.extend(name for name in joint_angles if isinstance(name, str))

    if not joint_names:
        return False
    allowed_suffixes = {'ElbowRoll', 'ElbowYaw', 'WristYaw'}
    for joint_name in joint_names:
        if joint_name.startswith('Head'):
            return False
        suffix = joint_name[1:] if joint_name[:1] in {'L', 'R'} else joint_name
        if suffix not in allowed_suffixes:
            return False
    return True


def _has_unreadably_small_posture(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    if not calls:
        return False
    saw_joint_posture = False
    for call in calls:
        fn_name = call.get('name')
        if fn_name in {'move_arm_ik', 'set_hand', 'oscillate_joint'}:
            return False
        if fn_name not in {'move_joint', 'move_joints', 'hold', 'idle'}:
            continue
        if fn_name == 'move_joint':
            angle = call['args'][1] if len(call.get('args') or []) >= 2 else call.get('kwargs', {}).get('angle')
            if not isinstance(angle, (int, float)) or abs(float(angle)) > 0.08:
                return False
            saw_joint_posture = True
        elif fn_name == 'move_joints':
            joint_angles = call['args'][0] if call.get('args') else call.get('kwargs', {}).get('joint_angles')
            if not isinstance(joint_angles, dict) or not joint_angles:
                return False
            numeric_values = [float(v) for v in joint_angles.values() if isinstance(v, (int, float))]
            if not numeric_values or any(abs(v) > 0.08 for v in numeric_values):
                return False
            saw_joint_posture = True
    return saw_joint_posture and len(_non_comment_code_lines(code)) <= 3


def _has_hand_without_postural_anchor(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    has_hand = False
    has_head = False
    has_shoulder_or_ik = False
    for call in calls:
        fn_name = call.get('name')
        if fn_name == 'move_arm_ik':
            has_shoulder_or_ik = True
            continue
        if fn_name == 'set_hand':
            has_hand = True
            continue
        if fn_name not in {'move_joint', 'move_joints', 'oscillate_joint'}:
            continue

        joint_names: List[str] = []
        if fn_name in {'move_joint', 'oscillate_joint'}:
            joint_name = call['args'][0] if call['args'] else call['kwargs'].get('name')
            if isinstance(joint_name, str):
                joint_names.append(joint_name)
        else:
            joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
            if isinstance(joint_angles, dict):
                joint_names.extend(name for name in joint_angles if isinstance(name, str))

        for joint_name in joint_names:
            if joint_name.startswith('Head'):
                has_head = True
            suffix = joint_name[1:] if joint_name[:1] in {'L', 'R'} else joint_name
            if suffix.startswith('Shoulder'):
                has_shoulder_or_ik = True

    return has_hand and not has_head and not has_shoulder_or_ik


def _has_readable_head_or_single_arm_focus(code: str) -> bool:
    calls = _parse_top_level_calls(code)
    for call in calls:
        if call.get('name') == 'move_arm_ik':
            return True
        if call.get('name') not in {'move_joint', 'move_joints'}:
            continue

        joint_names: List[str] = []
        if call.get('name') == 'move_joint':
            joint_name = call['args'][0] if call['args'] else call['kwargs'].get('name')
            if isinstance(joint_name, str):
                joint_names.append(joint_name)
        else:
            joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
            if isinstance(joint_angles, dict):
                joint_names.extend(name for name in joint_angles if isinstance(name, str))

        has_head = any(name.startswith('Head') for name in joint_names)
        left_count = sum(1 for name in joint_names if name.startswith('L'))
        right_count = sum(1 for name in joint_names if name.startswith('R'))
        if has_head:
            return True
        if left_count >= 2 and right_count == 0:
            return True
        if right_count >= 2 and left_count == 0:
            return True
    return False


def _upper_body_activity_stats(code: str) -> Dict[str, int]:
    calls = _parse_top_level_calls(code)
    stats = {
        'left_score': 0,
        'right_score': 0,
        'left_calls': 0,
        'right_calls': 0,
        'head_calls': 0,
    }
    for call in calls:
        fn_name = call.get('name')
        if fn_name == 'move_arm_ik':
            side = _normalize_arm_side_token(call['args'][0] if call['args'] else call['kwargs'].get('side'))
            if side == 'left':
                stats['left_score'] += 3
                stats['left_calls'] += 1
            elif side == 'right':
                stats['right_score'] += 3
                stats['right_calls'] += 1
            continue

        joint_names: List[str] = []
        if fn_name in {'move_joint', 'oscillate_joint'}:
            joint_name = call['args'][0] if call['args'] else call['kwargs'].get('name')
            if isinstance(joint_name, str):
                joint_names.append(joint_name)
        elif fn_name == 'move_joints':
            joint_angles = call['args'][0] if call['args'] else call['kwargs'].get('joint_angles')
            if isinstance(joint_angles, dict):
                joint_names.extend(name for name in joint_angles if isinstance(name, str))
        elif fn_name == 'set_hand':
            side = _normalize_arm_side_token(call['args'][0] if call['args'] else call['kwargs'].get('side'))
            if side == 'left':
                stats['left_score'] += 1
                stats['left_calls'] += 1
            elif side == 'right':
                stats['right_score'] += 1
                stats['right_calls'] += 1
            continue
        else:
            continue

        saw_left = False
        saw_right = False
        saw_head = False
        for joint_name in joint_names:
            if joint_name.startswith('Head'):
                saw_head = True
            elif joint_name.startswith('L'):
                saw_left = True
                stats['left_score'] += 1
            elif joint_name.startswith('R'):
                saw_right = True
                stats['right_score'] += 1
        if saw_left:
            stats['left_calls'] += 1
        if saw_right:
            stats['right_calls'] += 1
        if saw_head:
            stats['head_calls'] += 1
    return stats


def _has_fragmented_bilateral_upper_body_motion(code: str) -> bool:
    stats = _upper_body_activity_stats(code)
    if stats['head_calls'] > 0:
        return False
    if stats['left_calls'] == 0 or stats['right_calls'] == 0:
        return False
    if abs(stats['left_score'] - stats['right_score']) <= 2:
        return True
    if min(stats['left_calls'], stats['right_calls']) >= 1 and (stats['left_calls'] + stats['right_calls']) <= 3:
        return True
    return False


def _has_non_dominant_arm_noise(summary: Dict[str, Any], code: str) -> bool:
    stats = _upper_body_activity_stats(code)
    active_side = str((summary or {}).get('active_side') or 'balanced').lower()
    dominant_arm = _dominant_arm_side_in_code(code)
    if stats['head_calls'] > 0:
        return False
    if active_side not in {'left_image', 'right_image'}:
        return False
    if dominant_arm == 'balanced':
        return stats['left_calls'] >= 1 and stats['right_calls'] >= 1
    if dominant_arm == 'left':
        return stats['right_calls'] >= 1 and stats['right_score'] >= 2
    if dominant_arm == 'right':
        return stats['left_calls'] >= 1 and stats['left_score'] >= 2
    return False


def _dominant_arm_side_in_code(code: str) -> Optional[str]:
    stats = _upper_body_activity_stats(code)
    left_score = stats['left_score']
    right_score = stats['right_score']
    if left_score == 0 and right_score == 0:
        return None
    if left_score > right_score * 1.3:
        return 'left'
    if right_score > left_score * 1.3:
        return 'right'
    return 'balanced'


def _looks_too_generic_for_clip(summary: Dict[str, Any], semantic: Dict[str, Any], code: str) -> bool:
    if not code:
        return True
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith('#')]
    calls = _parse_code_calls(code)
    fn_names = [name for name, _ in calls]
    unique_fns = set(fn_names)
    activity = str((summary or {}).get('activity_level') or 'low').lower()
    active_side = str((summary or {}).get('active_side') or 'balanced').lower()
    dominant_arm = _dominant_arm_side_in_code(code)

    if len(lines) <= 2 and unique_fns == {'oscillate_joint'}:
        return True
    if len(lines) <= 2 and unique_fns <= {'move_joint', 'oscillate_joint', 'set_hand'}:
        return True
    if unique_fns == {'hold'}:
        return True
    if len(calls) <= 1:
        return True
    if _has_consecutive_duplicate_calls(code):
        return True
    if _has_symmetric_dual_arm_ik(code):
        return True
    if activity in {'low', 'medium'} and _has_mirrored_dual_arm_posture(code):
        return True
    if _has_fragmented_bilateral_upper_body_motion(code):
        return True
    if _has_too_many_micro_steps(code):
        return True
    if _has_unreadably_small_posture(code):
        return True
    if 'move_arm_ik' not in unique_fns and 'move_joints' not in unique_fns and 'set_hand' not in unique_fns and len(unique_fns) == 1:
        return True
    if _has_hand_without_postural_anchor(code):
        return True
    if _has_static_clip_with_arm_motion(summary, code):
        return True
    if activity in {'medium', 'high'} and 'move_arm_ik' not in unique_fns and 'move_joints' not in unique_fns:
        return True
    if activity in {'medium', 'high'} and not _has_readable_head_or_single_arm_focus(code):
        return True
    if active_side in {'left_image', 'right_image'} and dominant_arm == 'balanced':
        return True
    if active_side == 'left_image' and dominant_arm == 'right':
        return True
    if active_side == 'right_image' and dominant_arm == 'left':
        return True
    if _has_non_dominant_arm_noise(summary, code):
        return True
    if _has_no_settle_after_dynamic_motion(summary, code):
        return True
    return False


def _naturalness_penalty(summary: Dict[str, Any], code: str) -> float:
    if not code:
        return 1e6
    penalty = 0.0
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith('#')]
    calls = _parse_code_calls(code)
    fn_names = [name for name, _ in calls]
    unique_fns = set(fn_names)
    activity = str((summary or {}).get('activity_level') or 'low').lower()
    active_side = str((summary or {}).get('active_side') or 'balanced').lower()
    dominant_arm = _dominant_arm_side_in_code(code)

    if len(lines) > 8:
        penalty += 4.0
    if len(lines) <= 1:
        penalty += 5.0
    if _has_consecutive_duplicate_calls(code):
        penalty += 4.0
    if 'idle' in unique_fns and len(unique_fns) == 1:
        penalty += 4.0
    if 'hold' in unique_fns and len(unique_fns) == 1:
        penalty += 3.0

    oscillation_count = fn_names.count('oscillate_joint')
    move_arm_count = fn_names.count('move_arm_ik')
    move_joint_count = fn_names.count('move_joint') + fn_names.count('move_joints')
    hand_count = fn_names.count('set_hand')

    if oscillation_count >= 3:
        penalty += 2.5
    if oscillation_count >= 1 and move_arm_count == 0 and move_joint_count == 0:
        penalty += 3.0
    if move_arm_count >= 4:
        penalty += 2.0
    if _has_symmetric_dual_arm_ik(code):
        penalty += 4.0
    if _has_mirrored_dual_arm_posture(code):
        penalty += 3.0
    if _has_fragmented_bilateral_upper_body_motion(code):
        penalty += 4.0
    if _has_too_many_micro_steps(code):
        penalty += 3.5
    if _has_unreadably_small_posture(code):
        penalty += 4.0
    if _uses_only_distal_arm_joints(code):
        penalty += 2.5
    if _has_hand_without_postural_anchor(code):
        penalty += 3.0
    if _has_static_clip_with_arm_motion(summary, code):
        penalty += 4.0
    if activity in {'medium', 'high'} and not _has_readable_head_or_single_arm_focus(code):
        penalty += 2.5
    if active_side in {'left_image', 'right_image'} and dominant_arm == 'balanced':
        penalty += 3.0
    if active_side == 'left_image' and dominant_arm == 'right':
        penalty += 2.5
    if active_side == 'right_image' and dominant_arm == 'left':
        penalty += 2.5
    if _has_non_dominant_arm_noise(summary, code):
        penalty += 3.0
    if _has_no_settle_after_dynamic_motion(summary, code):
        penalty += 1.5
    if hand_count >= 4:
        penalty += 1.5

    if activity in {'medium', 'high'} and move_arm_count == 0 and move_joint_count == 0:
        penalty += 3.0
    if activity == 'low' and oscillation_count >= 2:
        penalty += 2.0

    if 'set_hand(' in code and 'move_arm_ik(' not in code and 'move_joints(' not in code:
        penalty += 2.5

    return penalty


def _candidate_quality(summary: Dict[str, Any], semantic: Dict[str, Any], code: str, validation_ok: bool) -> Tuple[float, int, float]:
    penalty = _naturalness_penalty(summary, code)
    generic_flag = 1 if _looks_too_generic_for_clip(summary, semantic, code) else 0
    validation_flag = 0 if validation_ok else 1
    return (penalty + generic_flag * 2.0 + validation_flag * 1000.0, generic_flag, penalty)


def _pick_best_generated_variant(
    processor,
    model,
    kind: str,
    images,
    system_prompt: str,
    user_prompt: str,
    joint_limits: Dict[str, Tuple[float, float]],
    validator: SandboxExecutor,
    summary: Dict[str, Any],
    attempts: int = 3,
) -> Tuple[Dict[str, Any], str, str, bool, Tuple[float, int, float]]:
    best_semantic: Dict[str, Any] = {}
    best_code = ''
    best_raw = ''
    best_valid = False
    best_quality: Tuple[float, int, float] = (1e9, 1, 1e9)

    for idx in range(max(1, attempts)):
        raw = _generate_with_prompt(
            processor,
            model,
            kind,
            images,
            system_prompt,
            user_prompt,
            do_sample=(idx > 0),
        )
        semantic, code = parse_vlm_output(raw)
        code = _sanitize_generated_code(_normalize_generated_code(code), joint_limits)
        validation = validator.validate(code)
        quality = _candidate_quality(summary, semantic, code, validation.ok)
        if _candidate_has_minimal_structure(semantic, code) and quality < best_quality:
            best_semantic, best_code, best_raw = semantic, code, raw
            best_valid = validation.ok
            best_quality = quality

    return best_semantic, best_code, best_raw, best_valid, best_quality


def _describe_code_quality_issue(summary: Dict[str, Any], semantic: Dict[str, Any], code: str) -> str:
    reasons: List[str] = []
    if _has_consecutive_duplicate_calls(code):
        reasons.append('the code repeats the exact same primitive call consecutively')
    if _has_symmetric_dual_arm_ik(code) or _has_mirrored_dual_arm_posture(code):
        reasons.append('the code relies on mirrored both-arm motion, which looks generic instead of clip-specific')
    if _has_fragmented_bilateral_upper_body_motion(code):
        reasons.append('the code splits attention across both arms without a clear head anchor or dominant side')
    if _has_too_many_micro_steps(code):
        reasons.append('the code uses too many tiny rapid steps, so the motion reads as twitchy instead of natural')
    if _has_unreadably_small_posture(code):
        reasons.append('the code is technically valid but the posture change is so small that it reads like a no-op')
    if _has_non_dominant_arm_noise(summary, code):
        reasons.append('the code invents extra motion on the non-dominant arm instead of keeping the response focused and readable')
    if _uses_only_distal_arm_joints(code):
        reasons.append('the code only moves distal arm joints and lacks a readable upper-body posture')
    if _has_hand_without_postural_anchor(code):
        reasons.append('the code uses hand opening without a readable shoulder or head anchor, so the pose looks unclear')
    if _has_static_clip_with_arm_motion(summary, code):
        reasons.append('the clip is nearly static, but the generated response still moves an arm or hand instead of staying mostly head-led and calm')
    if _has_no_settle_after_dynamic_motion(summary, code):
        reasons.append('the code lacks a short settle at the end, so the response looks abrupt')

    calls = _parse_code_calls(code)
    fn_names = [name for name, _ in calls]
    unique_fns = set(fn_names)
    activity = str((summary or {}).get('activity_level') or 'low').lower()
    if activity in {'medium', 'high'} and 'move_arm_ik' not in unique_fns and 'move_joints' not in unique_fns:
        reasons.append('the code is too small for the visible motion energy in the clip')
    if len(_non_comment_code_lines(code)) <= 2:
        reasons.append('the response is too short and generic')
    if not reasons:
        reasons.append('the code is not sufficiently grounded in the specific clip')
    return '; '.join(reasons)


def _pick_backend(api_key: Optional[str]) -> str:
    requested = config.VLM_BACKEND
    if requested in {'openai', 'local'}:
        return requested
    if requested != 'auto':
        raise ValueError(f'Unsupported VLM_BACKEND={requested!r}')
    return 'openai' if api_key else 'local'


def _normalize_model_kind(model_id: str) -> str:
    lowered = model_id.lower()
    if 'qwen2.5-vl' in lowered:
        return 'qwen2_5_vl'
    if 'qwen2-vl' in lowered:
        return 'qwen2_vl'
    if 'smolvlm' in lowered:
        return 'smolvlm'
    raise ValueError(f'Unsupported local model {model_id!r}')


def _should_use_4bit(model_id: str) -> bool:
    setting = config.LOCAL_VLM_LOAD_IN_4BIT
    if setting in {'1', 'true', 'yes'}:
        return True
    if setting in {'0', 'false', 'no'}:
        return False
    lowered = model_id.lower()
    return '7b' in lowered or '72b' in lowered


def _load_local_model(model_id: str):
    global _LOCAL_MODEL_ID, _LOCAL_PROCESSOR, _LOCAL_MODEL, _LOCAL_MODEL_KIND

    if (
        _LOCAL_MODEL_ID == model_id
        and _LOCAL_PROCESSOR is not None
        and _LOCAL_MODEL is not None
        and _LOCAL_MODEL_KIND is not None
    ):
        return _LOCAL_PROCESSOR, _LOCAL_MODEL, _LOCAL_MODEL_KIND

    with _LOCAL_MODEL_LOCK:
        if (
            _LOCAL_MODEL_ID == model_id
            and _LOCAL_PROCESSOR is not None
            and _LOCAL_MODEL is not None
            and _LOCAL_MODEL_KIND is not None
        ):
            return _LOCAL_PROCESSOR, _LOCAL_MODEL, _LOCAL_MODEL_KIND

        import torch

        kind = _normalize_model_kind(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        quantization_config = None
        if torch.cuda.is_available() and _should_use_4bit(model_id):
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception as exc:
                print(f'[VLMClient] 4-bit quantization unavailable, fallback to normal load: {exc}')

        if kind in {'qwen2_5_vl', 'qwen2_vl'}:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration

            model_cls = Qwen2_5_VLForConditionalGeneration if kind == 'qwen2_5_vl' else Qwen2VLForConditionalGeneration

            processor = AutoProcessor.from_pretrained(model_id)
            load_kwargs = {
                'device_map': 'auto' if torch.cuda.is_available() else None,
            }
            if quantization_config is not None:
                load_kwargs['quantization_config'] = quantization_config
            else:
                load_kwargs['torch_dtype'] = dtype
            model = model_cls.from_pretrained(model_id, **load_kwargs)
            if not torch.cuda.is_available():
                model = model.to('cpu')
        else:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else None,
            )
            if not torch.cuda.is_available():
                model = model.to('cpu')

        model.eval()
        _LOCAL_MODEL_ID = model_id
        _LOCAL_PROCESSOR = processor
        _LOCAL_MODEL = model
        _LOCAL_MODEL_KIND = kind
        return processor, model, kind


def _load_local_model_on_cpu(model_id: str):
    import torch

    kind = _normalize_model_kind(model_id)
    dtype = torch.float32

    if kind == 'qwen2_5_vl':
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=None,
        )
        model = model.to('cpu')
    else:
        from transformers import AutoModelForImageTextToText, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=None,
        )
        model = model.to('cpu')

    model.eval()
    return processor, model, kind


def _qwen_multi_image_generate(processor, model, images, system_prompt: str, user_prompt: str, do_sample: bool = False) -> str:
    messages = [{
        'role': 'system',
        'content': [{'type': 'text', 'text': system_prompt}],
    }, {
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': user_prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors='pt', padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max(256, config.VLM_MAX_TOKENS),
        do_sample=do_sample,
        temperature=(config.LOCAL_VLM_TEMPERATURE if do_sample else None),
        top_p=(config.LOCAL_VLM_TOP_P if do_sample else None),
    )
    trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], output_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw


def _smolvlm_multi_image_generate(processor, model, images, system_prompt: str, user_prompt: str, do_sample: bool = False) -> str:
    conversation = [{
        'role': 'system',
        'content': [{'type': 'text', 'text': system_prompt}],
    }, {
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': user_prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    inputs = {
        key: (value.to(model.device) if hasattr(value, 'to') else value)
        for key, value in inputs.items()
    }
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max(256, config.VLM_MAX_TOKENS),
        do_sample=do_sample,
        temperature=(config.LOCAL_VLM_TEMPERATURE if do_sample else None),
        top_p=(config.LOCAL_VLM_TOP_P if do_sample else None),
    )
    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return raw


def _qwen_multi_image_generate_with_prompt(processor, model, images, system_prompt: str, user_prompt: str, do_sample: bool = False) -> str:
    messages = [{
        'role': 'system',
        'content': [{'type': 'text', 'text': system_prompt}],
    }, {
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': user_prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors='pt', padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max(256, config.VLM_MAX_TOKENS),
        do_sample=do_sample,
        temperature=(config.LOCAL_VLM_TEMPERATURE if do_sample else None),
        top_p=(config.LOCAL_VLM_TOP_P if do_sample else None),
    )
    trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _smolvlm_multi_image_generate_with_prompt(processor, model, images, system_prompt: str, user_prompt: str, do_sample: bool = False) -> str:
    conversation = [{
        'role': 'system',
        'content': [{'type': 'text', 'text': system_prompt}],
    }, {
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': user_prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors='pt',
    )
    inputs = {key: (value.to(model.device) if hasattr(value, 'to') else value) for key, value in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max(256, config.VLM_MAX_TOKENS),
        do_sample=do_sample,
        temperature=(config.LOCAL_VLM_TEMPERATURE if do_sample else None),
        top_p=(config.LOCAL_VLM_TOP_P if do_sample else None),
    )
    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def _generate_with_prompt(processor, model, kind: str, images, system_prompt: str, user_prompt: str, do_sample: bool = False) -> str:
    if kind == 'qwen2_5_vl':
        return _qwen_multi_image_generate_with_prompt(processor, model, images, system_prompt, user_prompt, do_sample=do_sample)
    return _smolvlm_multi_image_generate_with_prompt(processor, model, images, system_prompt, user_prompt, do_sample=do_sample)


def _is_probable_cuda_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return 'out of memory' in text or 'cuda out of memory' in text


def _build_repair_prompt(semantic_context: Dict[str, Any], python_code: str, error_text: str) -> str:
    return _LOCAL_REPAIR_PROMPT_TEMPLATE.format(
        semantic_json=json.dumps(semantic_context or {}, ensure_ascii=False, indent=2),
        python_code=python_code or '',
        error_text=error_text or 'unknown_error',
    )


def _build_refinement_prompt(semantic_context: Dict[str, Any], python_code: str, error_text: str) -> str:
    return _LOCAL_REFINEMENT_PROMPT_TEMPLATE.format(
        semantic_json=json.dumps(semantic_context or {}, ensure_ascii=False, indent=2),
        python_code=python_code or '',
        error_text=error_text or 'too_generic',
    )


def _build_repair_prompt_with_summary(
    frames_b64: Sequence[str],
    semantic_context: Dict[str, Any],
    python_code: str,
    error_text: str,
) -> str:
    summary = _infer_visual_motion_summary(frames_b64)
    base = _build_repair_prompt(semantic_context, python_code, error_text)
    return (
        base
        + "\n\nCheap motion summary from the same frames:\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\nUse it to keep the repaired code grounded in the observed motion."
    )


def _build_refinement_prompt_with_summary(
    frames_b64: Sequence[str],
    semantic_context: Dict[str, Any],
    python_code: str,
    error_text: str,
) -> str:
    summary = _infer_visual_motion_summary(frames_b64)
    base = _build_refinement_prompt(semantic_context, python_code, error_text)
    return (
        base
        + "\n\nCheap motion summary from the same frames:\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\nUse it only as weak low-level grounding. Keep the frames as the main evidence."
    )


def _format_candidate_blocks(candidates: Sequence[Tuple[Dict[str, Any], str]]) -> str:
    blocks = []
    for index, (semantic, code) in enumerate(candidates, start=1):
        block = (
            f"Candidate {index}\n"
            "```json\n"
            f"{json.dumps(semantic or {}, ensure_ascii=False, indent=2)}\n"
            "```\n"
            "```python\n"
            f"{code or ''}\n"
            "```"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_selection_prompt(summary: Dict[str, Any], candidates: Sequence[Tuple[Dict[str, Any], str]]) -> str:
    return _LOCAL_SELECTION_PROMPT_TEMPLATE.format(
        motion_summary=json.dumps(summary or {}, ensure_ascii=False, indent=2),
        candidate_blocks=_format_candidate_blocks(candidates),
    )


class _GeneratedCodeSanitizer(ast.NodeTransformer):
    def __init__(self, joint_limits: Dict[str, Tuple[float, float]]) -> None:
        self.joint_limits = dict(joint_limits or {})
        self.normalized_joint_names = {
            ''.join(ch for ch in name if ch.isalnum()).lower(): name
            for name in self.joint_limits
        }
        self.comfort_ratios = {
            'HeadPitch': 0.78,
            'LShoulderPitch': 0.82,
            'RShoulderPitch': 0.82,
            'LShoulderRoll': 0.84,
            'RShoulderRoll': 0.84,
            'LElbowRoll': 0.86,
            'RElbowRoll': 0.86,
        }

    def _literal(self, node):
        try:
            value = ast.literal_eval(node)
            if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (int, float)):
                return value[0]
            return value
        except Exception:
            return None

    def _find_keyword(self, node: ast.Call, keyword_name: str):
        for kw in node.keywords:
            if kw.arg == keyword_name:
                return kw
        return None

    def _canonical_joint_name(self, raw_name: Any) -> Any:
        if not isinstance(raw_name, str):
            return raw_name
        if raw_name in self.joint_limits:
            return raw_name
        normalized = ''.join(ch for ch in raw_name if ch.isalnum()).lower()
        alias_map = {
            'head': 'HeadYaw',
            'neck': 'HeadYaw',
            'headyaw': 'HeadYaw',
            'neckyaw': 'HeadYaw',
            'headpitch': 'HeadPitch',
            'neckpitch': 'HeadPitch',
        }
        if normalized in alias_map:
            return alias_map[normalized]
        if normalized in self.normalized_joint_names:
            return self.normalized_joint_names[normalized]
        matches = difflib.get_close_matches(normalized, list(self.normalized_joint_names.keys()), n=1, cutoff=0.72)
        if matches:
            return self.normalized_joint_names[matches[0]]
        return raw_name

    def _clip_joint_angle(self, joint_name: Any, angle: Any) -> Any:
        if not isinstance(joint_name, str) or not isinstance(angle, (int, float)):
            return angle
        limits = self.joint_limits.get(joint_name)
        if limits is None or limits[0] == limits[1]:
            return angle
        lo, hi = float(limits[0]), float(limits[1])
        target = float(angle)
        ratio = self.comfort_ratios.get(joint_name)
        if ratio is not None:
            center = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) * float(ratio)
            lo = center - half
            hi = center + half
        return float(max(lo, min(hi, target)))

    def _sanitize_oscillation(self, joint_name: Any, center: Any, amplitude: Any):
        if not isinstance(joint_name, str) or not isinstance(center, (int, float)) or not isinstance(amplitude, (int, float)):
            return center, amplitude
        limits = self.joint_limits.get(joint_name)
        if limits is None or limits[0] == limits[1]:
            return center, amplitude
        max_amp = max(0.0, min(abs(float(amplitude)), 0.7, (limits[1] - limits[0]) * 0.45))
        min_center = limits[0] + max_amp
        max_center = limits[1] - max_amp
        if min_center > max_center:
            mid = 0.5 * (limits[0] + limits[1])
            return mid, 0.0
        safe_center = float(max(min_center, min(max_center, float(center))))
        signed_amp = max_amp if float(amplitude) >= 0 else -max_amp
        return safe_center, signed_amp

    def _sanitize_hand_openness(self, openness: Any) -> Any:
        if not isinstance(openness, (int, float)):
            return openness
        return float(max(0.0, min(1.0, float(openness))))

    def _normalize_arm_side(self, side: Any) -> Any:
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

    def _sanitize_arm_target(self, side: Any, xyz: Any):
        side = self._normalize_arm_side(side)
        if not isinstance(side, str) or not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            return xyz
        try:
            x, y, z = [float(v) for v in xyz]
        except Exception:
            return xyz
        x = max(0.06, min(0.20, x))
        lateral = max(0.06, min(0.18, abs(y)))
        y = lateral if side == 'left' else -lateral
        z = max(-0.16, min(0.13, z))
        return [round(x, 4), round(y, 4), round(z, 4)]

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        fn_name = None
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn_name = node.func.attr

        if fn_name == 'move_joint':
            if len(node.args) >= 1:
                raw_name = self._literal(node.args[0])
                canonical = self._canonical_joint_name(raw_name)
                if isinstance(canonical, str):
                    node.args[0] = ast.Constant(value=canonical)
            else:
                kw = self._find_keyword(node, 'name')
                if kw is not None:
                    canonical = self._canonical_joint_name(self._literal(kw.value))
                    if isinstance(canonical, str):
                        kw.value = ast.Constant(value=canonical)
            joint_name = self._literal(node.args[0]) if len(node.args) >= 1 else self._literal(self._find_keyword(node, 'name').value) if self._find_keyword(node, 'name') else None
            if len(node.args) >= 2:
                safe_angle = self._clip_joint_angle(joint_name, self._literal(node.args[1]))
                if isinstance(safe_angle, (int, float)):
                    node.args[1] = ast.Constant(value=float(safe_angle))

        if fn_name == 'move_joints' and len(node.args) >= 1 and isinstance(node.args[0], ast.Dict):
            new_keys = []
            new_values = []
            for key_node, value_node in zip(node.args[0].keys, node.args[0].values):
                raw_name = self._literal(key_node)
                canonical = self._canonical_joint_name(raw_name)
                safe_angle = self._clip_joint_angle(canonical, self._literal(value_node))
                new_keys.append(ast.Constant(value=canonical) if isinstance(canonical, str) else key_node)
                new_values.append(ast.Constant(value=float(safe_angle)) if isinstance(safe_angle, (int, float)) else value_node)
            node.args[0].keys = new_keys
            node.args[0].values = new_values

        if fn_name == 'move_arm_ik':
            side = self._literal(node.args[0]) if len(node.args) >= 1 else self._literal(self._find_keyword(node, 'side').value) if self._find_keyword(node, 'side') else None
            normalized_side = self._normalize_arm_side(side)
            if len(node.args) >= 1 and isinstance(normalized_side, str):
                node.args[0] = ast.Constant(value=normalized_side)
            else:
                kw = self._find_keyword(node, 'side')
                if kw is not None and isinstance(normalized_side, str):
                    kw.value = ast.Constant(value=normalized_side)
            if len(node.args) >= 2:
                safe_xyz = self._sanitize_arm_target(normalized_side, self._literal(node.args[1]))
                if isinstance(safe_xyz, list):
                    node.args[1] = ast.List(elts=[ast.Constant(value=v) for v in safe_xyz], ctx=ast.Load())
            else:
                kw = self._find_keyword(node, 'xyz')
                safe_xyz = self._sanitize_arm_target(normalized_side, self._literal(kw.value) if kw is not None else None)
                if kw is not None and isinstance(safe_xyz, list):
                    kw.value = ast.List(elts=[ast.Constant(value=v) for v in safe_xyz], ctx=ast.Load())

        if fn_name == 'set_hand':
            side = self._literal(node.args[0]) if len(node.args) >= 1 else self._literal(self._find_keyword(node, 'side').value) if self._find_keyword(node, 'side') else None
            normalized_side = self._normalize_arm_side(side)
            if len(node.args) >= 1 and isinstance(normalized_side, str):
                node.args[0] = ast.Constant(value=normalized_side)
            else:
                kw = self._find_keyword(node, 'side')
                if kw is not None and isinstance(normalized_side, str):
                    kw.value = ast.Constant(value=normalized_side)
            if len(node.args) >= 2:
                safe_openness = self._sanitize_hand_openness(self._literal(node.args[1]))
                if isinstance(safe_openness, (int, float)):
                    node.args[1] = ast.Constant(value=float(safe_openness))
            else:
                kw = self._find_keyword(node, 'openness')
                safe_openness = self._sanitize_hand_openness(self._literal(kw.value) if kw is not None else None)
                if kw is not None and isinstance(safe_openness, (int, float)):
                    kw.value = ast.Constant(value=float(safe_openness))

        if fn_name == 'oscillate_joint':
            if len(node.args) >= 1:
                raw_name = self._literal(node.args[0])
                canonical = self._canonical_joint_name(raw_name)
                if isinstance(canonical, str):
                    node.args[0] = ast.Constant(value=canonical)
            else:
                kw = self._find_keyword(node, 'name')
                canonical = self._canonical_joint_name(self._literal(kw.value) if kw is not None else None)
                if kw is not None and isinstance(canonical, str):
                    kw.value = ast.Constant(value=canonical)
            joint_name = self._literal(node.args[0]) if len(node.args) >= 1 else self._literal(self._find_keyword(node, 'name').value) if self._find_keyword(node, 'name') else None
            center_node = node.args[1] if len(node.args) >= 2 else self._find_keyword(node, 'center').value if self._find_keyword(node, 'center') else None
            amplitude_node = node.args[2] if len(node.args) >= 3 else self._find_keyword(node, 'amplitude').value if self._find_keyword(node, 'amplitude') else None
            safe_center, safe_amplitude = self._sanitize_oscillation(joint_name, self._literal(center_node), self._literal(amplitude_node))
            if len(node.args) >= 2 and isinstance(safe_center, (int, float)):
                node.args[1] = ast.Constant(value=float(safe_center))
            else:
                kw = self._find_keyword(node, 'center')
                if kw is not None and isinstance(safe_center, (int, float)):
                    kw.value = ast.Constant(value=float(safe_center))
            if len(node.args) >= 3 and isinstance(safe_amplitude, (int, float)):
                node.args[2] = ast.Constant(value=float(safe_amplitude))
            else:
                kw = self._find_keyword(node, 'amplitude')
                if kw is not None and isinstance(safe_amplitude, (int, float)):
                    kw.value = ast.Constant(value=float(safe_amplitude))

        return node


def _sanitize_generated_code(code: str, joint_limits: Dict[str, Tuple[float, float]]) -> str:
    if not code.strip():
        return code
    try:
        tree = ast.parse(code)
        tree = _GeneratedCodeSanitizer(joint_limits).visit(tree)
        ast.fix_missing_locations(tree)
        return textwrap.dedent(ast.unparse(tree)).strip()
    except Exception:
        return code


def _normalize_generated_code(code: str) -> str:
    if not code:
        return code

    normalized = textwrap.dedent(code).strip()

    if normalized.startswith('{') and normalized.endswith('}') and any(token in normalized for token in ('oscillate_joint', 'move_arm_ik', 'set_hand', 'move_joint', 'move_joints')):
        extracted_calls = re.findall(r'"([^"]*(?:move_joint|move_joints|move_arm_ik|set_hand|oscillate_joint|hold|idle)[^"]*)"', normalized)
        if extracted_calls:
            normalized = '\n'.join(call.strip() for call in extracted_calls if call.strip())

    for prefix in ('nao.', 'robot.', 'agent.'):
        normalized = normalized.replace(prefix, '')

    normalized = re.sub(
        r'move_head\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
        r"move_joints({'HeadYaw': \1, 'HeadPitch': \2}, \3, 'min_jerk')",
        normalized,
    )
    normalized = re.sub(
        r'move_head\s*\(\s*yaw\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*pitch\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*duration\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
        r"move_joints({'HeadYaw': \1, 'HeadPitch': \2}, \3, 'min_jerk')",
        normalized,
    )
    normalized = re.sub(
        r'move_head\s*\(\s*pitch\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*yaw\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*duration\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
        r"move_joints({'HeadYaw': \2, 'HeadPitch': \1}, \3, 'min_jerk')",
        normalized,
    )
    normalized = re.sub(
        r'move_head\s*\(\s*yaw\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*pitch\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
        r"move_joints({'HeadYaw': \1, 'HeadPitch': \2}, 0.2, 'min_jerk')",
        normalized,
    )
    normalized = re.sub(
        r'move_head\s*\(\s*pitch\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*yaw\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)',
        r"move_joints({'HeadYaw': \2, 'HeadPitch': \1}, 0.2, 'min_jerk')",
        normalized,
    )

    replacements = {
        'right_wrist': 'RWristYaw',
        'left_wrist': 'LWristYaw',
        'right_elbow': 'RElbowRoll',
        'left_elbow': 'LElbowRoll',
        'right_shoulder': 'RShoulderPitch',
        'left_shoulder': 'LShoulderPitch',
        'neck_yaw': 'HeadYaw',
        'neck_pitch': 'HeadPitch',
        'head_yaw': 'HeadYaw',
        'head_pitch': 'HeadPitch',
        'right_shoulder_pitch': 'RShoulderPitch',
        'right_shoulder_roll': 'RShoulderRoll',
        'right_elbow_yaw': 'RElbowYaw',
        'right_elbow_roll': 'RElbowRoll',
        'right_wrist_yaw': 'RWristYaw',
        'rightwristy': 'RWristYaw',
        'rightwristyaw': 'RWristYaw',
        'left_shoulder_pitch': 'LShoulderPitch',
        'left_shoulder_roll': 'LShoulderRoll',
        'left_elbow_yaw': 'LElbowYaw',
        'left_elbow_roll': 'LElbowRoll',
        'left_wrist_yaw': 'LWristYaw',
        'leftwristy': 'LWristYaw',
        'leftwristyaw': 'LWristYaw',
    }
    for src, dst in replacements.items():
        normalized = normalized.replace(f'"{src}"', f'"{dst}"')
        normalized = normalized.replace(f"'{src}'", f"'{dst}'")

    normalized = re.sub(r'\bheadYaw\b', 'HeadYaw', normalized)
    normalized = re.sub(r'\bheadPitch\b', 'HeadPitch', normalized)
    normalized = re.sub(r'\brShoulderPitch\b', 'RShoulderPitch', normalized)
    normalized = re.sub(r'\brShoulderRoll\b', 'RShoulderRoll', normalized)
    normalized = re.sub(r'\brElbowYaw\b', 'RElbowYaw', normalized)
    normalized = re.sub(r'\brElbowRoll\b', 'RElbowRoll', normalized)
    normalized = re.sub(r'\brWristYaw\b', 'RWristYaw', normalized)
    normalized = re.sub(r'\bRightWristY\b', 'RWristYaw', normalized)
    normalized = re.sub(r'\bRightWristYaw\b', 'RWristYaw', normalized)
    normalized = re.sub(r'\blShoulderPitch\b', 'LShoulderPitch', normalized)
    normalized = re.sub(r'\blShoulderRoll\b', 'LShoulderRoll', normalized)
    normalized = re.sub(r'\blElbowYaw\b', 'LElbowYaw', normalized)
    normalized = re.sub(r'\blElbowRoll\b', 'LElbowRoll', normalized)
    normalized = re.sub(r'\blWristYaw\b', 'LWristYaw', normalized)
    normalized = re.sub(r'\bLeftWristY\b', 'LWristYaw', normalized)
    normalized = re.sub(r'\bLeftWristYaw\b', 'LWristYaw', normalized)

    left_sign_fixes = {
        'LShoulderRoll': (0.0, None),
        'LElbowYaw': (None, None),
        'LElbowRoll': (None, 0.0),
    }
    for joint_name, (must_be_ge, must_be_le) in left_sign_fixes.items():
        pattern = re.compile(rf"(['\"])({joint_name})\1\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
        def _fix(match):
            quote = match.group(1)
            name = match.group(2)
            value = float(match.group(3))
            if must_be_ge is not None and value < must_be_ge:
                value = abs(value)
            if must_be_le is not None and value > must_be_le:
                value = -abs(value)
            return f"{quote}{name}{quote}: {value:.4f}"
        normalized = pattern.sub(_fix, normalized)

    normalized = re.sub(
        r"oscillate_joint\(\s*name\s*=\s*['\"]RWristYaw['\"]\s*,\s*center\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        lambda m: f"oscillate_joint(name='RWristYaw', center={max(-1.2, min(1.2, float(m.group(1)))):.4f}",
        normalized,
    )
    normalized = re.sub(
        r"oscillate_joint\(\s*name\s*=\s*['\"]LWristYaw['\"]\s*,\s*center\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        lambda m: f"oscillate_joint(name='LWristYaw', center={max(-1.2, min(1.2, float(m.group(1)))):.4f}",
        normalized,
    )
    normalized = re.sub(
        r"oscillate_joint\(\s*['\"]RWristYaw['\"]\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        lambda m: f"oscillate_joint('RWristYaw', {max(-1.2, min(1.2, float(m.group(1)))):.4f}",
        normalized,
    )
    normalized = re.sub(
        r"oscillate_joint\(\s*['\"]LWristYaw['\"]\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        lambda m: f"oscillate_joint('LWristYaw', {max(-1.2, min(1.2, float(m.group(1)))):.4f}",
        normalized,
    )

    lines = [line.rstrip() for line in normalized.splitlines()]
    return '\n'.join(lines).strip()


class VLMClient:
    def __init__(
        self,
        joint_limits: Dict[str, Tuple[float, float]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.LLM_API_KEY or None
        self.base_url = base_url or config.LLM_BASE_URL or None
        self.joint_limits = dict(joint_limits or {})
        self.system_prompt = build_system_prompt(joint_limits)
        self.static_validator = _build_static_validator(self.joint_limits)
        self.runtime_validator = _build_static_validator(self.joint_limits)
        self.backend = _pick_backend(self.api_key)
        self.client = None

        if self.backend == 'openai':
            if OpenAI is None:
                if config.VLM_BACKEND == 'auto':
                    self.backend = 'local'
                else:
                    raise RuntimeError('openai SDK missing; install `openai` or switch VLM_BACKEND=local')
            elif not self.api_key:
                if config.VLM_BACKEND == 'auto':
                    self.backend = 'local'
                else:
                    raise RuntimeError('VLM API key missing. Set `llm_api_key` or use VLM_BACKEND=local')

        if self.backend == 'openai':
            client_kwargs: Dict[str, Any] = {'api_key': self.api_key}
            if self.base_url:
                client_kwargs['base_url'] = self.base_url
            self.client = OpenAI(**client_kwargs)
            self.model = model or config.VLM_MODEL
        else:
            self.model = model or config.LOCAL_VLM_MODEL

    def call(self, frames_b64: Sequence[str]) -> VLMResponse:
        if self.backend == 'openai':
            return self._call_openai(frames_b64)
        return self._call_local(frames_b64)

    def repair(self, frames_b64: Sequence[str], semantic_context: Dict[str, Any], python_code: str, error_text: str) -> VLMResponse:
        if self.backend == 'openai':
            return VLMResponse({}, '', '', 0.0, False, error='repair_not_supported_on_openai_backend')
        return self._repair_local(frames_b64, semantic_context, python_code, error_text)

    def _call_openai(self, frames_b64: Sequence[str]) -> VLMResponse:
        if not frames_b64:
            return VLMResponse({}, '', '', 0.0, False, error='no frames provided')

        content: List[Dict[str, Any]] = [{"type": "text", "text": _USER_PROMPT}]
        for img_b64 in frames_b64:
            content.append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{img_b64}',
                    'detail': config.VLM_IMAGE_DETAIL,
                },
            })

        t0 = time.time()
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': content},
                ],
                max_tokens=config.VLM_MAX_TOKENS,
                temperature=config.VLM_TEMPERATURE,
            )
            raw = rsp.choices[0].message.content or ''
            elapsed = time.time() - t0
        except Exception as exc:
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(exc))

        semantic, code = parse_vlm_output(raw)
        code = _sanitize_generated_code(_normalize_generated_code(code), self.joint_limits)
        ok = bool(code) and bool(semantic)
        return VLMResponse(
            semantic_context=semantic,
            python_code=code,
            raw_text=raw,
            elapsed_seconds=elapsed,
            ok=ok,
            error=None if ok else 'parse_incomplete',
        )

    def _call_local(self, frames_b64: Sequence[str]) -> VLMResponse:
        if not frames_b64:
            return VLMResponse({}, '', '', 0.0, False, error='no frames provided')

        if config.LOCAL_VLM_SERVER_URL:
            return self._call_local_server(frames_b64)

        t0 = time.time()
        try:
            images = _decode_frames_to_pil(frames_b64)
            processor, model, kind = _load_local_model(self.model)

            summary = _infer_visual_motion_summary(frames_b64)
            user_prompt = _build_local_user_prompt(frames_b64)
            candidate_count = max(1, int(config.LOCAL_VLM_NUM_CANDIDATES))
            raw_candidates: List[str] = []

            for idx in range(candidate_count):
                do_sample = idx > 0
                if kind == 'qwen2_5_vl':
                    raw = _qwen_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=do_sample)
                else:
                    raw = _smolvlm_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=do_sample)
                raw_candidates.append(raw)

            parsed_candidates: List[Tuple[Dict[str, Any], str, str]] = []
            valid_candidates: List[Tuple[Dict[str, Any], str, str]] = []

            for raw in raw_candidates:
                semantic, code = parse_vlm_output(raw)
                code = _sanitize_generated_code(_normalize_generated_code(code), self.joint_limits)
                if not _candidate_has_minimal_structure(semantic, code):
                    if config.LOCAL_VLM_DEBUG:
                        print(f'[VLMClient][local] candidate rejected: parse/min-structure failed semantic={semantic}')
                    continue
                parsed_candidates.append((semantic, code, raw))
                validation = self.static_validator.validate(code)
                runtime_ok = validation.ok and _runtime_signature_ok(self.runtime_validator, code)
                if config.LOCAL_VLM_DEBUG:
                    print(f'[VLMClient][local] candidate valid={validation.ok and runtime_ok} semantic={semantic}')
                if validation.ok and runtime_ok:
                    valid_candidates.append((semantic, code, raw))

            if not valid_candidates:
                if parsed_candidates:
                    seed_semantic, seed_code, seed_raw = parsed_candidates[0]
                else:
                    seed_raw = raw_candidates[0] if raw_candidates else ''
                    seed_semantic, seed_code = parse_vlm_output(seed_raw)
                    seed_code = _sanitize_generated_code(_normalize_generated_code(seed_code), self.joint_limits)
                seed_issue = _describe_code_quality_issue(summary, seed_semantic, seed_code)
                parse_repair_prompt = _build_repair_prompt_with_summary(
                    frames_b64,
                    seed_semantic,
                    seed_code,
                    f'no_valid_candidate_after_static_validation; {seed_issue}',
                )
                repaired_semantic, repaired_code, repaired_raw, repaired_valid, repaired_quality = _pick_best_generated_variant(
                    processor, model, kind, images, self.system_prompt, parse_repair_prompt,
                    self.joint_limits, self.static_validator, summary, attempts=3,
                )
                repaired_validation = self.static_validator.validate(repaired_code)
                repaired_runtime_ok = repaired_validation.ok and _runtime_signature_ok(self.runtime_validator, repaired_code)
                if _candidate_has_minimal_structure(repaired_semantic, repaired_code) and repaired_validation.ok and repaired_runtime_ok:
                    if repaired_quality[1] == 1 or repaired_quality[0] >= 4.0:
                        second_prompt = parse_repair_prompt + (
                            "\n\nMake the response more readable and clip-specific: choose one dominant arm or head-led posture, "
                            "avoid mirrored both-arm motion, and keep 2-4 primitive calls."
                        )
                        second_semantic, second_code, second_raw, second_valid, second_quality = _pick_best_generated_variant(
                            processor, model, kind, images, self.system_prompt, second_prompt,
                            self.joint_limits, self.static_validator, summary, attempts=3,
                        )
                        second_validation = self.static_validator.validate(second_code)
                        second_runtime_ok = second_validation.ok and _runtime_signature_ok(self.runtime_validator, second_code)
                        if (
                            _candidate_has_minimal_structure(second_semantic, second_code)
                            and second_validation.ok and second_runtime_ok
                            and second_quality < repaired_quality
                        ):
                            repaired_semantic, repaired_code, repaired_raw = second_semantic, second_code, second_raw
                    return VLMResponse(
                        semantic_context=repaired_semantic,
                        python_code=repaired_code,
                        raw_text=repaired_raw,
                        elapsed_seconds=time.time() - t0,
                        ok=True,
                        error=None,
                    )
                runtime_err = None if repaired_runtime_ok else 'runtime_signature_mismatch'
                return VLMResponse({}, '', repaired_raw, time.time() - t0, False, error=f'parse_incomplete:{repaired_validation.error or runtime_err}')
            pool = sorted(
                valid_candidates,
                key=lambda item: _candidate_quality(summary, item[0], item[1], True),
            )

            if len(pool) == 1:
                best_semantic, best_code, best_raw = pool[0]
            else:
                selection_prompt = _build_selection_prompt(summary, [(semantic, code) for semantic, code, _ in pool])
                if kind == 'qwen2_5_vl':
                    selected_raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, selection_prompt)
                else:
                    selected_raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, selection_prompt)
                chosen_index = parse_chosen_candidate_index(selected_raw)
                if config.LOCAL_VLM_DEBUG:
                    print(f'[VLMClient][local] chosen candidate raw={selected_raw!r} parsed_index={chosen_index}')
                default_semantic, default_code, default_raw = pool[0]
                default_quality = _candidate_quality(summary, default_semantic, default_code, True)
                if chosen_index is not None and 1 <= chosen_index <= len(pool):
                    chosen_semantic, chosen_code, chosen_raw = pool[chosen_index - 1]
                    chosen_quality = _candidate_quality(summary, chosen_semantic, chosen_code, True)
                    if chosen_quality[0] <= default_quality[0] + 1.0:
                        best_semantic, best_code, best_raw = chosen_semantic, chosen_code, chosen_raw
                    else:
                        if config.LOCAL_VLM_DEBUG:
                            print(f'[VLMClient][local] selection guard kept pool[0]: chosen_quality={chosen_quality} default_quality={default_quality}')
                        best_semantic, best_code, best_raw = default_semantic, default_code, default_raw
                else:
                    best_semantic, best_code, best_raw = default_semantic, default_code, default_raw

            best_quality = _candidate_quality(summary, best_semantic, best_code, True)

            if best_code and (
                _looks_too_generic_for_clip(summary, best_semantic, best_code)
                or _naturalness_penalty(summary, best_code) >= 3.0
            ):
                quality_issue = _describe_code_quality_issue(summary, best_semantic, best_code)
                refinement_prompt = _build_refinement_prompt_with_summary(
                    frames_b64,
                    best_semantic,
                    best_code,
                    quality_issue,
                )
                refined_semantic, refined_code, refined_raw, refined_valid, refined_quality = _pick_best_generated_variant(
                    processor, model, kind, images, self.system_prompt, refinement_prompt,
                    self.joint_limits, self.static_validator, summary, attempts=3,
                )
                refined_validation = self.static_validator.validate(refined_code)
                refined_runtime_ok = refined_validation.ok and _runtime_signature_ok(self.runtime_validator, refined_code)
                if config.LOCAL_VLM_DEBUG:
                    print(f'[VLMClient][local] refinement valid={refined_validation.ok and refined_runtime_ok} semantic={refined_semantic} quality={refined_quality}')
                if (
                    _candidate_has_minimal_structure(refined_semantic, refined_code)
                    and refined_validation.ok and refined_runtime_ok
                    and refined_quality < best_quality
                ):
                    best_semantic, best_code, best_raw = refined_semantic, refined_code, refined_raw
                    best_quality = refined_quality
                elif best_quality[1] == 1 or best_quality[0] >= 4.0:
                    second_prompt = refinement_prompt + (
                        "\n\nTry a more readable pet-like response: use one dominant arm and/or head orientation, "
                        "avoid bilateral mirroring, make the intent visible in 2-4 primitive calls, and finish with a short hold or idle."
                    )
                    second_semantic, second_code, second_raw, second_valid, second_quality = _pick_best_generated_variant(
                        processor, model, kind, images, self.system_prompt, second_prompt,
                        self.joint_limits, self.static_validator, summary, attempts=3,
                    )
                    second_validation = self.static_validator.validate(second_code)
                    second_runtime_ok = second_validation.ok and _runtime_signature_ok(self.runtime_validator, second_code)
                    if config.LOCAL_VLM_DEBUG:
                        print(f'[VLMClient][local] refinement second-pass valid={second_validation.ok and second_runtime_ok} semantic={second_semantic} quality={second_quality}')
                    if (
                        _candidate_has_minimal_structure(second_semantic, second_code)
                        and second_validation.ok and second_runtime_ok
                        and second_quality < best_quality
                    ):
                        best_semantic, best_code, best_raw = second_semantic, second_code, second_raw
                        best_quality = second_quality

            final_validation = self.static_validator.validate(best_code)
            if ((not best_code or not best_semantic) or (not final_validation.ok)) and parsed_candidates:
                fallback_semantic, fallback_code, _ = parsed_candidates[0]
                parse_repair_prompt = _build_repair_prompt_with_summary(
                    frames_b64,
                    fallback_semantic,
                    fallback_code,
                    'candidate_parse_or_selection_failed',
                )
                repaired_semantic, repaired_code, repaired_raw, repaired_valid, repaired_quality = _pick_best_generated_variant(
                    processor, model, kind, images, self.system_prompt, parse_repair_prompt,
                    self.joint_limits, self.static_validator, summary, attempts=3,
                )
                repaired_validation = self.static_validator.validate(repaired_code)
                repaired_runtime_ok = repaired_validation.ok and _runtime_signature_ok(self.runtime_validator, repaired_code)
                if (
                    _candidate_has_minimal_structure(repaired_semantic, repaired_code)
                    and repaired_validation.ok and repaired_runtime_ok
                    and repaired_quality < best_quality
                ):
                    best_semantic, best_code, best_raw = repaired_semantic, repaired_code, repaired_raw
                    best_quality = repaired_quality

            final_validation = self.static_validator.validate(best_code)
            final_runtime_ok = final_validation.ok and _runtime_signature_ok(self.runtime_validator, best_code)
            ok = bool(best_code) and bool(best_semantic) and final_validation.ok and final_runtime_ok
            return VLMResponse(
                semantic_context=best_semantic,
                python_code=best_code,
                raw_text=best_raw,
                elapsed_seconds=time.time() - t0,
                ok=ok,
                error=None if ok else f'invalid_or_incomplete:{final_validation.error or "runtime_signature_mismatch"}',
            )
        except Exception as exc:
            if _is_probable_cuda_oom(exc):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    processor, model, kind = _load_local_model_on_cpu(self.model)
                    images = _decode_frames_to_pil(frames_b64)
                    user_prompt = _build_local_user_prompt(frames_b64)
                    if kind == 'qwen2_5_vl':
                        raw = _qwen_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=False)
                    else:
                        raw = _smolvlm_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=False)
                    semantic, code = parse_vlm_output(raw)
                    code = _sanitize_generated_code(_normalize_generated_code(code), self.joint_limits)
                    validation = self.static_validator.validate(code)
                    if (not _candidate_has_minimal_structure(semantic, code)) or (not validation.ok):
                        repair_prompt = _build_repair_prompt_with_summary(frames_b64, semantic, code, 'cpu_fallback_repair')
                        if kind == 'qwen2_5_vl':
                            raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)
                        else:
                            raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)
                        semantic, code = parse_vlm_output(raw)
                        code = _sanitize_generated_code(_normalize_generated_code(code), self.joint_limits)
                        validation = self.static_validator.validate(code)
                    ok = bool(code) and bool(semantic) and validation.ok
                    return VLMResponse(
                        semantic_context=semantic,
                        python_code=code,
                        raw_text=raw,
                        elapsed_seconds=time.time() - t0,
                        ok=ok,
                        error=None if ok else f'parse_incomplete_cpu_fallback:{validation.error}',
                    )
                except Exception as nested_exc:
                    return VLMResponse({}, '', '', time.time() - t0, False, error=str(nested_exc))
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(exc))

    def _call_local_server(self, frames_b64: Sequence[str]) -> VLMResponse:
        t0 = time.time()
        try:
            rsp = requests.post(
                config.LOCAL_VLM_SERVER_URL.rstrip('/') + '/generate_from_frames',
                json={
                    'frames_b64': list(frames_b64),
                    'model': self.model,
                },
                timeout=max(30.0, config.ONE_SHOT_VLM_TIMEOUT),
            )
            rsp.raise_for_status()
            payload = rsp.json()
            return VLMResponse(
                semantic_context=payload.get('semantic_context') or {},
                python_code=payload.get('python_code') or '',
                raw_text=payload.get('raw_text') or '',
                elapsed_seconds=float(payload.get('elapsed_seconds') or (time.time() - t0)),
                ok=bool(payload.get('ok')),
                error=payload.get('error'),
            )
        except Exception as exc:
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(exc))

    def _repair_local(self, frames_b64: Sequence[str], semantic_context: Dict[str, Any], python_code: str, error_text: str) -> VLMResponse:
        if not frames_b64:
            return VLMResponse({}, '', '', 0.0, False, error='no frames provided')

        t0 = time.time()
        try:
            images = _decode_frames_to_pil(frames_b64)
            processor, model, kind = _load_local_model(self.model)
            summary = _infer_visual_motion_summary(frames_b64)
            repair_prompt = _build_repair_prompt_with_summary(frames_b64, semantic_context, python_code, error_text)
            semantic, code, raw, _, _ = _pick_best_generated_variant(
                processor, model, kind, images, self.system_prompt, repair_prompt,
                self.joint_limits, self.static_validator, summary, attempts=3,
            )
            validation = self.static_validator.validate(code)
            ok = bool(code) and bool(semantic) and validation.ok
            return VLMResponse(
                semantic_context=semantic,
                python_code=code,
                raw_text=raw,
                elapsed_seconds=time.time() - t0,
                ok=ok,
                error=None if ok else f'parse_incomplete:{validation.error}',
            )
        except Exception as exc:
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(exc))


_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_PYTHON_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_CHOSEN_CANDIDATE_RE = re.compile(r"CHOSEN_CANDIDATE\s*:\s*(\d+)", re.IGNORECASE)


def parse_vlm_output(raw_text: str) -> Tuple[Dict[str, Any], str]:
    semantic: Dict[str, Any] = {}
    code = ''

    match_json = _JSON_RE.search(raw_text or '')
    if match_json:
        try:
            semantic = json.loads(match_json.group(1).strip())
        except json.JSONDecodeError as exc:
            print(f'[VLMClient] JSON parse error: {exc}')

    match_python = _PYTHON_RE.search(raw_text or '')
    if match_python:
        code = textwrap.dedent(match_python.group(1)).strip()

    return semantic, code


def parse_chosen_candidate_index(raw_text: str) -> Optional[int]:
    match = _CHOSEN_CANDIDATE_RE.search(raw_text or '')
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None
