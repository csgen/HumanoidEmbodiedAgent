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

    oscillate_joint(name: str, center: float, amplitude: float,
                    frequency: float, duration: float, decay: float = 0.0)

    hold(duration: float)

    idle(duration: float)

    speak(text: str)

    navigate_to(delta_x: float, delta_y: float, delta_theta: float)
        # Coarse demo locomotion primitive in Webots Supervisor mode.
        # Useful for a small pet-like approach or backoff when the human's
        # body language clearly invites or rejects the robot.

# Demo constraints
- For this project stage, DO NOT walk and DO NOT use locomotion.
- Do NOT call `navigate_to(...)`.
- Prefer upper-body behavior only: head, shoulders, elbows, wrists, hands.
- Avoid lower-body joints unless absolutely necessary; in this stage they are not needed.
- Keep motions short, smooth, physically plausible, and pet-like.
- React naturally rather than mimic exactly; the response should feel socially appropriate.
- Avoid exaggerated repeated oscillations unless the video strongly supports them.

# NAO coordinate convention
- Torso-centred, right-hand frame. Units are METRES.
- x forward, y LEFT is positive, z up from torso.
- Neutral arm: right hand ≈ [0.02, -0.10, -0.20]. Typical waving pose:
  right hand raised to roughly [0.15, -0.15, 0.10].

# Joint reference (radian limits)
{joint_limit_block}

# Diversity and grounding requirements
- Different input clips should generally produce DIFFERENT control code.
- Reusing the same wrist-only oscillation for unrelated clips is incorrect.
- Base side choice, dominant joints, tempo, and hold-vs-oscillation decisions on
  the actual frame sequence.
- If the human motion is mostly static, orienting and holding are often more
  grounded than repeated oscillation.
- If the motion is lateral or directional, let head/arm orientation naturally
  reflect that direction when appropriate.
- If the human appears approving, welcoming, or playful, a gentle pet-like
  acknowledgment is acceptable, but do not hardcode any named response.

# Composition guidance
- Prefer short sequences of 2-5 primitive calls.
- Use `move_joints(...)` when a coordinated upper-body posture is needed.
- Use `hold(...)` when the response should pause, freeze, or attend.
- Use `oscillate_joint(...)` only when the observed motion really suggests a
  rhythmic or waving-like response.
- Return smoothly toward a calm upper-body posture after a dynamic motion.

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
    "generated control program."
)

_LOCAL_REPAIR_PROMPT_TEMPLATE = """The previous robot-control program was not acceptable.

Use the SAME video frames and produce a corrected result.

Rules:
- Keep the robot fully controlled by VLM-generated code.
- Do not use any intermediate gesture labels.
- Do not use canned named actions.
- Use only these primitives: move_joint, move_joints, move_arm_ik, oscillate_joint, hold, idle, speak.
- Do NOT use navigate_to.
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
- Use only these primitives: move_joint, move_joints, move_arm_ik, oscillate_joint, hold, idle, speak.
- Do NOT use navigate_to.
- Do NOT use lower-body joints.
- The new code must differ materially from the previous code if the previous code was too generic.
- Avoid a wrist-only oscillation unless the video truly supports it.
- If the clip is directional, use head orientation and arm orientation to reflect that direction.
- If the clip is static or prohibitive, prefer hold / stillness over repeated waving.

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

    prev_cx = None
    prev_cy = None
    for prev, cur in zip(frames[:-1], frames[1:]):
        diff = np.abs(cur - prev)
        energy = float(diff.mean())
        total_energy += energy

        h, w = diff.shape[:2]
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

    return {
        'frame_count': len(frames_b64),
        'motion_energy': round(mean_energy, 4),
        'dominant_axis': dominant_axis,
        'lateral_bias': lateral_bias,
        'vertical_bias': vertical_bias,
        'activity_level': activity,
    }


def _build_motion_prior_text(summary: Dict[str, Any]) -> str:
    dominant_axis = str(summary.get('dominant_axis') or 'static')
    lateral_bias = str(summary.get('lateral_bias') or 'center')
    vertical_bias = str(summary.get('vertical_bias') or 'center')
    activity = str(summary.get('activity_level') or 'low')
    energy = float(summary.get('motion_energy') or 0.0)

    lines = [
        '# Primitive-level motion prior derived from the clip',
        '- This prior is low-level and kinematic; use it to ground primitive choice.',
    ]

    if dominant_axis == 'horizontal':
        lines.append('- Motion is mainly lateral: prefer HeadYaw / ShoulderRoll / ElbowYaw orientation over wrist-only waving.')
    elif dominant_axis == 'vertical':
        lines.append('- Motion is mainly vertical: prefer HeadPitch / ShoulderPitch changes and brief holds over lateral wrist oscillation.')
    elif dominant_axis == 'mixed':
        lines.append('- Motion is mixed and relatively energetic: a short composed response with arm lift plus recovery is acceptable.')
    else:
        lines.append('- Motion is weak or static: prefer stillness, small orientation, and hold instead of repeated oscillation.')

    if lateral_bias == 'leftward_in_image':
        lines.append('- The motion drifts left in the image: a lateral orienting response is more grounded than a symmetric response.')
    elif lateral_bias == 'rightward_in_image':
        lines.append('- The motion drifts right in the image: a lateral orienting response is more grounded than a symmetric response.')

    if vertical_bias == 'upward_in_image':
        lines.append('- Upward drift is present: lifting attention or slightly raising the upper body chain can be natural.')
    elif vertical_bias == 'downward_in_image':
        lines.append('- Downward drift is present: a mild lowering or settling response can be natural.')

    if activity == 'high':
        lines.append('- Activity is high: one brief dynamic phase is acceptable, but still return smoothly to a calm posture.')
    elif activity == 'medium':
        lines.append('- Activity is medium: use moderate motion; avoid long repetitive oscillations.')
    else:
        lines.append('- Activity is low: keep the response short and restrained.')

    if energy < 0.02:
        lines.append('- Because motion energy is low, holding posture is often more grounded than repeated rhythmic movement.')

    return '\n'.join(lines)


def _build_local_user_prompt(frames_b64: Sequence[str]) -> str:
    summary = _infer_visual_motion_summary(frames_b64)
    return (
        _LOCAL_USER_PROMPT
        + "\n\n# Cheap motion summary extracted from the frame sequence\n"
        + json.dumps(summary, ensure_ascii=False, indent=2)
        + "\nUse this only as weak grounding. The frames remain the main evidence.\n\n"
        + _build_motion_prior_text(summary)
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


def _build_static_validator(joint_limits: Dict[str, Tuple[float, float]]) -> SandboxExecutor:
    executor = SandboxExecutor()
    executor.register_many({
        'move_joint': lambda *a, **k: None,
        'move_joints': lambda *a, **k: None,
        'move_arm_ik': lambda *a, **k: None,
        'oscillate_joint': lambda *a, **k: None,
        'hold': lambda *a, **k: None,
        'idle': lambda *a, **k: None,
        'speak': lambda *a, **k: None,
        'navigate_to': lambda *a, **k: None,
    })
    executor.set_joint_limits(joint_limits)
    return executor


def _candidate_has_minimal_structure(semantic: Dict[str, Any], code: str) -> bool:
    if not semantic or not code:
        return False
    calls = _parse_code_calls(code)
    return bool(calls)


def _looks_too_generic_for_clip(semantic: Dict[str, Any], code: str) -> bool:
    if not code:
        return True
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.strip().startswith('#')]
    calls = _parse_code_calls(code)
    fn_names = [name for name, _ in calls]
    unique_fns = set(fn_names)
    intent = str(semantic.get('intent') or '').lower()
    dynamics = str(semantic.get('motion_dynamics') or '').lower()

    if len(lines) <= 2 and unique_fns == {'oscillate_joint'} and 'RWristYaw' in code:
        return True
    if len(lines) <= 2 and 'point' in intent and 'HeadYaw' not in code and 'ShoulderRoll' not in code and 'ElbowYaw' not in code:
        return True
    if len(lines) <= 2 and ('stop' in intent or dynamics == 'static') and 'hold(' not in code:
        return True
    if len(lines) <= 2 and any(word in intent for word in ('thumb', 'approve', 'approval', 'yes')) and 'HeadPitch' not in code:
        return True
    return False


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
    if 'smolvlm' in lowered:
        return 'smolvlm'
    raise ValueError(f'Unsupported local model {model_id!r}')


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

        if kind == 'qwen2_5_vl':
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            processor = AutoProcessor.from_pretrained(model_id)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map='auto' if torch.cuda.is_available() else None,
            )
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


def _qwen_multi_image_generate_with_prompt(processor, model, images, system_prompt: str, user_prompt: str) -> str:
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
    output_ids = model.generate(**inputs, max_new_tokens=max(256, config.VLM_MAX_TOKENS), do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def _smolvlm_multi_image_generate_with_prompt(processor, model, images, system_prompt: str, user_prompt: str) -> str:
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
    output_ids = model.generate(**inputs, max_new_tokens=max(256, config.VLM_MAX_TOKENS), do_sample=False)
    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


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
        + "\nUse it to produce a less generic and more clip-specific result."
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


def _normalize_generated_code(code: str) -> str:
    if not code:
        return code

    normalized = textwrap.dedent(code).strip()

    for prefix in ('nao.', 'robot.', 'agent.'):
        normalized = normalized.replace(prefix, '')

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
        code = _normalize_generated_code(code)
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
                code = _normalize_generated_code(code)
                if not _candidate_has_minimal_structure(semantic, code):
                    if config.LOCAL_VLM_DEBUG:
                        print(f'[VLMClient][local] candidate rejected: parse/min-structure failed semantic={semantic}')
                    continue
                parsed_candidates.append((semantic, code, raw))
                validation = self.static_validator.validate(code)
                if config.LOCAL_VLM_DEBUG:
                    print(f'[VLMClient][local] candidate valid={validation.ok} semantic={semantic}')
                if validation.ok:
                    valid_candidates.append((semantic, code, raw))

            pool = valid_candidates or parsed_candidates
            if not pool:
                return VLMResponse({}, '', raw_candidates[0] if raw_candidates else '', time.time() - t0, False, error='parse_incomplete')

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
                if chosen_index is not None and 1 <= chosen_index <= len(pool):
                    best_semantic, best_code, best_raw = pool[chosen_index - 1]
                else:
                    best_semantic, best_code, best_raw = pool[0]

            if best_code and _looks_too_generic_for_clip(best_semantic, best_code):
                refinement_prompt = _build_refinement_prompt_with_summary(
                    frames_b64,
                    best_semantic,
                    best_code,
                    'candidate_too_generic_for_this_clip',
                )
                if kind == 'qwen2_5_vl':
                    refined_raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, refinement_prompt)
                else:
                    refined_raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, refinement_prompt)
                refined_semantic, refined_code = parse_vlm_output(refined_raw)
                refined_code = _normalize_generated_code(refined_code)
                refined_validation = self.static_validator.validate(refined_code)
                if config.LOCAL_VLM_DEBUG:
                    print(f'[VLMClient][local] refinement valid={refined_validation.ok} semantic={refined_semantic}')
                if _candidate_has_minimal_structure(refined_semantic, refined_code) and refined_validation.ok:
                    best_semantic, best_code, best_raw = refined_semantic, refined_code, refined_raw

            if (not best_code or not best_semantic) and parsed_candidates:
                fallback_semantic, fallback_code, _ = parsed_candidates[0]
                parse_repair_prompt = _build_repair_prompt_with_summary(
                    frames_b64,
                    fallback_semantic,
                    fallback_code,
                    'candidate_parse_or_selection_failed',
                )
                if kind == 'qwen2_5_vl':
                    repaired_raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, parse_repair_prompt)
                else:
                    repaired_raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, parse_repair_prompt)
                repaired_semantic, repaired_code = parse_vlm_output(repaired_raw)
                repaired_code = _normalize_generated_code(repaired_code)
                repaired_validation = self.static_validator.validate(repaired_code)
                if _candidate_has_minimal_structure(repaired_semantic, repaired_code) and repaired_validation.ok:
                    best_semantic, best_code, best_raw = repaired_semantic, repaired_code, repaired_raw

            ok = bool(best_code) and bool(best_semantic)
            return VLMResponse(
                semantic_context=best_semantic,
                python_code=best_code,
                raw_text=best_raw,
                elapsed_seconds=time.time() - t0,
                ok=ok,
                error=None if ok else 'parse_incomplete',
            )
        except Exception as exc:
            if _is_probable_cuda_oom(exc):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    processor, model, kind = _load_local_model_on_cpu(self.model)
                    images = _decode_frames_to_pil(frames_b64)
                    summary = _infer_visual_motion_summary(frames_b64)
                    user_prompt = _build_local_user_prompt(frames_b64)
                    if kind == 'qwen2_5_vl':
                        raw = _qwen_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=False)
                    else:
                        raw = _smolvlm_multi_image_generate(processor, model, images, self.system_prompt, user_prompt, do_sample=False)
                    semantic, code = parse_vlm_output(raw)
                    code = _normalize_generated_code(code)
                    if (not _candidate_has_minimal_structure(semantic, code)) or (not self.static_validator.validate(code).ok):
                        repair_prompt = _build_repair_prompt_with_summary(frames_b64, semantic, code, 'cpu_fallback_repair')
                        if kind == 'qwen2_5_vl':
                            raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)
                        else:
                            raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)
                        semantic, code = parse_vlm_output(raw)
                        code = _normalize_generated_code(code)
                    ok = bool(code) and bool(semantic)
                    return VLMResponse(
                        semantic_context=semantic,
                        python_code=code,
                        raw_text=raw,
                        elapsed_seconds=time.time() - t0,
                        ok=ok,
                        error=None if ok else 'parse_incomplete_cpu_fallback',
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
            repair_prompt = _build_repair_prompt_with_summary(frames_b64, semantic_context, python_code, error_text)
            if kind == 'qwen2_5_vl':
                raw = _qwen_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)
            else:
                raw = _smolvlm_multi_image_generate_with_prompt(processor, model, images, self.system_prompt, repair_prompt)

            semantic, code = parse_vlm_output(raw)
            code = _normalize_generated_code(code)
            ok = bool(code) and bool(semantic)
            return VLMResponse(
                semantic_context=semantic,
                python_code=code,
                raw_text=raw,
                elapsed_seconds=time.time() - t0,
                ok=ok,
                error=None if ok else 'parse_incomplete',
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
