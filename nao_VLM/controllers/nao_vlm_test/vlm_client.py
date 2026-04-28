"""
VLM client — supports both remote OpenAI-compatible backends and a local
multi-image backend for desktop demos.

Remote backend:
  - preserves the original contract
  - frames -> semantic JSON + primitive Python code

Local backend:
  - uses a local VLM to classify the human communicative signal
  - converts that VLM decision into a pet-like, socially responsive robot
    action sequence rather than directly mimicking the human
"""
from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

import config


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

# NAO coordinate convention
- Torso-centred, right-hand frame. Units are METRES.
- x forward, y LEFT is positive, z up from torso.
- Neutral arm: right hand ≈ [0.02, -0.10, -0.20]. Typical waving pose:
  right hand raised to roughly [0.15, -0.15, 0.10].

# Joint reference (radian limits)
{joint_limit_block}

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

If the clip looks like a person greeting or trying to engage the robot, prefer
a clear and legible response that raises one arm and performs a brief wave.
Keep the full motion sequence short, stable, and visually obvious in Webots.
"""

_LOCAL_RESPONSE_PROMPT = """You are choosing how a friendly pet-like humanoid robot should respond to a person.
Observe the chronological frames and choose exactly one response label.
Return only one label from this list:
PET_GREET_HAPPY
PET_ORIENT_FOLLOW
PET_APPROACH_CURIOUS
PET_FREEZE_RESPECTFUL
PET_SOFT_BACKOFF
PET_EXCITED_ACK
PET_CONFUSED_HEAD_TILT
PET_WATCH_WAIT

Meanings:
- PET_GREET_HAPPY: the human greets, waves, or raises a hand to engage.
- PET_ORIENT_FOLLOW: the human points to a direction or object.
- PET_APPROACH_CURIOUS: the human beckons the robot to come closer.
- PET_FREEZE_RESPECTFUL: the human clearly shows a stop palm.
- PET_SOFT_BACKOFF: the human rejects, says no, shakes head no, or wags a finger no-no.
- PET_EXCITED_ACK: the human shows positive approval, claps, gives thumbs-up, or nods yes.
- PET_CONFUSED_HEAD_TILT: the human shrugs or seems unsure.
- PET_WATCH_WAIT: the signal is unclear.

Disambiguation rules:
- Open palm facing the robot means PET_FREEZE_RESPECTFUL, not greeting.
- Repeated hand wave toward self means PET_APPROACH_CURIOUS.
- Extended arm or index finger indicating a direction/object means PET_ORIENT_FOLLOW.
- Thumbs-up, clapping, or a clear yes nod means PET_EXCITED_ACK.
- Head shaking no or finger wagging no-no means PET_SOFT_BACKOFF.
- Shrugging with both shoulders/hands lifted means PET_CONFUSED_HEAD_TILT.
- Greeting wave is a side-to-side hello wave to the robot, not a stop palm.

Choose based on the full sequence, not a single frame. Output only one label."""

_LOCAL_LABELS = {
    'PET_GREET_HAPPY',
    'PET_ORIENT_FOLLOW',
    'PET_APPROACH_CURIOUS',
    'PET_FREEZE_RESPECTFUL',
    'PET_SOFT_BACKOFF',
    'PET_EXCITED_ACK',
    'PET_CONFUSED_HEAD_TILT',
    'PET_WATCH_WAIT',
}

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


def _extract_label(raw: str) -> str:
    text = ' '.join((raw or '').strip().split())
    for label in _LOCAL_LABELS:
        if label in text:
            return label
    return 'PET_WATCH_WAIT'


def _compile_local_semantics(label: str) -> Dict[str, Any]:
    semantic = {
        'intent': label,
        'social_distance': 'medium',
        'affect': 'neutral',
        'confidence': 0.7,
        'motion_dynamics': 'static',
        'response_label': label,
        'backend': 'local',
    }

    if label == 'PET_GREET_HAPPY':
        semantic.update({'intent': 'human greeting or raised-hand engagement', 'affect': 'friendly', 'confidence': 0.9, 'motion_dynamics': 'raising'})
    elif label == 'PET_ORIENT_FOLLOW':
        semantic.update({'intent': 'human pointing to a direction or object', 'affect': 'curious', 'confidence': 0.82, 'motion_dynamics': 'static'})
    elif label == 'PET_APPROACH_CURIOUS':
        semantic.update({'intent': 'human beckoning the robot closer', 'affect': 'curious', 'confidence': 0.8, 'motion_dynamics': 'approaching'})
    elif label == 'PET_FREEZE_RESPECTFUL':
        semantic.update({'intent': 'human stop signal', 'affect': 'submissive', 'confidence': 0.84, 'motion_dynamics': 'static'})
    elif label == 'PET_SOFT_BACKOFF':
        semantic.update({'intent': 'human rejection or no signal', 'affect': 'cautious', 'confidence': 0.8, 'motion_dynamics': 'retreating'})
    elif label == 'PET_EXCITED_ACK':
        semantic.update({'intent': 'human approval or positive feedback', 'affect': 'happy', 'confidence': 0.86, 'motion_dynamics': 'oscillatory'})
    elif label == 'PET_CONFUSED_HEAD_TILT':
        semantic.update({'intent': 'human uncertainty or shrug', 'affect': 'curious', 'confidence': 0.78, 'motion_dynamics': 'static'})
    return semantic


def _compile_local_code(label: str) -> str:
    if label == 'PET_GREET_HAPPY':
        return """
operate_gripper('right', 'open')
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': -0.10,
    'LShoulderPitch': 1.36,
    'RShoulderPitch': 0.34,
    'RShoulderRoll': -0.50,
    'RElbowYaw': 1.18,
    'RElbowRoll': 0.98,
    'RWristYaw': 0.20,
}, duration=0.85, trajectory='min_jerk')
hold(0.10)
for target in (-0.22, -0.72, -0.22, -0.66, -0.32):
    move_joint('RShoulderRoll', target, duration=0.18, trajectory='min_jerk')
move_joint('RWristYaw', -0.12, duration=0.16, trajectory='min_jerk')
move_joint('RWristYaw', 0.24, duration=0.16, trajectory='min_jerk')
operate_gripper('right', 'close')
move_joints({
    'HeadPitch': 0.0,
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'RShoulderRoll': -0.15,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.50,
    'RWristYaw': 0.0,
}, duration=0.82, trajectory='min_jerk')
""".strip()

    if label == 'PET_ORIENT_FOLLOW':
        return """
operate_gripper('right', 'open')
move_joints({
    'HeadYaw': 0.42,
    'HeadPitch': -0.10,
    'RShoulderPitch': 0.88,
    'RShoulderRoll': -0.16,
    'RElbowYaw': 1.10,
    'RElbowRoll': 0.30,
    'RWristYaw': -0.20,
}, duration=0.72, trajectory='min_jerk')
hold(0.70)
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': 0.0,
    'RShoulderPitch': 1.50,
    'RShoulderRoll': -0.15,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.50,
    'RWristYaw': 0.0,
}, duration=0.84, trajectory='min_jerk')
operate_gripper('right', 'close')
""".strip()

    if label == 'PET_APPROACH_CURIOUS':
        return """
operate_gripper('left', 'open')
operate_gripper('right', 'open')
move_joints({
    'HeadPitch': -0.16,
    'LShoulderPitch': 1.16,
    'RShoulderPitch': 1.16,
    'LElbowRoll': -0.68,
    'RElbowRoll': 0.68,
}, duration=0.52, trajectory='min_jerk')
navigate_to(0.08, 0.0, 0.0)
move_joint('HeadYaw', 0.12, duration=0.22, trajectory='min_jerk')
move_joint('HeadYaw', -0.10, duration=0.22, trajectory='min_jerk')
move_joint('HeadYaw', 0.0, duration=0.22, trajectory='min_jerk')
hold(0.30)
move_joints({
    'HeadPitch': 0.0,
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'LElbowRoll': -0.50,
    'RElbowRoll': 0.50,
}, duration=0.70, trajectory='min_jerk')
operate_gripper('left', 'close')
operate_gripper('right', 'close')
""".strip()

    if label == 'PET_FREEZE_RESPECTFUL':
        return """
move_joints({
    'HeadPitch': 0.16,
    'LShoulderPitch': 1.62,
    'RShoulderPitch': 1.62,
    'LElbowRoll': -0.36,
    'RElbowRoll': 0.36,
    'LHipPitch': -0.10,
    'RHipPitch': -0.10,
    'LKneePitch': 0.22,
    'RKneePitch': 0.22,
    'LAnklePitch': -0.08,
    'RAnklePitch': -0.08,
}, duration=0.45, trajectory='min_jerk')
hold(0.90)
move_joints({
    'HeadPitch': 0.0,
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'LElbowRoll': -0.50,
    'RElbowRoll': 0.50,
    'LHipPitch': 0.0,
    'RHipPitch': 0.0,
    'LKneePitch': 0.0,
    'RKneePitch': 0.0,
    'LAnklePitch': 0.0,
    'RAnklePitch': 0.0,
}, duration=0.62, trajectory='min_jerk')
""".strip()

    if label == 'PET_SOFT_BACKOFF':
        return """
move_joints({
    'HeadYaw': -0.15,
    'HeadPitch': 0.12,
    'LShoulderPitch': 1.58,
    'RShoulderPitch': 1.58,
    'LHipPitch': -0.06,
    'RHipPitch': -0.06,
    'LKneePitch': 0.14,
    'RKneePitch': 0.14,
}, duration=0.50, trajectory='min_jerk')
navigate_to(-0.06, 0.0, 0.0)
hold(0.30)
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': 0.0,
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'LHipPitch': 0.0,
    'RHipPitch': 0.0,
    'LKneePitch': 0.0,
    'RKneePitch': 0.0,
}, duration=0.55, trajectory='min_jerk')
""".strip()

    if label == 'PET_EXCITED_ACK':
        return """
move_joints({
    'HeadPitch': -0.10,
    'LShoulderPitch': 1.08,
    'RShoulderPitch': 1.08,
    'LShoulderRoll': 0.26,
    'RShoulderRoll': -0.26,
    'LHipPitch': -0.08,
    'RHipPitch': -0.08,
    'LKneePitch': 0.18,
    'RKneePitch': 0.18,
}, duration=0.42, trajectory='min_jerk')
move_joint('HeadPitch', 0.04, duration=0.16, trajectory='min_jerk')
move_joint('HeadPitch', -0.10, duration=0.16, trajectory='min_jerk')
move_joint('HeadPitch', 0.04, duration=0.16, trajectory='min_jerk')
move_joints({
    'LShoulderPitch': 0.92,
    'RShoulderPitch': 0.92,
    'LKneePitch': 0.28,
    'RKneePitch': 0.28,
}, duration=0.18, trajectory='min_jerk')
move_joints({
    'LShoulderPitch': 1.15,
    'RShoulderPitch': 1.15,
    'LKneePitch': 0.12,
    'RKneePitch': 0.12,
}, duration=0.18, trajectory='min_jerk')
move_joints({
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'HeadPitch': 0.0,
    'LShoulderRoll': 0.15,
    'RShoulderRoll': -0.15,
    'LHipPitch': 0.0,
    'RHipPitch': 0.0,
    'LKneePitch': 0.0,
    'RKneePitch': 0.0,
}, duration=0.48, trajectory='min_jerk')
""".strip()

    if label == 'PET_CONFUSED_HEAD_TILT':
        return """
operate_gripper('left', 'open')
operate_gripper('right', 'open')
move_joints({
    'HeadYaw': 0.22,
    'HeadPitch': -0.04,
    'LShoulderPitch': 1.30,
    'RShoulderPitch': 1.30,
    'LShoulderRoll': 0.34,
    'RShoulderRoll': -0.34,
    'LElbowYaw': -0.82,
    'RElbowYaw': 0.82,
    'LElbowRoll': -0.92,
    'RElbowRoll': 0.92,
}, duration=0.45, trajectory='min_jerk')
hold(0.35)
move_joint('HeadYaw', -0.18, duration=0.36, trajectory='min_jerk')
hold(0.25)
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': 0.0,
    'LShoulderPitch': 1.50,
    'RShoulderPitch': 1.50,
    'LShoulderRoll': 0.15,
    'RShoulderRoll': -0.15,
    'LElbowYaw': -1.20,
    'RElbowYaw': 1.20,
    'LElbowRoll': -0.50,
    'RElbowRoll': 0.50,
}, duration=0.52, trajectory='min_jerk')
operate_gripper('left', 'close')
operate_gripper('right', 'close')
""".strip()

    return "idle(1.00)"


def _compile_local_response(label: str) -> Tuple[Dict[str, Any], str]:
    semantic = _compile_local_semantics(label)
    code = _compile_local_code(label)
    return semantic, code


def _qwen_multi_image_classify(processor, model, images) -> str:
    messages = [{
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': _LOCAL_RESPONSE_PROMPT},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors='pt', padding=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], output_ids)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return raw


def _smolvlm_multi_image_classify(processor, model, images) -> str:
    conversation = [{
        'role': 'user',
        'content': [
            *({'type': 'image', 'image': image} for image in images),
            {'type': 'text', 'text': _LOCAL_RESPONSE_PROMPT},
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
    output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return raw


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
        self.system_prompt = build_system_prompt(joint_limits)
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

            if kind == 'qwen2_5_vl':
                raw_label_text = _qwen_multi_image_classify(processor, model, images)
            else:
                raw_label_text = _smolvlm_multi_image_classify(processor, model, images)

            label = _extract_label(raw_label_text)
            semantic, code = _compile_local_response(label)
            raw = json.dumps(
                {
                    'backend': 'local',
                    'model': self.model,
                    'raw_label_text': raw_label_text,
                    'response_label': label,
                },
                ensure_ascii=False,
                indent=2,
            )
            return VLMResponse(semantic, code, raw, time.time() - t0, True)
        except Exception as exc:
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(exc))

    def _call_local_server(self, frames_b64: Sequence[str]) -> VLMResponse:
        t0 = time.time()
        try:
            rsp = requests.post(
                config.LOCAL_VLM_SERVER_URL.rstrip('/') + '/classify_frames',
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


_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_PYTHON_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


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
        code = match_python.group(1).strip()

    return semantic, code
