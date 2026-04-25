"""
VLM client — supports both remote OpenAI-compatible backends and a local
multi-image backend for desktop demos.

The OpenAI path preserves the original contract: image sequence in, semantic
JSON + Python primitive composition out.

The local path is tuned for the current desktop demo: it first asks a small
video-language model to summarize the most salient human gesture across a short
frame sequence, then compiles that semantic label into a robot motion program.
This keeps the response VLM-driven while remaining reliable on a single GPU.
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

# Motion grammar (physical primitives)
You construct behavior, you do NOT select named actions. There is NO "wave()"
or "greet()" function. You must COMPOSE motion from these primitives:

    move_joint(name: str, angle: float, duration: float,
               trajectory: str = 'cubic')

    move_joints(joint_angles: dict, duration: float,
                trajectory: str = 'cubic')

    move_arm_ik(side: str, xyz: list, duration: float)

    oscillate_joint(name: str, center: float, amplitude: float,
                    frequency: float, duration: float, decay: float = 0.0)

    hold(duration: float)

    idle(duration: float)

    speak(text: str)

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

_LOCAL_DESCRIPTION_PROMPT = (
    "Observe these chronological webcam frames of a human interacting with a "
    "humanoid robot. Describe only the most salient human upper-body gesture "
    "or interaction cue in one short sentence. Focus on arm, hand, greeting, "
    "or engagement motion. If no clear gesture is visible, say: no clear gesture."
)

_LOCAL_STRONG_WAVE_PATTERNS = (
    r"\bwave\b",
    r"\bwaving\b",
    r"hand gesture",
    r"palm facing outward",
    r"greet",
    r"greeting",
    r"hello",
    r"raised hand",
    r"raising (?:an )?arm",
    r"raised (?:an )?arm",
    r"hand up",
)

_LOCAL_ARM_PATTERNS = (
    r"\barm\b",
    r"\bhand\b",
    r"gesture",
    r"upper-body",
    r"reaching",
    r"pointing",
)

_LOCAL_PRESENT_PATTERNS = (
    r"\bperson\b",
    r"\bhuman\b",
    r"\bman\b",
    r"\bwoman\b",
    r"standing",
)

_LOCAL_MODEL_LOCK = Lock()
_LOCAL_MODEL_ID: Optional[str] = None
_LOCAL_PROCESSOR = None
_LOCAL_MODEL = None


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


def _load_local_model(model_id: str):
    global _LOCAL_MODEL_ID, _LOCAL_PROCESSOR, _LOCAL_MODEL

    if _LOCAL_MODEL_ID == model_id and _LOCAL_PROCESSOR is not None and _LOCAL_MODEL is not None:
        return _LOCAL_PROCESSOR, _LOCAL_MODEL

    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL_ID == model_id and _LOCAL_PROCESSOR is not None and _LOCAL_MODEL is not None:
            return _LOCAL_PROCESSOR, _LOCAL_MODEL

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs: Dict[str, Any] = {'dtype': dtype}
        if torch.cuda.is_available():
            model_kwargs['device_map'] = 'auto'

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        if not torch.cuda.is_available():
            model = model.to('cpu')
        model.eval()

        _LOCAL_MODEL_ID = model_id
        _LOCAL_PROCESSOR = processor
        _LOCAL_MODEL = model
        return processor, model


def _extract_first_sentence(text: str) -> str:
    cleaned = ' '.join((text or '').strip().split())
    if not cleaned:
        return 'no clear gesture'
    parts = re.split(r'(?<=[.!?])\s+', cleaned, maxsplit=1)
    sentence = parts[0].strip()
    return sentence[:220] if sentence else 'no clear gesture'


def _match_any(patterns: Sequence[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _infer_local_label(summary: str) -> Tuple[str, float]:
    lowered = summary.lower()
    if 'no clear gesture' in lowered:
        return 'stay_idle', 0.55
    if _match_any(_LOCAL_STRONG_WAVE_PATTERNS, lowered):
        return 'wave_back', 0.86
    if _match_any(_LOCAL_ARM_PATTERNS, lowered):
        return 'acknowledge_raise_arm', 0.74
    if _match_any(_LOCAL_PRESENT_PATTERNS, lowered):
        return 'look_at_human', 0.62
    return 'stay_idle', 0.50


def _compile_local_semantics(label: str, summary: str, confidence: float) -> Dict[str, Any]:
    motion = 'static'
    affect = 'neutral'
    if label in {'wave_back', 'acknowledge_raise_arm'}:
        motion = 'raising'
        affect = 'friendly'
    elif label == 'look_at_human':
        affect = 'curious'

    return {
        'intent': summary,
        'social_distance': 'medium',
        'affect': affect,
        'confidence': round(float(confidence), 3),
        'motion_dynamics': motion,
        'response_label': label,
        'backend': 'local',
    }


def _compile_local_code(label: str) -> str:
    if label == 'wave_back':
        return """
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': -0.10,
    'RShoulderPitch': 0.30,
    'RShoulderRoll': -0.55,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.95,
    'RWristYaw': 0.10,
}, duration=0.90)
hold(0.15)
for target in (-0.35, -0.80, -0.35, -0.80, -0.55):
    move_joint('RShoulderRoll', target, duration=0.18)
for target in (-0.35, 0.45, -0.25, 0.25, 0.0):
    move_joint('RWristYaw', target, duration=0.14)
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': 0.0,
    'RShoulderPitch': 1.50,
    'RShoulderRoll': -0.15,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.50,
    'RWristYaw': 0.0,
}, duration=0.85)
""".strip()

    if label == 'acknowledge_raise_arm':
        return """
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': -0.08,
    'RShoulderPitch': 0.55,
    'RShoulderRoll': -0.35,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.80,
    'RWristYaw': 0.05,
}, duration=0.75)
hold(0.60)
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': 0.0,
    'RShoulderPitch': 1.50,
    'RShoulderRoll': -0.15,
    'RElbowYaw': 1.20,
    'RElbowRoll': 0.50,
    'RWristYaw': 0.0,
}, duration=0.75)
""".strip()

    if label == 'look_at_human':
        return """
move_joints({
    'HeadYaw': 0.0,
    'HeadPitch': -0.12,
}, duration=0.35)
hold(0.50)
move_joint('HeadPitch', 0.04, duration=0.18)
move_joint('HeadPitch', -0.08, duration=0.18)
move_joint('HeadPitch', 0.0, duration=0.18)
idle(0.60)
""".strip()

    return "idle(1.20)"


def _compile_local_response(summary: str) -> Tuple[Dict[str, Any], str, str]:
    label, confidence = _infer_local_label(summary)
    semantic = _compile_local_semantics(label, summary, confidence)
    code = _compile_local_code(label)
    return semantic, code, label


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
            self.model = config.LOCAL_VLM_MODEL

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

        t0 = time.time()
        try:
            images = _decode_frames_to_pil(frames_b64)
            processor, model = _load_local_model(self.model)

            conversation = [{
                'role': 'user',
                'content': [
                    *({'type': 'image', 'image': image} for image in images),
                    {'type': 'text', 'text': _LOCAL_DESCRIPTION_PROMPT},
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
                max_new_tokens=40,
                do_sample=False,
            )
            new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
            description = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            summary = _extract_first_sentence(description)
            semantic, code, label = _compile_local_response(summary)
            raw = json.dumps(
                {
                    'backend': 'local',
                    'model': self.model,
                    'gesture_summary': summary,
                    'response_label': label,
                },
                ensure_ascii=False,
                indent=2,
            )
            elapsed = time.time() - t0
            return VLMResponse(semantic, code, raw, elapsed, True)
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
