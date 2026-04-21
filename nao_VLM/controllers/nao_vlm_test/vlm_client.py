"""
VLM client — multi-image call to GPT-4o (or compatible) with a temporal prompt.

Returns a parsed result: (semantic_context_dict, python_code_str).

The prompt is built to match the Motion Grammar primitive API (see plan §7).
Joint limits are injected at construction time so the VLM sees the exact
constraints of the NAO it is actually controlling (rather than a stale list).

Phase 0 note: this module implements the "request" side. The `executor` side
(sandbox) lives in sandbox_exec.py. Neither imports the other; the controller
wires them together.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai SDK is required. Install with `pip install openai`."
    ) from e

import config


# ---------------------------------------------------------------------------
# Prompt building blocks
# ---------------------------------------------------------------------------

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
        # Smoothly move a single joint to `angle` radians over `duration` s.

    move_joints(joint_angles: dict, duration: float,
                trajectory: str = 'cubic')
        # Move multiple joints in parallel, each to its target angle.

    move_arm_ik(side: str, xyz: list, duration: float)
        # Cartesian IK: move the `side` hand ('left'|'right') to position
        # xyz=[x,y,z] metres in NAO's torso frame.

    oscillate_joint(name: str, center: float, amplitude: float,
                    frequency: float, duration: float, decay: float = 0.0)
        # Sinusoidal oscillation around `center` radians. Amplitude in radians,
        # frequency in Hz. If decay>0, amplitude decays exponentially with
        # time constant (duration / decay).

    hold(duration: float)
        # Keep the current pose for `duration` seconds.

    idle(duration: float)
        # Subtle breathing/scan motion; use ONLY as a neutral default.

    speak(text: str)
        # Brief spoken reply (string kept short).

# NAO coordinate convention
- Torso-centred, right-hand frame. Units are METRES.
- x forward, y LEFT is positive (right hand rests near y ≈ -0.10),
  z up from torso (hand at rest near z ≈ -0.10).
- Neutral arm: right hand ≈ [0.02, -0.10, -0.20]. Typical waving pose:
  right hand raised to roughly [0.15, -0.15, 0.10].

# Joint reference (radian limits)
{joint_limit_block}

# Example: a "decaying wave" that adapts amplitude to social distance
    move_arm_ik('right', xyz=[0.15, -0.15, 0.10], duration=0.4)
    oscillate_joint('RElbowRoll', center=1.0, amplitude=0.6,
                    frequency=2.0, duration=2.0, decay=0.5)
    move_arm_ik('right', xyz=[0.02, -0.10, -0.20], duration=0.4)

The `amplitude=0.6` above would be LARGER (say 0.9) for a far-away human and
SMALLER (say 0.3) for a close one. You choose these numbers using physical
intuition.

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
# use only the primitives listed above
```

Do NOT output any prose outside the two fenced blocks.
"""

_USER_PROMPT = (
    "Analyse the motion across the frames below and generate the required "
    "JSON context and Python primitive composition."
)


def build_system_prompt(joint_limits: Dict[str, Tuple[float, float]]) -> str:
    """
    Format the base system prompt with concrete joint limits.

    `joint_limits` maps joint name -> (min_rad, max_rad). Joints with
    min==max are treated as fixed and omitted.
    """
    rows = []
    for name in sorted(joint_limits):
        lo, hi = joint_limits[name]
        if lo == hi:
            continue
        rows.append(f"  - {name}: [{lo:+.3f}, {hi:+.3f}]")
    block = "\n".join(rows) if rows else "  (no joint limits provided)"

    return _BASE_SYSTEM_PROMPT.format(
        frame_count=config.VLM_FRAME_COUNT,
        window_seconds=config.VLM_WINDOW_SECONDS,
        joint_limit_block=block,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VLMResponse:
    semantic_context: Dict[str, Any]
    python_code: str
    raw_text: str
    elapsed_seconds: float
    ok: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class VLMClient:
    def __init__(
        self,
        joint_limits: Dict[str, Tuple[float, float]],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL or None
        self.model = model or config.VLM_MODEL

        if not self.api_key:
            raise RuntimeError(
                "VLM API key missing. Set `llm_api_key` env var or pass api_key."
            )

        # OpenAI SDK accepts base_url=None to use the default endpoint
        client_kwargs = {'api_key': self.api_key}
        if self.base_url:
            client_kwargs['base_url'] = self.base_url
        self.client = OpenAI(**client_kwargs)

        self.system_prompt = build_system_prompt(joint_limits)

    # ------------------------------------------------------------------ call

    def call(self, frames_b64: Sequence[str]) -> VLMResponse:
        """
        Send `frames_b64` (list of base64-encoded JPEGs) to the VLM and
        return a parsed VLMResponse.

        This method is blocking and may take several seconds. Call it from a
        background thread so the main Webots control loop is not stalled.
        """
        if not frames_b64:
            return VLMResponse({}, '', '', 0.0, False, error='no frames provided')

        content: List[Dict[str, Any]] = [
            {"type": "text", "text": _USER_PROMPT}
        ]
        for img_b64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}",
                    "detail": config.VLM_IMAGE_DETAIL,
                },
            })

        t0 = time.time()
        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                max_tokens=config.VLM_MAX_TOKENS,
                temperature=config.VLM_TEMPERATURE,
            )
            raw = rsp.choices[0].message.content or ''
            elapsed = time.time() - t0
        except Exception as e:
            return VLMResponse({}, '', '', time.time() - t0, False, error=str(e))

        semantic, code = parse_vlm_output(raw)
        ok = bool(code) and bool(semantic)
        return VLMResponse(semantic, code, raw, elapsed, ok,
                           error=None if ok else 'parse_incomplete')


# ---------------------------------------------------------------------------
# Parser (re-entrant, no hidden state)
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_PYTHON_RE = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_vlm_output(raw_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract the JSON semantic block and the Python control block from a
    VLM response. Missing blocks yield empty defaults rather than raising,
    so the caller can still invoke the fallback policy.
    """
    semantic: Dict[str, Any] = {}
    code: str = ''

    mj = _JSON_RE.search(raw_text or '')
    if mj:
        try:
            semantic = json.loads(mj.group(1).strip())
        except json.JSONDecodeError as e:
            print(f"[VLMClient] JSON parse error: {e}")

    mp = _PYTHON_RE.search(raw_text or '')
    if mp:
        code = mp.group(1).strip()

    return semantic, code
