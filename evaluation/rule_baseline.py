from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CTRL_DIR = REPO_ROOT / 'nao_VLM' / 'controllers' / 'nao_vlm_test'
if str(CTRL_DIR) not in sys.path:
    sys.path.insert(0, str(CTRL_DIR))

import config
from vlm_client import VLMResponse

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


INTENT_TO_CODE: Dict[str, str] = {
    'wave': """move_arm_ik('right', [0.15, -0.15, 0.10], 0.4)
oscillate_joint('RElbowRoll', 1.0, 0.45, 2.0, 1.2, 0.3)
move_arm_ik('right', [0.02, -0.10, -0.20], 0.4)
hold(0.3)""",
    'reject': """move_head(-0.18, -0.03, 0.35)
move_arm_ik('right', [0.10, -0.10, 0.04], 0.45)
set_hand('right', 0.75, 0.25)
hold(0.5)""",
    'approach': """move_joints({'LShoulderPitch': 1.6, 'RShoulderPitch': 1.6, 'HeadPitch': -0.05}, 0.5, 'min_jerk')
move_arm_ik('right', [0.12, -0.10, 0.05], 0.4)
set_hand('right', 0.7, 0.3)
hold(0.5)""",
    'walk_away': """move_head(0.16, -0.02, 0.35)
hold(0.4)
move_head(-0.10, 0.00, 0.35)
idle(0.5)""",
    'crouch': """move_head(0.0, 0.22, 0.45)
hold(0.6)
move_head(0.0, 0.02, 0.35)""",
    'greet': """move_arm_ik('right', [0.13, -0.14, 0.08], 0.4)
set_hand('right', 0.8, 0.25)
hold(0.4)
move_arm_ik('right', [0.02, -0.10, -0.20], 0.4)""",
    'idle': """move_head(0.16, -0.04, 0.4)
hold(0.5)
idle(0.5)""",
    'point': """move_head(-0.18, -0.02, 0.35)
move_arm_ik('right', [0.12, -0.14, 0.02], 0.5)
hold(0.5)""",
    'stop': """move_arm_ik('right', [0.11, -0.08, 0.08], 0.45)
set_hand('right', 0.85, 0.25)
hold(0.6)""",
    'approval': """move_head(0.10, -0.04, 0.3)
move_arm_ik('left', [0.08, 0.12, 0.06], 0.45)
set_hand('left', 0.65, 0.25)
hold(0.5)""",
    'agreement': """move_head(0.0, 0.18, 0.25)
move_head(0.0, -0.08, 0.25)
hold(0.4)""",
    'disagreement': """move_head(0.22, -0.02, 0.25)
move_head(-0.22, -0.02, 0.25)
hold(0.4)""",
}

SCENARIO_TO_INTENT = {
    'wave': 'wave',
    'waving': 'wave',
    'finger_no': 'reject',
    'stop': 'stop',
    'thumbs_up': 'approval',
    'yes_nod': 'agreement',
    'no_shake': 'disagreement',
    'pointing': 'point',
    'beckon': 'greet',
    'clap': 'approval',
    'shrug': 'idle',
    'lean_forward': 'approach',
    'walk_away': 'walk_away',
    'crouch': 'crouch',
    'cross_arms': 'reject',
    'reject_gesture': 'reject',
    'greet_handshake': 'greet',
    'idle_standing': 'idle',
}


def code_for_intent(intent: str) -> str:
    return INTENT_TO_CODE.get(intent, INTENT_TO_CODE['idle'])


def infer_label_from_scenario_id(scenario_id: str) -> str:
    lowered = (scenario_id or '').lower()
    for key, label in SCENARIO_TO_INTENT.items():
        if key in lowered:
            return label
    return 'idle'


class RuleBaselineClient:
    """Intent-label baseline: VLM/classifier label -> fixed primitive code."""

    def __init__(self, joint_limits=None, api_key=None, base_url=None, model=None) -> None:
        self.joint_limits = dict(joint_limits or {})
        self.api_key = api_key or config.LLM_API_KEY or None
        self.base_url = base_url or config.LLM_BASE_URL or None
        self.model = model or os.getenv('RULE_BASELINE_MODEL', config.VLM_MODEL)
        self.client = None
        if OpenAI is not None and self.api_key:
            kwargs: Dict[str, Any] = {'api_key': self.api_key}
            if self.base_url:
                kwargs['base_url'] = self.base_url
            self.client = OpenAI(**kwargs)

    def _heuristic_response(self, frames_b64: Sequence[str], reason: str) -> VLMResponse:
        label = infer_label_from_scenario_id(config.EVAL_SCENARIO_ID or config.VLM_SCENARIO_HINT)
        semantic = {
            'intent': f'rule-baseline label={label}',
            'social_distance': 'medium',
            'affect': 'neutral',
            'confidence': 0.35,
            'motion_dynamics': 'static',
            'robot_intent': f'fixed baseline response for {label}',
            'intent_label': label,
        }
        raw = json.dumps({'fallback_reason': reason, 'intent_label': label})
        return VLMResponse(semantic, code_for_intent(label), raw, 0.0, True)

    def call(self, frames_b64: Sequence[str]) -> VLMResponse:
        if not frames_b64:
            return VLMResponse({}, '', '', 0.0, False, error='no frames provided')
        if self.client is None:
            return self._heuristic_response(frames_b64, 'openai_unavailable')

        labels = sorted(INTENT_TO_CODE)
        content = [{
            'type': 'text',
            'text': (
                'Classify the human gesture into exactly one intent_label from '
                f'{labels}. Return only JSON: {"{"}"intent_label": "...", '
                '"confidence": 0.0, "reason": "short"}.'
            ),
        }]
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
                    {'role': 'system', 'content': 'You are an intent-only rule-baseline classifier for a robot benchmark.'},
                    {'role': 'user', 'content': content},
                ],
                temperature=0.0,
                max_tokens=120,
            )
            raw = rsp.choices[0].message.content or ''
        except Exception as exc:
            return self._heuristic_response(frames_b64, f'openai_error:{exc}')

        try:
            cleaned = raw.strip()
            if cleaned.startswith('```'):
                cleaned = cleaned.strip('`')
                cleaned = cleaned.replace('json', '', 1).strip()
            payload = json.loads(cleaned)
        except Exception:
            payload = {}
        label = str(payload.get('intent_label') or '').strip().lower()
        if label not in INTENT_TO_CODE:
            label = infer_label_from_scenario_id(config.EVAL_SCENARIO_ID)
        semantic = {
            'intent': f'rule-baseline label={label}',
            'social_distance': 'medium',
            'affect': 'neutral',
            'confidence': float(payload.get('confidence') or 0.0),
            'motion_dynamics': 'static',
            'robot_intent': f'fixed baseline response for {label}',
            'intent_label': label,
            'classifier_reason': payload.get('reason', ''),
        }
        return VLMResponse(
            semantic_context=semantic,
            python_code=code_for_intent(label),
            raw_text=raw,
            elapsed_seconds=time.time() - t0,
            ok=True,
        )

    def repair(self, frames_b64, semantic_context, python_code, error_text):
        return VLMResponse({}, '', '', 0.0, False, error='repair_not_supported_on_rule_baseline')
