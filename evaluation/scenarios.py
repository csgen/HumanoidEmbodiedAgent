from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    video_path: Path
    expected_intent: str
    expected_motion_dynamics: str
    expected_response: str

    def resolved_video_path(self) -> Path:
        return self.video_path if self.video_path.is_absolute() else REPO_ROOT / self.video_path


PILOT_SCENARIOS: Dict[str, ScenarioSpec] = {
    'pilot_waving': ScenarioSpec(
        'pilot_waving',
        Path('debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4'),
        'greeting',
        'oscillatory',
        'short upper-body greeting response',
    ),
    'pilot_pointing': ScenarioSpec(
        'pilot_pointing',
        Path('debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4'),
        'attention_direction',
        'static',
        'attentive head or single-arm response',
    ),
    'pilot_finger_no': ScenarioSpec(
        'pilot_finger_no',
        Path('debug_video_samples/finger_no__no_no_finger_wave__82vLYYXukIE.mp4'),
        'rejection',
        'oscillatory',
        'small cautious acknowledgement',
    ),
    'pilot_stop': ScenarioSpec(
        'pilot_stop',
        Path('debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4'),
        'stop',
        'static',
        'brief pause or soft wait-palm response',
    ),
    'pilot_thumbs_up': ScenarioSpec(
        'pilot_thumbs_up',
        Path('debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4'),
        'approval',
        'static',
        'positive restrained acknowledgement',
    ),
    'pilot_yes_nod': ScenarioSpec(
        'pilot_yes_nod',
        Path('debug_video_samples/yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4'),
        'agreement',
        'oscillatory',
        'head-led acknowledgement',
    ),
    'pilot_no_shake': ScenarioSpec(
        'pilot_no_shake',
        Path('debug_video_samples/no_shake__man_shake_head_no__yZ-351AUZqE.mp4'),
        'disagreement',
        'oscillatory',
        'head-led cautious acknowledgement',
    ),
    'pilot_clap': ScenarioSpec(
        'pilot_clap',
        Path('debug_video_samples/clap__clapping_hands__YTQJL_kpZd8.mp4'),
        'celebration',
        'oscillatory',
        'happy upper-body acknowledgement',
    ),
    'pilot_beckon': ScenarioSpec(
        'pilot_beckon',
        Path('debug_video_samples/beckon__woman_come_here_beckon__9CeeTCQskFs.mp4'),
        'beckoning',
        'oscillatory',
        'curious attentive response',
    ),
    'pilot_shrug': ScenarioSpec(
        'pilot_shrug',
        Path('debug_video_samples/shrug__shrugging_person__bx_US6Mwdhk.mp4'),
        'uncertainty',
        'raising',
        'small curious head response',
    ),
}


CANONICAL_SCENARIOS: Dict[str, ScenarioSpec] = {
    'wave': ScenarioSpec('wave', Path('videos/scenario_01_wave.mp4'), 'greeting', 'oscillatory', 'wave or acknowledge warmly'),
    'cross_arms': ScenarioSpec('cross_arms', Path('videos/scenario_02_cross_arms.mp4'), 'rejection', 'static', 'respectful cautious response'),
    'lean_forward': ScenarioSpec('lean_forward', Path('videos/scenario_03_lean_forward.mp4'), 'approach', 'approaching', 'soft cautious lean-back/open-palm response'),
    'walk_away': ScenarioSpec('walk_away', Path('videos/scenario_04_walk_away.mp4'), 'leaving', 'retreating', 'small farewell or attentive pause'),
    'crouch': ScenarioSpec('crouch', Path('videos/scenario_05_crouch.mp4'), 'lower_focus', 'lowering', 'look down attentively'),
    'reject_gesture': ScenarioSpec('reject_gesture', Path('videos/scenario_06_reject.mp4'), 'rejection', 'static', 'respectful stop/withdraw cue'),
    'greet_handshake': ScenarioSpec('greet_handshake', Path('videos/scenario_07_handshake.mp4'), 'greeting', 'raising', 'gentle greeting hand response'),
    'idle_standing': ScenarioSpec('idle_standing', Path('videos/scenario_08_idle.mp4'), 'neutral', 'static', 'calm idle/head attention'),
}


def scenario_sets() -> Mapping[str, Dict[str, ScenarioSpec]]:
    return {
        'pilot': PILOT_SCENARIOS,
        'canonical': CANONICAL_SCENARIOS,
        'all': {**PILOT_SCENARIOS, **CANONICAL_SCENARIOS},
    }


def get_scenarios(scenario_set: str = 'pilot', ids: Optional[Iterable[str]] = None,
                  existing_only: bool = True) -> List[ScenarioSpec]:
    sets = scenario_sets()
    if scenario_set not in sets:
        raise ValueError(f'unknown scenario set {scenario_set!r}; choose one of {sorted(sets)}')
    registry = sets[scenario_set]
    if ids:
        selected = []
        for scenario_id in ids:
            if scenario_id not in registry:
                raise ValueError(f'unknown scenario id {scenario_id!r} in set {scenario_set!r}')
            selected.append(registry[scenario_id])
    else:
        selected = list(registry.values())
    if existing_only:
        selected = [spec for spec in selected if spec.resolved_video_path().exists()]
    return selected
