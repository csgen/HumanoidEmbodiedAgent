from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CTRL_DIR = REPO_ROOT / 'nao_VLM' / 'controllers' / 'nao_vlm_test'
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CTRL_DIR))

import config as ctrl_config
from evaluation.metrics import compute_com_excursion, compute_jerk, compute_result_metrics, load_jsonl
from evaluation.rule_baseline import code_for_intent, infer_label_from_scenario_id
from evaluation.scenarios import get_scenarios
from metrics_recorder import MetricsRecorder
from sandbox_exec import SandboxExecutor


class _FakeRobot:
    """Minimal stand-in for the Webots Supervisor — just enough for
    MetricsRecorder.maybe_capture_motion_frame()."""

    def __init__(self) -> None:
        self.export_calls = 0

    def exportImage(self, path, quality):  # noqa: N802 (Webots API name)
        self.export_calls += 1
        Path(path).write_bytes(b'\xff\xd8\xff\xd9')  # minimal JPEG-ish bytes


class Phase5EvaluationTests(unittest.TestCase):
    def test_pilot_scenarios_exist(self):
        scenarios = get_scenarios('pilot', existing_only=True)
        self.assertGreaterEqual(len(scenarios), 5)
        for spec in scenarios:
            self.assertTrue(spec.resolved_video_path().exists())

    def test_metric_functions_on_synthetic_log(self):
        rows = []
        for idx in range(10):
            t = idx * 0.02
            rows.append({
                'sim_time': t,
                'joints': {'HeadYaw': 0.1 * idx, 'HeadPitch': 0.01 * idx * idx},
                'com_xyz': [0.0 + 0.001 * idx, 0.0, 0.25],
            })
        jerk = compute_jerk(rows)
        com = compute_com_excursion(rows)
        self.assertTrue(jerk['avg_abs_jerk'] >= 0.0)
        self.assertTrue(com['max_xy_excursion_m'] > 0.0)

    def test_rule_baseline_mapping(self):
        self.assertEqual(infer_label_from_scenario_id('pilot_waving'), 'wave')
        self.assertIn('move_arm_ik', code_for_intent('wave'))
        self.assertIn('idle', code_for_intent('unknown_label'))

    def test_sandbox_event_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MetricsRecorder('unit', Path(tmp))
            executor = SandboxExecutor()
            executor.register('hold', lambda duration: None)
            executor.set_metrics_recorder(recorder)
            result = executor.run('hold(0.1)')
            self.assertTrue(result.ok)
            events = [
                json.loads(line)
                for line in recorder.sandbox_log_path.read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual([event['event'] for event in events], ['validate_pass', 'exec_ok'])

    def test_compute_result_metrics_from_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            joint_path = tmp_path / 'joint_states.jsonl'
            sandbox_path = tmp_path / 'sandbox_events.jsonl'
            joint_path.write_text(
                '\n'.join(json.dumps({
                    'sim_time': idx * 0.02,
                    'joints': {'HeadYaw': 0.01 * idx},
                    'com_xyz': [0.001 * idx, 0.0, 0.25],
                }) for idx in range(6)) + '\n',
                encoding='utf-8',
            )
            sandbox_path.write_text(
                json.dumps({'event': 'validate_pass'}) + '\n' + json.dumps({'event': 'exec_ok'}) + '\n',
                encoding='utf-8',
            )
            result = {
                'artifacts': {'joint_states': str(joint_path), 'sandbox_events': str(sandbox_path)},
                'exec_outcome': {'ok': True},
                'vlm_response': {'python_code': 'hold(0.1)'},
                'fallback_stats': {'tier_a_fires': 0, 'tier_b_fires': 0, 'tier_c_fires': 0},
            }
            metrics = compute_result_metrics(result)
            self.assertEqual(metrics['execution_success'], 1.0)
            self.assertEqual(metrics['safety_adherence'], 1.0)

    def test_motion_frame_capture_throttling(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = MetricsRecorder('unit_motion', Path(tmp))
            robot = _FakeRobot()
            interval = ctrl_config.MOTION_FRAME_INTERVAL_S
            cap = ctrl_config.MOTION_FRAME_MAX

            # Not armed -> no-op.
            self.assertIsNone(recorder.maybe_capture_motion_frame(robot, 0.0))
            self.assertEqual(robot.export_calls, 0)

            recorder.begin_motion_capture()
            # First capture always succeeds.
            self.assertIsNotNone(recorder.maybe_capture_motion_frame(robot, 0.0))
            # Too soon -> throttled.
            self.assertIsNone(recorder.maybe_capture_motion_frame(robot, interval * 0.5))
            # Enough sim time elapsed -> captures again.
            self.assertIsNotNone(recorder.maybe_capture_motion_frame(robot, interval * 1.1))

            # Spam well past the cap; should stop at MOTION_FRAME_MAX.
            t = interval * 2.0
            for _ in range(cap + 5):
                recorder.maybe_capture_motion_frame(robot, t)
                t += interval * 1.1

            frames = recorder.end_motion_capture()
            self.assertEqual(len(frames), cap)
            for fp in frames:
                self.assertTrue(Path(fp).exists())

            # Disarmed -> no-op again.
            self.assertIsNone(recorder.maybe_capture_motion_frame(robot, t + 100.0))

    def test_missing_artifact_paths_are_empty_logs(self):
        self.assertEqual(load_jsonl(''), [])
        self.assertEqual(load_jsonl(REPO_ROOT), [])
        metrics = compute_result_metrics({
            'artifacts': {},
            'exec_outcome': {'ok': False},
            'vlm_response': {'python_code': ''},
            'fallback_stats': {},
        })
        self.assertEqual(metrics['execution_success'], 0.0)
        self.assertEqual(metrics['safety_adherence'], 1.0)


if __name__ == '__main__':
    unittest.main()
