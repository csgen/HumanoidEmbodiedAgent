# Phase-5 benchmark harness. It runs the Webots controller per scenario,
# then reads + augments each run's result.json.
# result.json schema (Stage 1 + Stage 2 keys): see evaluation/RESULT_SCHEMA.md.
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from .metrics import compute_result_metrics
from .scenarios import REPO_ROOT, ScenarioSpec, get_scenarios


WORLD_FILE = REPO_ROOT / 'nao_VLM' / 'worlds' / 'nao_VLM.wbt'


def find_webots() -> str:
    candidates = [
        os.getenv('WEBOTS_BIN', ''),
        shutil.which('webots') or '',
        '/usr/local/webots/webots',
        '/usr/local/bin/webots',
        '/Applications/Webots.app/Contents/MacOS/webots',
        str(Path.home() / '.local' / 'opt' / 'webots' / 'webots'),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError('Webots executable not found. Set WEBOTS_BIN=/path/to/webots.')


def build_webots_command(webots_bin: str, *, headless: bool) -> List[str]:
    cmd = [webots_bin, '--batch', '--mode=fast', '--stdout', '--stderr', str(WORLD_FILE)]
    if headless and shutil.which('xvfb-run'):
        return ['xvfb-run', '-a', *cmd]
    return cmd


def run_one(
    spec: ScenarioSpec,
    *,
    method: str,
    round_index: int,
    run_group: str,
    headless: bool,
    timeout_s: float,
) -> Dict[str, Any]:
    webots_bin = find_webots()
    run_id = f'{run_group}__{method}__{spec.id}__r{round_index:02d}'
    metrics_dir = REPO_ROOT / 'artifacts' / 'oneshot' / run_id
    env = os.environ.copy()
    env.update({
        'REPO_DIR': str(REPO_ROOT),
        'RUN_MODE': 'oneshot',
        'INPUT_MODE': 'webcam',
        'WEBCAM_SOURCE': str(spec.resolved_video_path()),
        'VLM_BACKEND': 'rule_baseline' if method == 'rule_baseline' else env.get('VLM_BACKEND', 'openai'),
        'EVAL_METHOD': method,
        'EVAL_SCENARIO_ID': spec.id,
        'VLM_SCENARIO_HINT': f'{spec.expected_intent}; expected response: {spec.expected_response}',
        'METRICS_RUN_ID': run_id,
        'METRICS_OUTPUT_DIR': str(metrics_dir),
        'ONE_SHOT_EXIT_AFTER_EXECUTE': '1',
        'ONE_SHOT_POST_EXECUTION_SECONDS': env.get('ONE_SHOT_POST_EXECUTION_SECONDS', '0'),
        'ONE_SHOT_BUFFER_TIMEOUT': env.get('ONE_SHOT_BUFFER_TIMEOUT', '8'),
        'ONE_SHOT_VLM_TIMEOUT': env.get('ONE_SHOT_VLM_TIMEOUT', '120'),
        'ONE_SHOT_VIDEO_SETTLE_SECONDS': env.get('ONE_SHOT_VIDEO_SETTLE_SECONDS', '0'),
        'FRAME_BUFFER_SECONDS': env.get('FRAME_BUFFER_SECONDS', '3'),
        'FRAME_BUFFER_FPS': env.get('FRAME_BUFFER_FPS', '10'),
        'VLM_FRAME_COUNT': env.get('VLM_FRAME_COUNT', '5'),
    })
    if method == 'cap' and env.get('VLM_BACKEND', '').strip().lower() in {'', 'auto'}:
        env['VLM_BACKEND'] = 'openai'

    cmd = build_webots_command(webots_bin, headless=headless)
    started = time.time()
    timed_out = False
    timeout_error = ''
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
        webots_returncode = proc.returncode
        stdout_tail = proc.stdout[-4000:]
        stderr_tail = proc.stderr[-4000:]
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        timeout_error = str(exc)
        webots_returncode = None
        stdout = exc.stdout or ''
        stderr = exc.stderr or ''
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors='replace')
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors='replace')
        stdout_tail = stdout[-4000:]
        stderr_tail = stderr[-4000:]

    result_path = metrics_dir / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding='utf-8'))
    else:
        result = {
            'run_id': run_id,
            'scenario_id': spec.id,
            'method': method,
            'status': 'timeout' if timed_out else 'missing_result',
            'exec_outcome': {
                'ok': False,
                'error': timeout_error if timed_out else 'result.json missing',
            },
            'artifacts': {'run_dir': str(metrics_dir), 'result_json': str(result_path)},
            'fallback_stats': {},
        }
    result.update({
        'scenario_expected_intent': spec.expected_intent,
        'scenario_expected_motion_dynamics': spec.expected_motion_dynamics,
        'scenario_expected_response': spec.expected_response,
        'video_path': str(spec.resolved_video_path()),
        'webots_returncode': webots_returncode,
        'webots_timed_out': timed_out,
        'webots_elapsed_seconds': time.time() - started,
        'webots_stdout_tail': stdout_tail,
        'webots_stderr_tail': stderr_tail,
    })
    result['metrics'] = compute_result_metrics(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result


def write_aggregate(results: List[Dict[str, Any]], out_dir: Path, run_group: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f'{run_group}.json'
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    csv_path = out_dir / f'{run_group}.csv'
    fieldnames = [
        'run_id', 'scenario_id', 'method', 'status', 'execution_success',
        'safety_adherence', 'fallback_activation_count', 'jerk_avg_abs_jerk',
        'com_max_xy_excursion_m', 'webots_returncode',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result.get('metrics') or {}
            writer.writerow({
                'run_id': result.get('run_id'),
                'scenario_id': result.get('scenario_id'),
                'method': result.get('method'),
                'status': result.get('status'),
                'execution_success': metrics.get('execution_success'),
                'safety_adherence': metrics.get('safety_adherence'),
                'fallback_activation_count': metrics.get('fallback_activation_count'),
                'jerk_avg_abs_jerk': metrics.get('jerk_avg_abs_jerk'),
                'com_max_xy_excursion_m': metrics.get('com_max_xy_excursion_m'),
                'webots_returncode': result.get('webots_returncode'),
            })
    return json_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run Phase-5 Webots benchmark.')
    parser.add_argument('--scenario-set', default='pilot', choices=['pilot', 'canonical', 'all'])
    parser.add_argument('--scenarios', nargs='*', help='Optional scenario ids inside the selected set')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--method', choices=['cap', 'rule_baseline'], default='cap')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--timeout-s', type=float, default=180.0)
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'artifacts' / 'eval')
    args = parser.parse_args(argv)

    scenarios = get_scenarios(args.scenario_set, args.scenarios, existing_only=True)
    if not scenarios:
        print(f'No existing videos found for scenario set {args.scenario_set!r}.', file=sys.stderr)
        return 1

    run_group = time.strftime(f'{args.method}_%Y%m%d_%H%M%S')
    results: List[Dict[str, Any]] = []
    for round_index in range(1, max(1, args.rounds) + 1):
        for spec in scenarios:
            print(f'[benchmark] {args.method} {spec.id} round {round_index}/{args.rounds}')
            try:
                results.append(
                    run_one(
                        spec,
                        method=args.method,
                        round_index=round_index,
                        run_group=run_group,
                        headless=args.headless,
                        timeout_s=args.timeout_s,
                    )
                )
            except subprocess.TimeoutExpired as exc:
                # run_one normally handles timeouts so it can salvage a
                # completed result.json. Keep this as a defensive fallback.
                results.append({
                    'run_id': f'{run_group}__{args.method}__{spec.id}__r{round_index:02d}',
                    'scenario_id': spec.id,
                    'method': args.method,
                    'status': 'timeout',
                    'exec_outcome': {'ok': False, 'error': str(exc)},
                    'metrics': {'execution_success': 0.0, 'safety_adherence': 0.0},
                })
    aggregate = write_aggregate(results, args.output_dir, run_group)
    print(f'[benchmark] wrote {aggregate}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
