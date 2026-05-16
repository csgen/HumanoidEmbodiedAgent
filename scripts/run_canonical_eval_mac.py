#!/usr/bin/env python3
"""
Mac-compatible canonical evaluation launcher.

Runs all 8 canonical scenarios in ONE persistent Webots window (batch mode).
Displays each human input video in an OpenCV window on the left side of the
screen while Webots processes it on the right.

Usage:
    python3 scripts/run_canonical_eval_mac.py [--method cap|rule_baseline|both] [--rounds N]

Requirements (Mac):
    - Webots R2025a at /Applications/Webots.app or in PATH
    - ffmpeg/ffplay (brew install ffmpeg) — optional, used for video display fallback
    - OpenCV (pip install opencv-python or conda install opencv)
    - OPENAI_API_KEY set in .env or environment (for cap method)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / '.env')
except ImportError:
    pass

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False


def find_webots() -> str:
    candidates = [
        os.getenv('WEBOTS_BIN', ''),
        '/Applications/Webots.app/Contents/MacOS/webots',
    ]
    try:
        import shutil
        candidates.append(shutil.which('webots') or '')
    except Exception:
        pass
    for c in candidates:
        if c and Path(c).exists():
            return c
    raise FileNotFoundError(
        'Webots not found. Set WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots'
    )


def position_webots_window() -> None:
    """Use osascript to move the Webots window to the right half of the screen."""
    script = '''
    tell application "System Events"
        tell process "webots"
            set position of window 1 to {900, 0}
            set size of window 1 to {900, 768}
        end tell
    end tell
    '''
    try:
        subprocess.run(['osascript', '-e', script], check=False, capture_output=True, timeout=5)
    except Exception:
        pass


def play_video_loop(video_path: str, stop_event: threading.Event) -> None:
    """Display a video in a loop in an OpenCV window until stop_event is set."""
    if not _CV2_OK:
        print(f'[panel] opencv not available; skipping video display for {video_path}')
        stop_event.wait()
        return

    cap = cv2.VideoCapture(video_path)
    window_name = 'Human Input Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 860, 700)
    cv2.moveWindow(window_name, 0, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    delay = max(1, int(1000.0 / fps))

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        cv2.imshow(window_name, frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)


def build_playlist(scenarios: list, method: str, run_group: str, round_index: int) -> dict:
    scenario_specs = []
    for spec in scenarios:
        run_id = f'{run_group}__{method}__{spec["id"]}__r{round_index:02d}'
        metrics_dir = REPO_ROOT / 'artifacts' / 'oneshot' / run_id
        scenario_specs.append({
            'id': spec['id'],
            'video_path': spec['video_path'],
            'method': method,
            'run_id': run_id,
            'metrics_dir': str(metrics_dir),
            'hint': spec.get('hint', ''),
            'status': 'pending',
        })
    return {'scenarios': scenario_specs, 'batch_status': 'pending'}


def run_batch(scenarios: list, method: str, rounds: int, headless: bool) -> list:
    webots_bin = find_webots()
    world_file = REPO_ROOT / 'nao_VLM' / 'worlds' / 'nao_VLM.wbt'
    run_group = time.strftime(f'{method}_%Y%m%d_%H%M%S')
    all_results: list = []

    for round_index in range(1, rounds + 1):
        playlist = build_playlist(scenarios, method, run_group, round_index)
        playlist_file = Path(tempfile.mktemp(suffix='_batch_playlist.json'))
        playlist_file.write_text(json.dumps(playlist, indent=2) + '\n', encoding='utf-8')

        env = os.environ.copy()
        env.update({
            'REPO_DIR': str(REPO_ROOT),
            'RUN_MODE': 'batch',
            'INPUT_MODE': 'webcam',
            'WEBCAM_SOURCE': scenarios[0]['video_path'],
            'VLM_BACKEND': 'rule_baseline' if method == 'rule_baseline' else env.get('VLM_BACKEND', 'openai'),
            'EVAL_METHOD': method,
            'BATCH_PLAYLIST_FILE': str(playlist_file),
            'BATCH_INTER_SCENARIO_HOLD': '2.0',
            'ONE_SHOT_EXIT_AFTER_EXECUTE': '1',
            'ONE_SHOT_POST_EXECUTION_SECONDS': '2.0',
            'ONE_SHOT_BUFFER_TIMEOUT': '10',
            'ONE_SHOT_VLM_TIMEOUT': '120',
            'ONE_SHOT_VIDEO_SETTLE_SECONDS': '0',
            'FRAME_BUFFER_SECONDS': '3',
            'FRAME_BUFFER_FPS': '10',
            'VLM_FRAME_COUNT': '5',
        })
        if method == 'cap' and env.get('VLM_BACKEND', '').lower() in {'', 'auto'}:
            env['VLM_BACKEND'] = 'openai'

        cmd = [webots_bin, '--batch', '--mode=realtime', '--stdout', '--stderr', str(world_file)]

        print(f'\n[panel] Starting Webots (round {round_index}/{rounds}) — ONE window for all {len(scenarios)} scenarios')
        print(f'[panel] Method: {method}')
        print(f'[panel] Playlist: {playlist_file}')
        print('[panel] Webots window will open shortly. It will stay open for ALL scenarios.\n')

        webots_proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Give Webots 5s to open, then position it
        time.sleep(5)
        position_webots_window()

        # Watch playlist for scenario changes and play corresponding video
        current_scenario_idx = -1
        stop_video_event = threading.Event()
        video_thread: threading.Thread | None = None

        def watch_and_display() -> None:
            nonlocal current_scenario_idx, video_thread, stop_video_event
            while webots_proc.poll() is None:
                try:
                    pl = json.loads(playlist_file.read_text(encoding='utf-8'))
                    specs = pl.get('scenarios', [])
                    running_idx = next(
                        (i for i, s in enumerate(specs) if s.get('status') == 'running'),
                        -1
                    )
                    if running_idx != current_scenario_idx and running_idx >= 0:
                        # Stop previous video
                        stop_video_event.set()
                        if video_thread is not None:
                            video_thread.join(timeout=2.0)

                        current_scenario_idx = running_idx
                        spec = specs[running_idx]
                        print(f'\n[panel] Now: scenario {running_idx + 1}/{len(specs)}: {spec["id"]}')
                        stop_video_event = threading.Event()
                        video_thread = threading.Thread(
                            target=play_video_loop,
                            args=(spec['video_path'], stop_video_event),
                            daemon=True,
                        )
                        video_thread.start()
                except Exception:
                    pass
                time.sleep(0.5)

            # Stop video when Webots exits
            stop_video_event.set()
            if video_thread is not None:
                video_thread.join(timeout=2.0)

        watcher_thread = threading.Thread(target=watch_and_display, daemon=True)
        watcher_thread.start()

        # Stream Webots logs to terminal
        if webots_proc.stdout:
            for line in webots_proc.stdout:
                print(f'[webots] {line}', end='')

        webots_proc.wait()
        watcher_thread.join(timeout=5.0)

        # Read results
        try:
            final_playlist = json.loads(playlist_file.read_text(encoding='utf-8'))
        except Exception:
            final_playlist = playlist

        for spec_dict in final_playlist.get('scenarios', []):
            run_id = spec_dict.get('run_id', '')
            metrics_dir = Path(spec_dict.get('metrics_dir', ''))
            result_path = metrics_dir / 'result.json'
            if result_path.exists():
                try:
                    all_results.append(json.loads(result_path.read_text(encoding='utf-8')))
                except Exception:
                    pass

        try:
            playlist_file.unlink()
        except Exception:
            pass

    return all_results


def print_summary(results: list) -> None:
    print('\n' + '=' * 70)
    print(' CANONICAL EVALUATION RESULTS')
    print('=' * 70)
    print(f'{"Scenario":<25} {"Method":<15} {"Success":<10} {"Safety":<10} {"Jerk":<8}')
    print('-' * 70)
    for r in results:
        m = r.get('metrics') or {}
        print(
            f'{r.get("scenario_id", "?"):<25} '
            f'{r.get("method", "?"):<15} '
            f'{m.get("execution_success", "?")!s:<10} '
            f'{m.get("safety_adherence", "?")!s:<10} '
            f'{m.get("jerk_avg_abs_jerk", "?")!s:<8}'
        )
    print('=' * 70)


def main() -> int:
    parser = argparse.ArgumentParser(description='Mac canonical evaluation panel')
    parser.add_argument('--method', choices=['cap', 'rule_baseline', 'both'], default='cap')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from evaluation.scenarios import get_scenarios
    scenario_specs_objs = get_scenarios('canonical', existing_only=True)
    if not scenario_specs_objs:
        print('[panel] ERROR: No canonical videos found in videos/scenario_*.mp4')
        print('[panel] Expected files: videos/scenario_01_wave.mp4 ... videos/scenario_08_idle.mp4')
        return 1

    scenarios = [
        {
            'id': s.id,
            'video_path': str(s.resolved_video_path()),
            'hint': f'{s.expected_intent}; expected response: {s.expected_response}',
        }
        for s in scenario_specs_objs
    ]

    print(f'[panel] Found {len(scenarios)} canonical scenarios')
    for s in scenarios:
        print(f'  - {s["id"]}: {s["video_path"]}')

    methods = ['rule_baseline', 'cap'] if args.method == 'both' else [args.method]
    all_results: list = []
    for method in methods:
        print(f'\n[panel] Running method: {method}')
        results = run_batch(scenarios, method, args.rounds, args.headless)
        all_results.extend(results)

    print_summary(all_results)

    out_dir = REPO_ROOT / 'artifacts' / 'eval'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'canonical_{args.method}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    out_file.write_text(json.dumps(all_results, indent=2) + '\n', encoding='utf-8')
    print(f'\n[panel] Results saved: {out_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
