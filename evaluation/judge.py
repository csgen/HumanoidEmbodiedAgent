from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Load the repo-root .env so llm_api_key / OPENAI_API_KEY / base_url are
# available. judge.py runs from the shell (not inside the Webots controller,
# which is the only component that previously loaded .env), so without this
# the judge silently "skips" whenever the key is only in .env.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / '.env')
except ImportError:
    pass


try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _iter_results(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        if isinstance(payload, list):
            out.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            out.append(payload)
    return out


def _image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip('.') or 'png'
    mime = 'jpeg' if suffix in {'jpg', 'jpeg'} else suffix
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'data:image/{mime};base64,{data}'


def _find_images(result: Dict[str, Any]) -> Tuple[Path | None, Path | None]:
    artifacts = result.get('artifacts') or {}
    run_dir = Path(artifacts.get('run_dir') or '').expanduser()
    input_frame = run_dir / 'frame_01.jpg'
    if not input_frame.exists():
        input_frame = None
    robot = Path(artifacts.get('robot_screenshot') or '')
    if not robot.exists():
        robot = run_dir / 'robot_response.png'
    if not robot.exists():
        robot = None
    return input_frame, robot


def _cache_key(result: Dict[str, Any], input_frame: Path, robot_frame: Path) -> str:
    code = ((result.get('vlm_response') or {}).get('python_code') or '')
    return '|'.join([
        result.get('scenario_id', ''),
        result.get('method', ''),
        str(input_frame.stat().st_size),
        str(robot_frame.stat().st_size),
        str(abs(hash(code))),
    ])


def judge_one(client, model: str, result: Dict[str, Any], input_frame: Path, robot_frame: Path) -> Dict[str, Any]:
    ctx = (result.get('vlm_response') or {}).get('semantic_context') or {}
    robot_intent = ctx.get('robot_intent') or result.get('scenario_expected_response') or ''
    prompt = (
        'You are judging a humanoid robot response. The first image is the human input; '
        'the second image is the robot response after executing its generated motion. '
        'Return only JSON with pass:boolean and rationale:string <=30 words. '
        f'Expected/claimed robot intent: {robot_intent}'
    )
    rsp = client.chat.completions.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': _image_data_url(input_frame), 'detail': 'low'}},
                {'type': 'image_url', 'image_url': {'url': _image_data_url(robot_frame), 'detail': 'low'}},
            ],
        }],
        temperature=0.0,
        max_tokens=120,
    )
    raw = rsp.choices[0].message.content or ''
    cleaned = raw.strip()
    if cleaned.startswith('```'):
        cleaned = cleaned.strip('`')
        cleaned = cleaned.replace('json', '', 1).strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {'pass': False, 'rationale': f'unparseable judge output: {raw[:80]}'}
    return {
        'pass': bool(parsed.get('pass')),
        'rationale': str(parsed.get('rationale', ''))[:240],
        'raw': raw,
    }


def write_report(results: List[Dict[str, Any]], judgements: Dict[str, Dict[str, Any]], output: Path) -> None:
    rows = []
    for result in results:
        key = result.get('_judge_key')
        judgement = judgements.get(key or '', {})
        rows.append((result, judgement))
    total = len([j for _, j in rows if j])
    passed = sum(1 for _, j in rows if j.get('pass'))
    lines = [
        '# Phase 5 VLM-as-Judge Report',
        '',
        f'Judged runs: {total}',
        f'Pass rate: {(passed / total * 100.0) if total else 0.0:.1f}%',
        '',
        '| Scenario | Method | Pass | Rationale |',
        '|---|---:|---:|---|',
    ]
    for result, judgement in rows:
        if not judgement:
            continue
        lines.append(
            f"| {result.get('scenario_id', '')} | {result.get('method', '')} | "
            f"{'yes' if judgement.get('pass') else 'no'} | {judgement.get('rationale', '')} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Run VLM-as-Judge over benchmark result JSON files.')
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/eval/report.md'))
    parser.add_argument('--cache', type=Path, default=Path('artifacts/eval/judge_cache.json'))
    parser.add_argument('--model', default=os.getenv('VLM_JUDGE_MODEL', 'gpt-4o'))
    args = parser.parse_args(argv)

    results = _iter_results(args.results)
    cache: Dict[str, Dict[str, Any]] = {}
    if args.cache.exists():
        cache = _read_json(args.cache)

    api_key = os.getenv('llm_api_key') or os.getenv('OPENAI_API_KEY')
    if OpenAI is None or not api_key:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            'Judge skipped: openai package or llm_api_key/OPENAI_API_KEY is unavailable.\n',
            encoding='utf-8',
        )
        print(f'[judge] skipped; wrote {args.output}')
        return 0

    client = OpenAI(api_key=api_key, base_url=os.getenv('base_url') or None)
    for result in results:
        input_frame, robot_frame = _find_images(result)
        if input_frame is None or robot_frame is None:
            continue
        key = _cache_key(result, input_frame, robot_frame)
        result['_judge_key'] = key
        if key not in cache:
            cache[key] = judge_one(client, args.model, result, input_frame, robot_frame)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(results, cache, args.output)
    print(f'[judge] wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
