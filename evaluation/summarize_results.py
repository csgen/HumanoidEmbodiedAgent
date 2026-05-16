from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .metrics import compute_result_metrics


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _iter_results(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in paths:
        payload = _read_json(path)
        if isinstance(payload, list):
            results.extend(item for item in payload if isinstance(item, dict))
        elif isinstance(payload, dict):
            results.append(payload)
    return results


def _ensure_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = result.get('metrics')
    if isinstance(metrics, dict):
        return metrics
    metrics = compute_result_metrics(result)
    result['metrics'] = metrics
    return metrics


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _method_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get('method') or 'unknown')].append(result)

    stats: Dict[str, Dict[str, float]] = {}
    for method, rows in grouped.items():
        metrics_rows = [_ensure_metrics(row) for row in rows]
        judge_passes = [
            1.0
            for row in rows
            if ((row.get('judge') or {}).get('pass')) is True
        ]
        judge_total = [
            1.0
            for row in rows
            if isinstance(row.get('judge'), dict) and 'pass' in row.get('judge', {})
        ]
        stats[method] = {
            'runs': float(len(rows)),
            'execution_success': _mean(m.get('execution_success', 0.0) for m in metrics_rows),
            'safety_adherence': _mean(m.get('safety_adherence', 0.0) for m in metrics_rows),
            'fallback_activation_count': _mean(m.get('fallback_activation_count', 0.0) for m in metrics_rows),
            'jerk_avg_abs_jerk': _mean(m.get('jerk_avg_abs_jerk', 0.0) for m in metrics_rows),
            'com_max_xy_excursion_m': _mean(m.get('com_max_xy_excursion_m', 0.0) for m in metrics_rows),
            'judge_pass_rate': (sum(judge_passes) / len(judge_total)) if judge_total else -1.0,
        }
    return stats


def _scenario_rows(results: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        scenario_id = str(result.get('scenario_id') or 'unknown')
        method = str(result.get('method') or 'unknown')
        grouped[scenario_id][method].append(result)

    rows: List[Tuple[str, Dict[str, Dict[str, float]]]] = []
    for scenario_id in sorted(grouped):
        methods: Dict[str, Dict[str, float]] = {}
        for method, method_results in grouped[scenario_id].items():
            metrics_rows = [_ensure_metrics(row) for row in method_results]
            methods[method] = {
                'execution_success': _mean(m.get('execution_success', 0.0) for m in metrics_rows),
                'safety_adherence': _mean(m.get('safety_adherence', 0.0) for m in metrics_rows),
                'fallback_activation_count': _mean(m.get('fallback_activation_count', 0.0) for m in metrics_rows),
                'jerk_avg_abs_jerk': _mean(m.get('jerk_avg_abs_jerk', 0.0) for m in metrics_rows),
            }
        rows.append((scenario_id, methods))
    return rows


def _winner_label(methods: Dict[str, Dict[str, float]]) -> str:
    if len(methods) < 2:
        return next(iter(methods), 'n/a')
    ranked = sorted(
        methods.items(),
        key=lambda item: (
            item[1].get('execution_success', 0.0),
            item[1].get('safety_adherence', 0.0),
            -item[1].get('fallback_activation_count', 0.0),
            -item[1].get('jerk_avg_abs_jerk', 0.0),
        ),
        reverse=True,
    )
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return 'tie'
    return ranked[0][0]


def _failure_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bad = []
    for result in results:
        metrics = _ensure_metrics(result)
        fallback_count = float(metrics.get('fallback_activation_count', 0.0) or 0.0)
        if (
            result.get('status') != 'ok'
            or float(metrics.get('execution_success', 0.0) or 0.0) < 1.0
            or result.get('webots_timed_out')
            or fallback_count > 0.0
        ):
            bad.append(result)
    return bad


def _best_artifacts(results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    def key_fn(result: Dict[str, Any]):
        metrics = _ensure_metrics(result)
        artifacts = result.get('artifacts') or {}
        has_visual = 1 if artifacts.get('demo_summary') or artifacts.get('robot_motion_contact_sheet') or artifacts.get('robot_screenshot') else 0
        return (
            float(metrics.get('execution_success', 0.0) or 0.0),
            float(metrics.get('safety_adherence', 0.0) or 0.0),
            -float(metrics.get('fallback_activation_count', 0.0) or 0.0),
            has_visual,
            -float(metrics.get('jerk_avg_abs_jerk', 0.0) or 0.0),
        )

    ranked = sorted(results, key=key_fn, reverse=True)
    chosen: List[Dict[str, Any]] = []
    seen = set()
    for result in ranked:
        key = (result.get('scenario_id'), result.get('method'))
        if key in seen:
            continue
        chosen.append(result)
        seen.add(key)
        if len(chosen) >= max(1, top_k):
            break
    return chosen


def render_summary(results: List[Dict[str, Any]], *, source_paths: List[Path], top_k: int = 5) -> str:
    if not results:
        return '# Phase 5 Benchmark Summary\n\nNo results found.\n'

    method_stats = _method_stats(results)
    scenario_rows = _scenario_rows(results)
    failures = _failure_rows(results)
    best = _best_artifacts(results, top_k=top_k)

    lines = [
        '# Phase 5 Benchmark Summary',
        '',
        f'Source files: {", ".join(str(path) for path in source_paths)}',
        f'Total runs: {len(results)}',
        '',
        '## Per-Method Averages',
        '',
        '| Method | Runs | Exec Success | Safety | Avg Fallbacks | Avg Jerk | Avg CoM XY (m) | Judge Pass |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for method in sorted(method_stats):
        stats = method_stats[method]
        judge = 'n/a' if stats['judge_pass_rate'] < 0.0 else f"{stats['judge_pass_rate'] * 100.0:.1f}%"
        lines.append(
            f"| {method} | {int(stats['runs'])} | {stats['execution_success']:.2f} | "
            f"{stats['safety_adherence']:.2f} | {stats['fallback_activation_count']:.2f} | "
            f"{stats['jerk_avg_abs_jerk']:.2f} | {stats['com_max_xy_excursion_m']:.4f} | {judge} |"
        )

    lines.extend([
        '',
        '## Per-Scenario Comparison',
        '',
        '| Scenario | cap | rule_baseline | Winner |',
        '|---|---|---|---|',
    ])
    for scenario_id, methods in scenario_rows:
        def fmt(method: str) -> str:
            row = methods.get(method)
            if not row:
                return 'n/a'
            return (
                f"success={row['execution_success']:.2f}, "
                f"safety={row['safety_adherence']:.2f}, "
                f"fallbacks={row['fallback_activation_count']:.2f}, "
                f"jerk={row['jerk_avg_abs_jerk']:.2f}"
            )
        lines.append(
            f"| {scenario_id} | {fmt('cap')} | {fmt('rule_baseline')} | {_winner_label(methods)} |"
        )

    lines.extend([
        '',
        '## Failures / Timeouts / Fallbacks',
        '',
    ])
    if not failures:
        lines.append('None.')
    else:
        for result in failures:
            metrics = _ensure_metrics(result)
            outcome = result.get('exec_outcome') or {}
            reason = outcome.get('error') or result.get('status') or 'unknown'
            lines.append(
                f"- `{result.get('run_id', '')}`: status={result.get('status', '')}, "
                f"success={metrics.get('execution_success', 0.0):.2f}, "
                f"fallbacks={metrics.get('fallback_activation_count', 0.0):.0f}, "
                f"timed_out={bool(result.get('webots_timed_out'))}, "
                f"reason={reason}"
            )

    lines.extend([
        '',
        '## Strongest Demo Artifacts',
        '',
    ])
    for result in best:
        artifacts = result.get('artifacts') or {}
        visual = (
            artifacts.get('demo_summary')
            or artifacts.get('robot_motion_contact_sheet')
            or artifacts.get('robot_screenshot')
            or ''
        )
        lines.append(
            f"- `{result.get('scenario_id', '')}` / `{result.get('method', '')}`: "
            f"`{visual}` | `{artifacts.get('result_json', '')}` | `{artifacts.get('run_dir', '')}`"
        )

    return '\n'.join(lines) + '\n'


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Summarize benchmark aggregate JSON files.')
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts/eval/summary.md'))
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args(argv)

    results = _iter_results(args.results)
    summary = render_summary(results, source_paths=args.results, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding='utf-8')
    print(f'[summary] wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
