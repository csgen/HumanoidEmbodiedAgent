from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def compute_jerk(joint_log: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(joint_log)
    if len(rows) < 5:
        return {'avg_abs_jerk': 0.0, 'max_abs_jerk': 0.0, 'samples': float(len(rows))}

    joint_names = sorted({
        name
        for row in rows
        for name in (row.get('joints') or {}).keys()
    })
    if not joint_names:
        return {'avg_abs_jerk': 0.0, 'max_abs_jerk': 0.0, 'samples': float(len(rows))}

    t = np.asarray([float(row.get('sim_time', idx)) for idx, row in enumerate(rows)], dtype=float)
    if np.any(np.diff(t) <= 0):
        t = np.arange(len(rows), dtype=float)
    values = np.asarray([
        [float((row.get('joints') or {}).get(name, 0.0)) for name in joint_names]
        for row in rows
    ], dtype=float)

    velocity = np.gradient(values, t, axis=0, edge_order=1)
    accel = np.gradient(velocity, t, axis=0, edge_order=1)
    jerk = np.gradient(accel, t, axis=0, edge_order=1)
    abs_jerk = np.abs(jerk[np.isfinite(jerk)])
    if abs_jerk.size == 0:
        return {'avg_abs_jerk': 0.0, 'max_abs_jerk': 0.0, 'samples': float(len(rows))}
    return {
        'avg_abs_jerk': float(abs_jerk.mean()),
        'max_abs_jerk': float(abs_jerk.max()),
        'samples': float(len(rows)),
    }


def compute_com_excursion(joint_log: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    points = []
    for row in joint_log:
        com = row.get('com_xyz')
        if isinstance(com, list) and len(com) >= 3:
            try:
                points.append([float(com[0]), float(com[1]), float(com[2])])
            except Exception:
                continue
    if len(points) < 2:
        return {'max_xy_excursion_m': 0.0, 'max_z_excursion_m': 0.0, 'samples': float(len(points))}
    arr = np.asarray(points, dtype=float)
    origin = arr[0]
    xy = np.linalg.norm(arr[:, :2] - origin[:2], axis=1)
    z = np.abs(arr[:, 2] - origin[2])
    return {
        'max_xy_excursion_m': float(np.max(xy)),
        'max_z_excursion_m': float(np.max(z)),
        'samples': float(len(points)),
    }


def sandbox_summary(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(events)
    counts: Dict[str, int] = {}
    errors = []
    for row in rows:
        event = str(row.get('event') or 'unknown')
        counts[event] = counts.get(event, 0) + 1
        if row.get('error'):
            errors.append(str(row.get('error')))
    safety_errors = [
        err for err in errors
        if 'joint_limit' in err or 'lower_body' in err or 'forbidden' in err
    ]
    return {
        'event_counts': counts,
        'errors': errors,
        'safety_error_count': len(safety_errors),
        'safety_adherence': 1.0 if not safety_errors else 0.0,
    }


def compute_result_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    artifacts = result.get('artifacts') or {}
    joint_log = load_jsonl(artifacts.get('joint_states', ''))
    sandbox_events = load_jsonl(artifacts.get('sandbox_events', ''))
    exec_outcome = result.get('exec_outcome') or {}
    fallback_stats = result.get('fallback_stats') or {}
    fallback_count = int(fallback_stats.get('tier_a_fires', 0) or 0)
    fallback_count += int(fallback_stats.get('tier_b_fires', 0) or 0)
    fallback_count += int(fallback_stats.get('tier_c_fires', 0) or 0)

    code = ((result.get('vlm_response') or {}).get('python_code') or '').strip()
    execution_success = bool(exec_outcome.get('ok')) and bool(code)

    summary = {
        'execution_success': 1.0 if execution_success else 0.0,
        'fallback_activation_count': fallback_count,
    }
    summary.update({f'jerk_{k}': v for k, v in compute_jerk(joint_log).items()})
    summary.update({f'com_{k}': v for k, v in compute_com_excursion(joint_log).items()})
    sandbox = sandbox_summary(sandbox_events)
    summary['safety_adherence'] = sandbox['safety_adherence']
    summary['sandbox_event_counts'] = sandbox['event_counts']
    summary['sandbox_errors'] = sandbox['errors']
    return summary
