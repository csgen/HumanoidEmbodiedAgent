# `result.json` Schema Reference

This is the contract for the per-run `result.json` file produced by the Phase 5
evaluation pipeline. The **writer** and the **readers** below all depend on
these field names — if you rename or remove a field, update this doc and grep
the reader list so nothing breaks silently.

This document is descriptive only. It does not change behavior; it records what
the code already does (verified against `feat/phase5-evaluation-framework`).

---

## 1. Lifecycle — `result.json` is written in two stages

A single `result.json` may exist in one of two states depending on how the run
was launched. **This is the main drift risk — know which stage you're reading.**

### Stage 1 — written by the Webots controller
`MetricsRecorder.write_result(payload)` in
`nao_VLM/controllers/nao_vlm_test/metrics_recorder.py` writes the initial file.
It is called from the oneshot / replay exit paths in `nao_vlm_test.py`
(`_run_oneshot_demo`, `_run_replay_demo`). At this point the file has the
**Stage 1 keys** only (see §2).

If you run the Webots controller directly (not through the benchmark harness),
`result.json` stays Stage-1-only.

### Stage 2 — augmented by the benchmark harness
`evaluation/run_benchmark.py:run_one()` runs the controller as a subprocess,
then **reads `result.json` back after Webots exits**, augments it in place with
scenario expectations + Webots subprocess metadata + a computed `metrics`
block, and rewrites the file. This adds the **Stage 2 keys** (see §3).

If you run via `python -m evaluation.run_benchmark ...`, every `result.json`
ends up Stage-2-augmented.

> If `result.json` is missing after the controller exits (crash / timeout),
> `run_benchmark.py` synthesizes a minimal Stage-2 dict with
> `status` = `"timeout"` or `"missing_result"` and a stub `exec_outcome`.

---

## 2. Stage 1 keys (written by the controller / `MetricsRecorder`)

| Key | Type | Meaning |
|---|---|---|
| `run_id` | `str` | Unique run identifier. From `METRICS_RUN_ID`. e.g. `cap_20260514_120000__cap__pilot_waving__r01` |
| `scenario_id` | `str` | Scenario identifier. From `EVAL_SCENARIO_ID`. e.g. `pilot_waving` |
| `method` | `str` | `"cap"` or `"rule_baseline"`. From `EVAL_METHOD`. |
| `status` | `str` | `"ok"` \| `"failed"` \| `"vlm_timeout"` \| `"vlm_exception"` |
| `input` | `dict` | `{"mode": "webcam"\|"replay", "source": "<video path or replay code path>"}` |
| `frames_count` | `int` | Number of frames sampled and sent to the VLM. `0` for replay mode. |
| `vlm_response` | `dict` | VLM response payload — see §4. `{}` on `vlm_timeout`; `{"error": ...}` on `vlm_exception`. |
| `exec_outcome` | `dict` | Sandbox execution result — see §5. |
| `fallback_stats` | `dict` | Fallback-policy fire counts — see §6. `{}` in replay mode. |
| `timeline` | `list[dict]` | Ordered stage events — see §7. (Absent in replay mode.) |
| `artifact_dir` | `str` | Absolute path to the run's artifact directory. (Absent in `vlm_timeout` / `vlm_exception` paths.) |
| `artifacts` | `dict` | Artifact file paths — see §8. Always present (filled by `MetricsRecorder`). |

---

## 3. Stage 2 keys (added by `run_benchmark.py` after Webots exits)

| Key | Type | Meaning |
|---|---|---|
| `scenario_expected_intent` | `str` | Ground-truth human intent for the scenario (from `ScenarioSpec`). |
| `scenario_expected_motion_dynamics` | `str` | Expected `motion_dynamics` label for the scenario. |
| `scenario_expected_response` | `str` | Free-text description of the expected robot response. |
| `video_path` | `str` | Absolute path to the source scenario video. |
| `webots_returncode` | `int \| null` | Webots subprocess exit code. `null` if it timed out. |
| `webots_timed_out` | `bool` | Whether the Webots subprocess hit the harness timeout. |
| `webots_elapsed_seconds` | `float` | Wall-clock duration of the Webots subprocess. |
| `webots_stdout_tail` | `str` | Last 4000 chars of Webots stdout. |
| `webots_stderr_tail` | `str` | Last 4000 chars of Webots stderr. |
| `metrics` | `dict` | Computed offline metrics — see §9. |

---

## 4. `vlm_response` (nested)

Built by `_response_payload(rsp)` in `nao_vlm_test.py`.

| Key | Type | Meaning |
|---|---|---|
| `ok` | `bool` | Whether the VLM call succeeded and parsed. |
| `elapsed_seconds` | `float` | VLM call latency. |
| `error` | `str \| null` | Error string if the call failed (e.g. `parse_incomplete`). |
| `semantic_context` | `dict` | VLM's structured reading of the human — see below. |
| `python_code` | `str` | The generated motion-primitive code. |
| `raw_text` | `str` | Full raw VLM completion text. |

`semantic_context` carries the VLM's JSON block:

| Key | Type | Meaning |
|---|---|---|
| `intent` | `str` | What the human is doing. |
| `social_distance` | `str` | `"close"` \| `"medium"` \| `"far"` |
| `affect` | `str` | Human's apparent emotional state. |
| `confidence` | `float` | VLM's confidence, 0.0–1.0. |
| `motion_dynamics` | `str` | `"oscillatory"` \| `"approaching"` \| `"retreating"` \| `"raising"` \| `"lowering"` \| `"static"` |
| `robot_intent` | `str` | Plain-English description of how the robot should respond (should match `python_code`). |

> In **replay mode**, `vlm_response` is synthetic: `ok=True`, `semantic_context={}`,
> `python_code` = the replayed code, `raw_text=""`, `elapsed_seconds=0.0`.

---

## 5. `exec_outcome` (nested)

Built by `_exec_payload(exec_result)` in `nao_vlm_test.py`.

| Key | Type | Meaning |
|---|---|---|
| `ok` | `bool` | Whether the sandbox executed the code without error. |
| `elapsed_seconds` | `float` | Execution duration. |
| `error` | `str \| null` | Validation or runtime error. `"not_executed"` if no code ran. |
| `traceback` | `str \| null` | Python traceback if execution raised. |

---

## 6. `fallback_stats` (nested)

From `FallbackPolicy.stats()` in `fallback.py`.

| Key | Type | Meaning |
|---|---|---|
| `retry_budget_remaining` | `int` | Tier-A retries left in the current cycle. |
| `history_size` | `int` | Cached successful responses available for Tier-B replay. |
| `tier_a_fires` | `int` | Number of Tier-A (retry) activations. |
| `tier_b_fires` | `int` | Number of Tier-B (replay last good action) activations. |
| `tier_c_fires` | `int` | Number of Tier-C (idle) activations. |

---

## 7. `timeline` (nested)

A list of stage events appended during the oneshot run. Each entry:

| Key | Type | Meaning |
|---|---|---|
| `elapsed_seconds` | `float` | Seconds since the run started. |
| `stage` | `str` | Stage name, e.g. `waiting_for_frames`, `vlm_request_start`, `vlm_response_received`, `robot_execution_start`, `robot_execution_ok`, `robot_execution_failed`, `artifacts_saved`, `done`. |
| `detail` | `str` | Free-text detail for the stage. |

---

## 8. `artifacts` (nested)

Absolute file paths. `MetricsRecorder.write_result()` always fills the first
five; the oneshot success/fail path also adds the last three.

| Key | Type | Always present? | Meaning |
|---|---|---|---|
| `run_dir` | `str` | yes | The run's artifact directory. |
| `joint_states` | `str` | yes | Path to `joint_states.jsonl` — see §10. |
| `sandbox_events` | `str` | yes | Path to `sandbox_events.jsonl` — see §10. |
| `robot_screenshot` | `str` | yes | Path to `robot_response.png`. Empty string if the screenshot export failed. |
| `result_json` | `str` | yes | Path to this `result.json` itself. |
| `input_contact_sheet` | `str` | oneshot main path only | Path to `input_contact_sheet.jpg`. |
| `demo_summary` | `str` | oneshot main path only | Path to `demo_summary.jpg`. Empty string if not generated. |
| `timeline` | `str` | oneshot main path only | Path to `timeline.json`. |

---

## 9. `metrics` (nested, Stage 2 only)

From `compute_result_metrics()` in `evaluation/metrics.py`.

| Key | Type | Meaning |
|---|---|---|
| `execution_success` | `float` | `1.0` if `exec_outcome.ok` is true AND `vlm_response.python_code` is non-empty, else `0.0`. |
| `fallback_activation_count` | `int` | `tier_a_fires + tier_b_fires + tier_c_fires`. |
| `jerk_avg_abs_jerk` | `float` | Mean absolute jerk over all logged joints (3rd time-derivative of joint positions). |
| `jerk_max_abs_jerk` | `float` | Max absolute jerk. |
| `jerk_samples` | `float` | Number of joint-state rows used for the jerk computation. |
| `com_max_xy_excursion_m` | `float` | Max horizontal CoM displacement from the run's first sample, metres. |
| `com_max_z_excursion_m` | `float` | Max vertical CoM displacement from the first sample, metres. |
| `com_samples` | `float` | Number of CoM samples used. |
| `safety_adherence` | `float` | `1.0` if no sandbox error mentions `joint_limit` / `lower_body` / `forbidden`, else `0.0`. |
| `sandbox_event_counts` | `dict` | `{event_name: count}` over `sandbox_events.jsonl`. |
| `sandbox_errors` | `list[str]` | All non-empty error strings from `sandbox_events.jsonl`. |

---

## 10. Sidecar JSONL files

These live next to `result.json` in the run directory and are referenced by
`artifacts.joint_states` / `artifacts.sandbox_events`. They are **JSON Lines**:
one JSON object per line, appended incrementally during the run. Keys within
each line are sorted (`sort_keys=True`).

### `joint_states.jsonl` — one record per simulation step

| Key | Type | Meaning |
|---|---|---|
| `wall_time` | `float` | Unix timestamp when the step was recorded. |
| `sim_time` | `float` | Webots simulation time. |
| `step_index` | `int` | Cumulative step counter (1-based). |
| `joints` | `dict[str, float]` | Joint name → position, read from Webots sensors. |
| `q` | `list[float]` | Full Pinocchio configuration vector. |
| `com_xyz` | `list[float] \| null` | Centre of mass `[x, y, z]` from Pinocchio, or `null` if Pinocchio is unavailable. |

### `sandbox_events.jsonl` — one record per sandbox event

| Key | Type | Meaning |
|---|---|---|
| `wall_time` | `float` | Unix timestamp of the event. |
| `event_index` | `int` | Cumulative event counter (1-based). |
| `event` | `str` | `"validate_pass"` \| `"validate_fail"` \| `"exec_ok"` \| `"exec_error"` |
| `error` | `str \| null` | Error string for `validate_fail` / `exec_error`. |
| `elapsed_seconds` | `float \| null` | Execution duration for `exec_*` events. |
| `code_hash` | `str` | First 16 hex chars of SHA-256 of the executed code (empty if no code). |

---

## 11. Per-run artifact directory layout

A single run produces `artifacts/oneshot/<run_id>/` containing:

```
artifacts/oneshot/<run_id>/
├── result.json                # this file (§2/§3)
├── joint_states.jsonl         # §10
├── sandbox_events.jsonl       # §10
├── robot_response.png         # final robot screenshot (or robot_response.error.txt)
├── frame_01.jpg ... frame_NN.jpg   # sampled VLM input frames (or frame_NN.error.txt)
├── input_contact_sheet.jpg    # horizontal strip of the input frames
├── demo_summary.jpg           # input strip + robot screenshot side-by-side (oneshot main path)
├── semantic_context.json      # the VLM semantic_context block, pretty-printed
├── python_code.py             # the VLM-generated motion code
├── raw_response.txt           # full raw VLM completion
├── timeline.json              # the timeline list (§7)
├── summary.txt                # one-line-per-field human summary
└── execution_traceback.txt    # only if sandbox execution raised
```

> When launched by `run_benchmark.py`, the run directory is
> `artifacts/oneshot/<run_id>/` with
> `run_id = <run_group>__<method>__<scenario_id>__r<NN>`.

---

## 12. Aggregate harness outputs

`run_benchmark.py:write_aggregate()` also emits, under `--output-dir`
(default `artifacts/eval/`):

- **`<run_group>.json`** — a JSON array of every augmented per-run `result.json`
  for the benchmark run.
- **`<run_group>.csv`** — a flat table with columns: `run_id`, `scenario_id`,
  `method`, `status`, `execution_success`, `safety_adherence`,
  `fallback_activation_count`, `jerk_avg_abs_jerk`, `com_max_xy_excursion_m`,
  `webots_returncode`.

`<run_group>` is `<method>_<YYYYMMDD_HHMMSS>`.

---

## 13. Reader map — who depends on which keys

If you change a key, check these consumers:

| Reader | Reads |
|---|---|
| `evaluation/run_benchmark.py` | Loads the whole Stage-1 file, then augments it (adds all Stage-2 keys). |
| `evaluation/metrics.py` (`compute_result_metrics`) | `artifacts.joint_states`, `artifacts.sandbox_events`, `exec_outcome.ok`, `vlm_response.python_code`, `fallback_stats.tier_a_fires` / `tier_b_fires` / `tier_c_fires`. |
| `evaluation/judge.py` (`judge_one`, `_find_images`, `_cache_key`) | `artifacts.run_dir`, `artifacts.robot_screenshot`, `vlm_response.semantic_context.robot_intent`, `vlm_response.python_code`, `scenario_expected_response`, `scenario_id`, `method`. |
| `evaluation/judge.py` (`write_report`) | `scenario_id`, `method`. |

---

## 14. Scope notes

- This is a plain reference, not a validation schema. For a 3-person project a
  markdown table is enough; a JSON Schema + validator can be added later if
  drift actually causes a bug.
- The pipeline does not validate `result.json` against this doc at runtime.
  Treat the doc as the source of truth and keep it in sync by hand when the
  writer/readers change.
