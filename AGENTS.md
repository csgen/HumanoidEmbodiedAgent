# HumanoidEmbodiedAgent Agent Handoff

This file is the root-level handoff for future agents and collaborators. The short version: this project is now a Webots + Python + Pinocchio + VLM system, and the active milestone is Phase 5 evaluation, not a ROS/Gazebo rebuild.

## Current Project Shape

- Simulator: Webots, with the main world at `nao_VLM/worlds/nao_VLM.wbt`.
- Robot: NAO H25 assets and URDF under `nao_VLM/nao_robot/`.
- Controller: `nao_VLM/controllers/nao_vlm_test/nao_vlm_test.py`.
- Robot API/safety path: VLM returns Python primitive code, the sandbox validates it, then the controller executes primitives against Webots/Pinocchio.
- Active VLM controller config: `nao_VLM/controllers/nao_vlm_test/config.py`.
- Main planning doc: `Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan.md`.
- Mac ROS notes: `SamLearningsMacROS.md` is useful context but low signal for this repo's main path. Do not revive ROS unless explicitly asked.

## Phase 5 Decision

The next milestone is evaluation-first:

1. Run repeatable benchmark scenarios from prerecorded videos.
2. Record per-run artifacts and metrics.
3. Compare CaP/VLM control against a rule-based baseline.
4. Use live camera as a demo path after the benchmark is trustworthy.

Live camera support already exists conceptually through:

```bash
INPUT_MODE=webcam
WEBCAM_SOURCE=0
RUN_MODE=periodic
```

It can also consume a video file or MJPEG URL through `WEBCAM_SOURCE`.

## Phase Narrative And Value

The project should not be presented as "we called an LLM API and mapped the answer to robot functions." That framing is too shallow and will make the work look like basic integration. The stronger and more accurate framing is:

> A humanoid embodied interaction system that converts multi-frame human visual
> cues into safety-constrained, physically grounded, low-level robot motion
> programs, then evaluates those programs against a fixed baseline using
> reproducible metrics.

What was inherited before Phase 5:

- Webots simulation with a NAO H25 humanoid model rather than the original custom 9-DoF wheeled-base URDF.
- A Python Webots controller with NAO motors, sensors, Pinocchio IK, and CoM synchronization.
- VLM integration capable of reading sampled human video frames and generating primitive Python code.
- Motion grammar primitives such as `move_joint`, `move_joints`, `move_arm_ik`, `move_head`, `set_hand`, `oscillate_joint`, `hold`, and `idle`.
- A sandbox executor that validates VLM-generated code before actuation.
- Fallback policy, state-aware trigger, rolling frame buffer, idle animation, and static/pre-recorded video oneshot demos.

What Phase 5 adds:

- Reproducible benchmark harness instead of ad hoc demos.
- Per-run artifact capture: sampled human frames, generated code, semantic context, sandbox events, joint-state logs, CoM traces, final robot screenshot, visual summary image, timeline, and result JSON.
- Offline metrics: execution success, safety adherence, fallback count, jerk, and CoM excursion proxy.
- A fair rule-based baseline that uses the same video input, sandbox, and execution path but replaces generative code with `intent label -> fixed code`.
- Optional VLM-as-Judge report generation.
- Portable script paths and Sam's local Webots/macOS env.

Why this is academically stronger than a plain API integration:

- The VLM is not asked to choose a named behavior like `wave`.
- It must emit a low-level physical program with continuous parameters such as IK targets, oscillation frequency, amplitude, duration, decay, and head angle.
- The generated program is statically checked by an AST validator before motor writes are allowed.
- The execution layer logs physical traces, so the result can be judged by both semantic appropriateness and robot-motion metrics.
- The baseline comparison directly tests the professor's concern: whether the system is more than rule-based animation.

## CaP Vs Rule Baseline

CaP means Code-as-Policies. In this repo, CaP is the `--method cap` path:

1. The controller samples multiple frames from the human video.
2. GPT-4o receives the frame sequence plus the allowed robot primitive API.
3. The model returns a semantic context object and executable Python primitive code.
4. The sandbox validates the code, then Webots executes it on the NAO model.

Example CaP output for a wave:

```python
move_arm_ik('right', xyz=[0.15, -0.15, 0.1], duration=0.4)
oscillate_joint('RElbowRoll', center=1.0, amplitude=0.5, frequency=2.0, duration=1.5, decay=0.3)
move_arm_ik('right', xyz=[0.06, -0.1, -0.16], duration=0.4)
```

The rule baseline is `--method rule_baseline`:

1. The same frames enter the same pipeline.
2. A classifier produces a single label such as `wave`, `stop`, `approval`, or `disagreement`.
3. `evaluation/rule_baseline.py` maps that label to fixed prewritten code.
4. The same sandbox and Webots execution path runs that code.

This is intentionally a strong but limited baseline. It can succeed on simple gestures, but it cannot naturally adapt amplitude, timing, side choice, or motion composition to subtle context unless those cases are manually added to the lookup table. CaP should win by producing more context-sensitive, varied, and appropriate primitive programs while preserving safety.

The first smoke tests mean:

- `cap_20260510_232720` succeeded on `pilot_waving`: execution success 1.0, safety adherence 1.0, no fallback, average jerk around 1.23, CoM XY excursion about 7 mm.
- `rule_baseline_20260510_232547` also succeeded on the same easy clip: execution success 1.0, safety adherence 1.0, no fallback, average jerk around 2.21, CoM XY excursion about 9.6 mm.
- This proves the harness and both methods work on one clip. It does not yet prove CaP beats the baseline. That requires the pilot/canonical benchmark.


## What The Two JSONs Show

Both runs succeeded.

`cap_20260510_232720.json`:
- Human interpreted as waving hello.
- VLM generated primitive code directly: `move_arm_ik(...)`, `oscillate_joint(...)`, `move_arm_ik(...)`.
- Execution succeeded.
- Safety adherence: `1.0`.
- Fallbacks: `0`.
- Average jerk: `1.23`.
- CoM XY excursion: about `7 mm`.

`rule_baseline_20260510_232547.json`:
- Baseline classified intent as `wave`.
- Then used a fixed prewritten wave program.
- Execution succeeded.
- Safety adherence: `1.0`.
- Fallbacks: `0`.
- Average jerk: `2.21`.
- CoM XY excursion: about `9.6 mm`.

Interpretation: the harness works, both methods are safe on the easy waving case, and CaP produced a slightly smoother/lower-CoM-motion response in this one run. But one waving run does **not** prove CaP beats baseline. We need all pilot/canonical scenarios and judge scores.

## Proposal Gap And Competition Risk

The original approved proposal promised:

- Simulation-first humanoid embodied interaction.
- A custom simplified URDF in Gazebo/RViz.
- ROS 2/ros2_control style hierarchy.
- VLM-driven Code-as-Policies, not fixed animation lookup.
- Safety sandbox and fallback policy.
- Quantitative evaluation with jerk, CoM stability, VLM-as-Judge, canonical scenarios, and a rule-based baseline.

Current reality:

- The custom URDF/Gazebo/ROS stack changed to NAO/Webots/Python.
- The core embodied-intelligence claim survived: VLM -> physical primitive code -> safety-constrained execution -> measurable robot response.
- The biggest missing proposal items are not "more LLM calls"; they are complete evaluation, canonical videos, support-polygon stability, and a clear report narrative explaining why Webots Controller replaces ROS 2 for this implementation.

Do not bolt on rushed RL just to look complex. That would likely be weaker than a finished, well-evaluated embodied interaction system. Better additions:

- Run the full pilot benchmark across the 10 existing debug clips.
- Record the canonical 8 proposal videos and run the canonical benchmark.
- Add VLM-as-Judge results with cached rationales.
- Add an ablation: single-frame vs multi-frame VLM input.
- Add a diversity metric: count unique primitive sequences and parameter variation across scenarios/rounds, showing CaP is not just repeating a fixed animation.
- Add a simple support-polygon/stability approximation if time allows.
- Add side-by-side demo recordings: source video on the left, Webots robot on the right, with saved `demo_summary.jpg` and `timeline.json`.

Competitiveness narrative:

- Hardware teams may look flashy, but this project can score well if the final story is about embodied AI architecture, safety, physical control, and reproducible evaluation.
- The professor's likely criticism is not "no ROS"; it is "is this more than a rule-based demo?" The CaP-vs-baseline benchmark is the answer.
- The report must emphasize physical constraints, generated motion programs, measurable robot dynamics, and benchmark comparison.


## Your Concern Is Legit

If we present this as “GPT sees video, outputs robot function calls,” it may sound like basic integration.

But the actual project is stronger than that. The correct framing is:

- Multi-frame temporal VLM perception, not single image classification.
- Code-as-Policies over low-level physical primitives, not behavior labels.
- Pinocchio IK and CoM logging.
- AST sandbox with joint-limit, lower-body, IK-radius, duration, oscillation checks.
- Async VLM worker plus idle latency masking.
- Three-tier fallback policy.
- Reproducible benchmark with rule baseline, motion metrics, and VLM-as-Judge.

That is not simplistic integration. But we need to **prove it with evaluation**, not just describe it.


## Phase Summary

Phase 1: Motion primitives  
Added low-level robot actions: joint moves, multi-joint moves, IK arm movement, oscillation, hold, min-jerk trajectories.

Phase 2: Interaction loop  
Added idle animation, rolling frame buffer, VLM worker thread, and state-aware trigger so the robot does not freeze during inference.

Phase 3: Prompt and behavior generation  
Moved away from named canned actions. The VLM now receives physical primitives and generates parameterized code.

Phase 4: Safety and robustness  
Added sandbox validation, joint typo tolerance, forbidden lower-body constraints, fallback retry/replay/idle policy, and local VLM hooks.

Phase 5: Evaluation  
This is what we added: metrics recorder, benchmark runner, scenario registry, rule baseline, judge scaffolding, portable paths, Mac Webots env, and successful baseline/CaP smoke tests.

## Phase 5 Additions

The evaluation layer lives in `evaluation/`:

- `evaluation/scenarios.py`: pilot scenarios from `debug_video_samples/` plus canonical future scenarios from `videos/scenario_*.mp4`.
- `evaluation/metrics.py`: computes execution success, safety adherence, fallback rate, jerk proxy, and CoM excursion proxy.
- `evaluation/run_benchmark.py`: launches Webots once per scenario/round/method and aggregates JSON/CSV outputs.
- `evaluation/rule_baseline.py`: fixed-code baseline that uses the same sandbox and execution path as CaP. It can fall back to scenario-id heuristics when no OpenAI key is available.
- `evaluation/judge.py`: optional VLM-as-judge report with cached judgments.

The controller metrics hook lives at:

- `nao_VLM/controllers/nao_vlm_test/metrics_recorder.py`

When `METRICS_RUN_ID` is set, each run should write:

- `artifacts/oneshot/<run_id>/joint_states.jsonl`
- `artifacts/oneshot/<run_id>/sandbox_events.jsonl`
- `artifacts/oneshot/<run_id>/input_contact_sheet.jpg`
- `artifacts/oneshot/<run_id>/demo_summary.jpg`
- `artifacts/oneshot/<run_id>/timeline.json`
- `artifacts/oneshot/<run_id>/robot_response.png`
- `artifacts/oneshot/<run_id>/result.json`

Benchmark aggregates go under `artifacts/eval/`.

## Environment Variables

Core run mode:

```bash
RUN_MODE=oneshot              # benchmark/default evaluation mode
RUN_MODE=periodic             # live webcam/demo loop
RUN_MODE=replay               # execute precomputed code
INPUT_MODE=webcam             # webcam, video file, or MJPEG source
WEBCAM_SOURCE=0               # native camera
WEBCAM_SOURCE=videos/foo.mp4  # prerecorded file
ONE_SHOT_POST_EXECUTION_SECONDS=5  # visual demo hold; benchmark default is 0
```

Phase 5 metadata:

```bash
METRICS_RUN_ID=<unique-run-id>
METRICS_OUTPUT_DIR=artifacts/oneshot/<unique-run-id>
EVAL_SCENARIO_ID=pilot_waving
EVAL_METHOD=cap               # or rule_baseline
```

OpenAI/cloud VLM:

```bash
OPENAI_API_KEY=<key>
# or the existing teammate-compatible name:
llm_api_key=<key>
VLM_BACKEND=openai
VLM_MODEL=gpt-4o
```

Local/open-source VLM is optional and more experimental:

```bash
VLM_BACKEND=local
LOCAL_VLM_SERVER_URL=http://127.0.0.1:8000
LOCAL_VLM_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

Implementation and unit tests do not need an API key. CaP benchmark runs and judge reports do need a cloud key unless a local VLM server is explicitly being used. OpenAI API calls are paid usage; there is no free hosted GPT-OSS model in this repo's default path.

## How To Test

Start with pure local checks. These do not require Webots, ROS, or an API key:

```bash
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m unittest tests/test_phase5_evaluation.py
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m py_compile \
  evaluation/*.py \
  nao_VLM/controllers/nao_vlm_test/config.py \
  nao_VLM/controllers/nao_vlm_test/metrics_recorder.py \
  nao_VLM/controllers/nao_vlm_test/nao_vlm_test.py \
  nao_VLM/controllers/nao_vlm_test/sandbox_exec.py
bash -n run_example_video_demo.sh scripts/*.sh
python3 -m evaluation.run_benchmark --help
python3 -m evaluation.judge --help
```

Then verify Webots availability:

```bash
which webots
/Applications/Webots.app/Contents/MacOS/webots --version
```

If Webots is not on PATH:

```bash
export WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots
```

Run a no-key baseline smoke test once Webots is available:

```bash
WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots \
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m evaluation.run_benchmark \
  --scenario-set pilot \
  --scenarios pilot_waving \
  --rounds 1 \
  --method rule_baseline \
  --headless
```

Run a side-by-side visual demo with human video playback and Webots:

```bash
export OPENAI_API_KEY=<your-key>
VLM_BACKEND=openai ONE_SHOT_POST_EXECUTION_SECONDS=5 \
  bash scripts/launch_side_by_side_demo.sh \
  debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4
```

This opens `ffplay` for the human source video and Webots for the robot. It also writes a metrics/artifact run under `artifacts/oneshot/visual_demo_*`.

Run one cloud CaP smoke test after setting an API key:

```bash
export OPENAI_API_KEY=<your-key>
WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots \
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m evaluation.run_benchmark \
  --scenario-set pilot \
  --scenarios pilot_waving \
  --rounds 1 \
  --method cap \
  --headless
```

Generate a judge report after benchmark JSON files exist:

```bash
python3 -m evaluation.judge artifacts/eval/*.json --output artifacts/eval/report.md
```

Expected success conditions:

- Every benchmark run writes a valid `result.json`.
- `joint_states.jsonl` contains enough samples for jerk calculation.
- `sandbox_events.jsonl` records validation/execution events.
- `input_contact_sheet.jpg` shows the human frames sent to the model.
- `demo_summary.jpg` shows those frames above the final Webots robot screenshot.
- `timeline.json` shows when frames were sampled, when the VLM call started, when the response returned, and when robot execution completed.
- Metrics contain numeric values, not NaNs.
- Safety adherence remains 100%.
- Initial target: CaP execution success above 80%, fallback activation below 20%, judge appropriateness at least 75%, and CaP better than the rule baseline.

## Pilot Benchmark Meaning

The canonical benchmark is the original proposal's 8 phone-recorded scenarios:

1. Wave.
2. Cross arms.
3. Lean forward.
4. Walk away.
5. Crouch.
6. Reject gesture.
7. Greeting/handshake.
8. Idle standing.

Those canonical files do not yet exist locally under `videos/scenario_*.mp4`. The repo currently has 10 existing pilot clips under `debug_video_samples/` such as waving, pointing, stop, thumbs-up, yes-nod, no-shake, clap, beckon, and shrug. "Run the pilot benchmark now" means use those existing clips immediately to test the harness and compare CaP vs baseline before spending time recording the final canonical 8.

Pilot commands:

```bash
WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots \
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m evaluation.run_benchmark \
  --scenario-set pilot --rounds 1 --method rule_baseline --headless

export OPENAI_API_KEY=<your-key>
WEBOTS_BIN=/Applications/Webots.app/Contents/MacOS/webots \
/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm/bin/python -m evaluation.run_benchmark \
  --scenario-set pilot --rounds 1 --method cap --headless
```

After pilot passes, record/import the canonical 8 and repeat with `--scenario-set canonical`.


## Why Webots Quits Automatically

Yes, that is designed. In benchmark mode we run `RUN_MODE=oneshot`, so Webots opens, samples the video, calls either `cap` or `rule_baseline`, executes one robot response, writes artifacts, then exits via `simulationQuit(0)`. Benchmark mode defaults to 0, so it exits fast.

Visual demo can use 5, so Webots stays open briefly after actuation. Closing Webots manually during that hold is legitimate.

Why: automated evaluation needs one clean process per scenario. If Webots stayed open, `run_benchmark.py` would hang and you could not run 10 videos × 3 rounds × 2 methods.

For a visual/live demo, use `RUN_MODE=periodic` or set `ONE_SHOT_EXIT_AFTER_EXECUTE=0` when launching manually. Benchmarks should quit.

## Transparency Notes

The Webots camera view is not the human input video. It is the simulated NAO / world camera. The human video is read off-screen by OpenCV through `FrameBuffer`, sampled into base64 JPEGs, and sent to the VLM. This is why the Webots window can show the world horizon while the VLM still sees the human clip.

Low OpenAI token counts are expected. The system sends a compact prompt plus `VLM_FRAME_COUNT` low-detail images. `VLM_IMAGE_DETAIL='low'` is explicitly set to keep calls cheap. A one-clip smoke test around a few thousand input tokens is normal. Important: the Webots camera showing the horizon is expected. That is the simulated robot/world view, not the human video feed. The human video is read off-screen by OpenCV and sent as low-detail sampled frames to OpenAI. That also explains the low token count: 5 low-detail images plus compact prompt, so ~3.5K input tokens per call is plausible.



For human inspection, use the generated artifacts:

- `frame_01.jpg` ... `frame_05.jpg`: exact frames sent to the VLM.
- `input_contact_sheet.jpg`: all sampled input frames in one image.
- `robot_response.png`: final Webots screenshot.
- `demo_summary.jpg`: input frame sheet plus final robot screenshot.
- `timeline.json`: elapsed seconds for sampling, VLM call, execution, and exit.
- `python_code.py`: exact primitive program executed.
- `semantic_context.json`: model-interpreted intent and robot intent.

Benchmark mode should exit automatically. Visual demo mode should set `ONE_SHOT_POST_EXECUTION_SECONDS=5`, which keeps Webots open briefly after actuation; if the user closes Webots earlier, the controller exits naturally.


## What Still Remains

Most important:
1. Record or add the canonical 8 scenario videos from the proposal.
2. Run full benchmark: `pilot` now, canonical later, 3 rounds each.
3. Run VLM-as-Judge and produce a report table.
4. Show CaP vs rule baseline across scenarios.
5. Add plots/tables for execution success, safety, fallback rate, jerk, CoM excursion.
6. Write the report narrative around “generative embodied policy under safety constraints,” not “LLM API integration.”

High-value optional upgrades:
- Add a simple support-polygon CoM stability check.
- Add a diversity metric showing CaP generates varied primitive programs while baseline repeats fixed code.
- Add an ablation: single-frame vs multi-frame perception.
- Add a state-aware trigger cost analysis: fewer VLM calls than periodic polling.
- Add short live-camera demo after fixed-video benchmark is done.

I would **not** bolt on RL now. It would likely look rushed. The stronger play is to finish the proposal’s evaluation promises and make the sophistication visible.

## Live Camera Demo

Native Linux webcam:

```bash
WEBCAM_SOURCE=0 bash scripts/run_live_camera_demo.sh
```

Laptop camera streamed to a remote Linux session (e.g. a cloud VM running Webots):

```bash
python3 scripts/local_camera_server.py --source 0 --port 5000 --fps 10
```

Use this source on the remote Linux session after setting up an SSH reverse tunnel:

```bash
WEBCAM_SOURCE=http://127.0.0.1:5000/video_feed
```

## Mac Reality Check

Do not treat Webots as the same problem as ROS/Gazebo on macOS.

ROS 2 + Gazebo tends to be painful on macOS because ROS support and robotics sim dependencies are much more Linux-centered. This repo's current stack avoids that by using Webots directly with a Python controller. Webots itself has a Mac application, so native Mac use is plausible for editing and visual demos.

However, the benchmark still depends on Webots launching the controller with a Python environment that has Pinocchio, NumPy, OpenCV, OpenAI, and other deps. That means Mac can still fail on Python binary/dependency issues even though it is not a ROS/Gazebo issue. For reproducible grading/report runs, a teammate's native Linux machine remains cleaner.

As of 2026-05-10 on Sam's Mac:

- Dedicated controller env exists at `/Users/SamarthSoni/miniforge3/envs/humanoid_webots_vlm`.
- `nao_VLM/controllers/nao_vlm_test/runtime.ini` points Webots at that env's Python. It is machine-specific and gitignored.
- `webots` is not on PATH.
- `/Applications/Webots.app/Contents/MacOS/webots --version` reports `R2025a`.
- Pure Python Phase 5 tests pass locally.
- One no-key rule-baseline Webots smoke test passed locally: `rule_baseline_20260510_232147__rule_baseline__pilot_waving__r01` had `status=ok`, `webots_returncode=0`, `execution_success=1.0`, `safety_adherence=1.0`, and 940 joint-state samples.

## Collaboration Rules

- Preserve teammate native Linux usage. Prefer env overrides (`WEBOTS_BIN`, `PYTHON_BIN`, `REPO_DIR`) over hardcoded machine paths.
- Do not reintroduce Darian-specific paths such as `/home/darian/...`.
- Keep Mandarin original text if it is useful, but add English summaries around it. Work in English by default.
- Do not commit generated benchmark artifacts from `artifacts/eval/` or `artifacts/oneshot/` unless explicitly requested. The historical convention is to share only selected demo recordings under matched screen-recording paths.
- Do not revert unrelated user or teammate changes in the worktree.
- Keep primitive names and sandbox invariants stable unless the user explicitly asks for an API change.
