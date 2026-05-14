# NAO VLM Embodied Agent

A simulated humanoid embodied-AI system. A Vision-Language Model (VLM) captures video clips of a human and **generates low-level robot motion code** that
a NAO H25 humanoid executes in the Webots simulator. The VLM does not pick from
canned animations — it composes motion from physical primitives, and a safety
sandbox validates every program before it reaches the motors.

## A. Platform & Installation

**Stack**
- **Simulator:** Webots R2025a (the world file declares `#VRML_SIM R2025a`)
- **Robot:** Aldebaran NAO H25 (25-DoF); URDF assets bundled under `nao_VLM/nao_robot/`
- **Kinematics:** Pinocchio (inverse kinematics + centre-of-mass)
- **VLM:** OpenAI GPT-4o by default (an optional local VLM backend exists — see `AGENTS.md`)
- **OS / runtime:** Ubuntu 22.04, Python 3.10+

**How it works**
- Webots runs the physical body and collisions; Pinocchio runs the kinematics maths.
- The VLM composes motion from primitives — `move_joint`, `move_joints`,
  `move_arm_ik`, `move_head`, `set_hand`, `oscillate_joint`, `hold`, `idle` —
  not from named canned actions.
- A sandbox AST-validates every VLM-generated program (joint limits, forbidden
  lower-body joints, parameter bounds) before any motor write.

### 1. Install Webots
Download the Webots R2025a Debian package for Ubuntu 22.04 from the
[Cyberbotics releases](https://github.com/cyberbotics/webots/releases) and install it:
```bash
sudo apt install ./webots_2025a_amd64.deb
```

### 2. Install Python dependencies
A virtualenv is the simplest path:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> `requirements.txt` installs the `pin` package for Pinocchio — **not**
> `pinocchio` from PyPI (that's an unrelated library). The import name is
> still `import pinocchio`.

Conda is also supported: `conda env create -f environment.yml`.

### 3. NAO URDF
Already bundled under `nao_VLM/nao_robot/` — Pinocchio reads it for IK and CoM.
No action needed.

## B. Quick Start (run the demo)

After installing Webots and the Python dependencies (Section A):

### 1. Point Webots at your Python — `runtime.ini`
Webots launches the controller with its own Python unless told otherwise. Copy
the template and edit it:
```bash
cp nao_VLM/controllers/nao_vlm_test/runtime.ini.example nao_VLM/controllers/nao_vlm_test/runtime.ini
```
Edit the `COMMAND =` line to the **absolute path** of the Python interpreter
that has the project dependencies (e.g. your `.venv/bin/python3` or conda env
python). `runtime.ini` is gitignored — it is per-machine.

### 2. Configure `.env`
```bash
cp .env.example .env
```
Set `llm_api_key=` to your OpenAI key. `VLM_BACKEND=openai` is the default.

### 3. Choose an input mode
The controller reads its input source from `WEBCAM_SOURCE` in `.env`.

**Example-video mode** — works everywhere; recommended for a first run or on a VM.
`.env.example` already ships in this mode (`RUN_MODE=oneshot`, `WEBCAM_SOURCE`
pointing at a sample clip). To use a different clip:
```
RUN_MODE=oneshot
WEBCAM_SOURCE=debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4
```
The controller samples a few frames from the video, calls the VLM once,
executes the returned motion once, writes artifacts under `artifacts/oneshot/`,
and exits.

**Local webcam mode** — native Linux with a webcam:
```
RUN_MODE=periodic
WEBCAM_SOURCE=0
```
The controller continuously samples the webcam and responds in a loop.
> **VirtualBox note:** a VirtualBox Ubuntu guest does **not** see the host's
> webcam by default — `/dev/video0` will not exist. A native (bare-metal)
> Ubuntu install has the webcam directly. On a VM you have three options:
> 1. **Example-video mode** — skip the webcam entirely (recommended for a VM).
> 2. **USB passthrough** — VirtualBox *Devices → Webcams → \<your camera\>*
>    forwards the host webcam into the guest; then keep `WEBCAM_SOURCE=0`.
> 3. **Stream from the Windows host** — on the host run
>    `python scripts/local_camera_server.py --source 0 --host 0.0.0.0`, then in
>    the guest set `WEBCAM_SOURCE=http://10.0.2.2:5000/video_feed`
>    (`10.0.2.2` is the host as seen from a VirtualBox NAT guest). Allow Python
>    through the host firewall on first run.

### 4. Launch Webots and run
```bash
webots nao_VLM/worlds/nao_VLM.wbt
```
Press Reset (⏪) then Play (▶️). The `nao_vlm_test` controller starts
automatically and drives the robot.
> The bundled world already has the NAO node's `supervisor` field set to
> `TRUE` (the controller needs it for screenshots and a clean exit in oneshot
> mode). If you see supervisor-related errors, re-check that field in the
> Webots Scene Tree.

## C. Benchmark Evaluation

The evaluation framework runs the controller across a set of scenario videos,
computes metrics (execution success, safety adherence, joint jerk, CoM
stability, fallback rate), and can score responses with a VLM-as-Judge —
comparing the generative system (`cap`) against a rule-based baseline.

### One command (recommended)
```bash
python -m evaluation.run_benchmark --scenario-set pilot --method both --judge
```
Runs both methods over every scenario, then judges — producing, under
`artifacts/eval/`:
- `both_<timestamp>.json` — every per-run result
- `both_<timestamp>.csv` — flat metrics table (compare rows side by side)
- `both_<timestamp>_report.md` — VLM-as-Judge verdicts

Add `--realtime` to watch runs at normal speed (default is fast batch mode);
add `--headless` for windowless runs.

### Or run the pieces separately
```bash
python -m evaluation.run_benchmark --scenario-set pilot --method rule_baseline
python -m evaluation.run_benchmark --scenario-set pilot --method cap
python -m evaluation.judge artifacts/eval/<exact_file>.json --output artifacts/eval/report.md
```

### Scenario sets
- **`pilot`** — 10 sample gesture clips already in `debug_video_samples/`.
  Ready to run now.
- **`canonical`** — the 8 project benchmark scenarios. Record these clips and
  place them in the `videos/` folder with these exact names:

  | File | Scenario |
  |---|---|
  | `videos/scenario_01_wave.mp4` | wave |
  | `videos/scenario_02_cross_arms.mp4` | crossed arms (rejection) |
  | `videos/scenario_03_lean_forward.mp4` | lean forward (approach) |
  | `videos/scenario_04_walk_away.mp4` | walk away |
  | `videos/scenario_05_crouch.mp4` | crouch |
  | `videos/scenario_06_reject.mp4` | reject gesture |
  | `videos/scenario_07_handshake.mp4` | handshake greeting |
  | `videos/scenario_08_idle.mp4` | idle standing |

  Do **not** put them in `debug_video_samples/` — that is the separate pilot
  set. The benchmark skips any canonical video not yet recorded, so they can
  be added incrementally. The scenario registry (intents, expected responses)
  lives in `evaluation/scenarios.py`.

Then run with `--scenario-set canonical` (or `--scenario-set all`).

The per-run `result.json` schema is documented in `evaluation/RESULT_SCHEMA.md`.

## D. Project Plan & Documentation Reference

- **`Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan.md`**
  — the project implementation plan: the original proposal, the gap analysis,
  and the phased build-out (motion primitives → interaction loop → behaviour
  generation → safety → evaluation). Start here to understand *why* the system
  is shaped the way it is.
- **`AGENTS.md`** — deep technical handoff for contributors: architecture, the
  cap-vs-baseline design, environment variables, testing.
- **`COLLABORATOR_GUIDE.md`** — day-to-day collaborator runbook: demo scripts,
  recording conventions, and branch workflow.

## Helpful Links
- [Webots official site](https://cyberbotics.com)
- [Webots GitHub](https://github.com/cyberbotics/webots)
- [Webots installation guide](https://cyberbotics.com/doc/guide/installation-procedure#from-the-installation-file)
