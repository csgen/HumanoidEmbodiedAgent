# Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan

## Context

This project is a course group assignment for building a humanoid embodied agent. The original proposal (v2.1) planned a custom primitive-based URDF (9-DoF upper body + wheeled base) in Gazebo + ROS 2. However, the teammate responsible for URDF imported an open-source NAO H25 V5.0 model from the `ros-naoqi/nao_robot` package and runs it in Webots instead. This plan analyzes the gap, responds to the professor's feedback ("appears rule-based rather than true embodied intelligence"), and lays out a revised implementation path.

> **Reading guide.** §1–§14 captures the analysis and per-phase plans we agreed on. §15 is the original Phase 3 + 4 plan. **§16 is a retrospective showing what is now done**; the Phase 3 + 4 work plus a cleanup iteration are merged on `main`. **§17 is the active forward plan for Phase 5** — that's where the next teammate should focus. §18 lists out-of-scope items.

---

## 1. High-Level Comparison

| Original Proposal | Current Implementation | Deviation |
|---|---|---|
| Custom primitive URDF (9-DoF + wheeled base) | NAO H25 V5.0 open-source URDF (25-DoF biped) | **Major** |
| Gazebo simulation + RViz visualization | Webots simulation | **Platform change** |
| ROS 2 architecture (ros2_control, joint_trajectory_controller) | Webots Controller API (no ROS 2) | **Major** |
| Code-as-Policies VLM integration | `vlm.py` + `sandbox.py` (partial) | Direction aligned but disconnected |
| Hierarchical control (high/low-freq decoupling) | Direct motor control | **Not implemented** |
| Automated evaluation framework | None | **Not implemented** |

---

## 2. Already Implemented (at the start of this planning)

### 2.1 VLM Perception + Code-as-Policies Skeleton (`vlm.py`)
- Webcam image capture + base64 encoding
- OpenAI GPT-4o API call
- Structured prompt design with intermediate semantic object: `{intent, social_distance, affect, confidence}`
- VLM output parsing (extracts JSON block + Python code block)
- **Matches the intent of proposal §2 "Generative Behavior Policy via Code-as-Policies"**

### 2.2 Safety Sandbox (`sandbox.py`)
- Restricted builtins, blocks `import`/`open`/`eval`
- Whitelist-based function execution
- Exception capture → fallback trigger
- **Matches proposal §3.3 "Safety Sandbox", but mocked**

### 2.3 Robot Control API (`nao_vlm_test.py` — `NaoVlmAPI`)
- `look_at()` — 2-DoF head control
- `move_arm()` — Cartesian IK via Pinocchio
- `operate_gripper()` — finger open/close
- `set_posture()` — whole-body preset (stand / squat)
- `navigate_to()` — supervisor-mode translation (macroscopic navigation)
- `capture_camera_image()` — NAO `CameraTop` capture
- `speak()` — text-to-speech (mock print)
- Validated via an 8-step automated test sequence in the Webots world

### 2.4 Walking Gait (`nao_walk.py`)
- Open-loop CPG-style gait
- Real-time CoM tracking via Pinocchio
- Was originally a stretch goal — already delivered

---

## 3. Core Problem: Impact of Using NAO URDF

### 3.1 Different Kinematic Structure
- **Proposal**: 9-DoF upper body (Head 2 + Arms 2×2 + Torso 3) + wheeled base
- **NAO**: 25-DoF (Head 2 + Arms 5×2 + Legs 6×2 + Hands 1×2)
- **Consequences**:
  - **No torso DoF** — NAO's torso is rigid; no pitch/roll/yaw joints. The proposal's core selling point (lean-forward greeting, sway-while-idle) cannot be done directly
  - **No wheeled base** — biped locomotion, completely different
  - **More arm DoF** — 5-DoF per arm is more expressive than the 2-DoF proposed. Actually an upside

### 3.2 VLM Prompt ↔ Actual API Mismatch (resolved during Phase 0–4)
- `vlm.py` prompt exposed: `set_torso_pitch()`, `blend_arm_pose()`, `play_idle_animation()`
- `nao_vlm_test.py` actually implemented: `look_at()`, `move_arm()`, `set_posture()`, `operate_gripper()`, `navigate_to()`
- **The two API sets do not overlap**. VLM-generated code cannot execute on Webots
- This was resolved by replacing both with the Motion Grammar primitives (§7) and removing legacy duplicates from sandbox registration during the cleanup iteration (§16.3).

### 3.3 Simulation Platform and Middleware Change
- **Gazebo → Webots**: functionally equivalent, low impact
- **ROS 2 → Webots Controller**: biggest architectural deviation
  - No `ros2_control` / `joint_trajectory_controller`
  - No high/low-frequency decoupling (VLM at 0.5-1 Hz vs. control at 100 Hz)
  - No spline interpolation for trajectory smoothing
  - Direct `motor.setPosition()` produces the "mechanical jitter" the proposal aimed to avoid

---

## 4. Per-item Feasibility Assessment

### 4.1 Can continue as-is
| Proposal Item | Reason |
|---|---|
| VLM → Code generation loop | Core architecture exists, just needs API alignment |
| Intermediate Semantic Structure | Already implemented in `vlm.py` |
| Safety Sandbox | Skeleton exists, needs hardening (joint limits, CoM check) |
| Three-tier fallback policy | Basic exception catching exists, needs completion |
| VLM-as-Judge evaluation | Platform-independent |
| 8 canonical test scenarios | Platform-independent |
| Rule-based baseline comparison | Platform-independent |
| Execution Success Rate metric | Platform-independent |
| Camera perception | NAO `CameraTop` already bound |

### 4.2 Requires modified design
| Proposal Item | Why | Recommended Path |
|---|---|---|
| 3-DoF torso semantic expression | NAO has no torso joints | Compose arm + head + whole-body posture. NAO's 5-DoF arm is more expressive than proposed 2-DoF |
| `set_torso_pitch()` VLM API | Doesn't exist on NAO | Either (a) simulate via `HipPitch + Knee + Ankle` compensation (bow-like motion), or (b) drop and replace with arm-based semantics |
| VLM prompt API list | Mismatch with actual controller | Rewrite prompt around real NAO kinematics + primitive operators (see §7) |
| `sandbox.py` whitelist | Mock functions | Inject `NaoVlmAPI` bound methods into the sandbox globals |
| Wheeled base locomotion | NAO is biped | Keep `navigate_to()` supervisor workaround for macroscopic motion; use `nao_walk.py` gait if needed |
| ROS 2 `ros2_control` | No ROS 2 | Option A: build a spline-interpolation layer in Webots that plays the same role. Option B: argue Webots Controller provides equivalent joint control |
| High/low-freq decoupling | Direct control currently | Implement it: VLM generates parameters → queue → interpolated executor (async) |
| Latency masking (idle) | Not implemented | Needs idle primitive + background thread (see §9) |
| CoM stability monitoring | Needs support polygon | CoM is available via Pinocchio (already used in `nao_walk.py`); support polygon needs foot contact detection |
| Joint jerk smoothness metric | Needs trajectory logging | Log `joint_states` per step, compute 3rd-order finite differences offline |
| Self-collision check | Not implemented | Pinocchio supports collision if URDF has `<collision>` tags (NAO V5.0 URDF does) |

---

## 5. Summary of the URDF Situation

The teammate's choice of NAO URDF causes **"form change but soul intact"**:
- Every core innovation (VLM Code-as-Policies, safety sandbox, hierarchical control, evaluation framework) is **implementable on NAO**
- Main loss: 3-DoF torso expressiveness — partially compensated by NAO's richer arms
- Webots vs Gazebo is a minor change
- Biggest technical debt: **ROS 2 architecture absence** and **VLM prompt ↔ actual API disconnect** (the latter resolved in Phase 0–4)

Points to explain in the final report:
- Why NAO instead of custom URDF (industrial-grade humanoid, demonstrates generality on a real robot)
- Why Webots instead of Gazebo (native NAO support, faster dev)
- Why no ROS 2 (Webots Controller provides equivalent joint control; ROS 2 is future work)

---

## 6. [DEPRECATED] Pose Library Approach (kept for history)

An earlier iteration proposed a named pose library with `blend_arm_pose('wave', duration)`. This was **rejected by professor's feedback**:

> current scope relies mainly on predefined behaviors, basic position control, and simple intent triggers ... may make it appear like a rule-based animation system rather than a true embodied intelligence project

A named pose library is effectively a lookup table and falls under "rule-based animation". Superseded by §7.

---

## 7. Motion Grammar: Physical Primitives, Not Behavior Recipes

### 7.1 Design Principle
- **No semantic names** in the API — no `wave`, `greet`, `reject`, `hug`
- Expose control-theoretic composition operators instead: `move_joint`, `oscillate_joint`, `move_arm_ik`, `move_joints`
- Prompt **teaches VLM physical principles + NAO kinematics**, letting it **construct** motions on the fly
- Single exception: `idle` is kept as a system-level fallback (and implemented as a composition of primitives itself — so the "everything is primitive" principle still holds)

### 7.2 API V2 (replaces the V1 pose library)

**Perception:**
```python
get_robot_state() -> dict           # joint positions + CoM + contact
get_joint_limits() -> dict          # from URDF/Webots, injected into prompt
capture_image() -> base64
```

**Motion primitives (blocking, internally call `robot.step()`):**
```python
move_joint(name, angle, duration, trajectory='cubic')
move_joints(joint_dict, duration, trajectory='cubic')
move_arm_ik(side, xyz, duration, orientation=None)
move_head(yaw, pitch, duration, trajectory='min_jerk')      # added during Phase 4 cleanup
set_hand(side, openness, duration, trajectory='cubic')      # added by teammate Phase 3+4
oscillate_joint(name, center, amplitude, frequency, duration, decay=0.0)
hold(duration)
```

**System-level (not part of VLM's everyday toolkit):**
```python
idle(duration)                      # finite-duration primitive composition
start_idle_loop() / stop_idle_loop()   # background latency masking
speak(text)                         # registered but not advertised in prompt (mock TTS)
```

### 7.3 Prompt Philosophy Shift
- ❌ "Call `wave('decaying')` to wave at someone"
- ✅ "NAO joints: [RShoulderPitch (-2.08, 2.08), RElbowRoll (0.03, 1.54), ...]. To create waving: raise shoulder (RShoulderPitch ~-0.8) then oscillate elbow (RElbowRoll center 1.0, amp 0.5, freq 2 Hz). Amplitude/frequency should adapt to social context."

### 7.4 Example: "Decaying wave" under V2 API (generated by VLM)
```python
# VLM reasoning: raise arm, oscillate elbow with decay, lower arm
move_arm_ik('right', xyz=[0.15, -0.15, 0.10], duration=0.4)
oscillate_joint('RElbowRoll', center=1.0, amplitude=0.6,
                frequency=2.0, duration=2.0, decay=0.5)
move_arm_ik('right', xyz=[0.0, -0.1, -0.1], duration=0.4)
```
VLM is genuinely **programming** — it must understand NAO kinematics, choose frequency/amplitude parameters sensibly. This is what the professor wants by "generative" rather than "predefined".

---

## 8. Implementation Difficulties Against Current Code

### 8.1 Critical: Three modules are disconnected (resolved Phase 0)

**Original state**:
- `vlm.py` used `cv2.VideoCapture(0)` → **laptop webcam, not NAO's eye**
- `sandbox.py` was a standalone script with mocked functions
- `nao_vlm_test.py` was the Webots controller with the real robot/camera/motors
- **No data flow between the three**

**Resolution (Phase 0)**: VLM call and sandbox execution moved into the Webots controller; FrameBuffer abstracts video source per `WEBCAM_SOURCE`; cv2.VideoCapture handles file / device / MJPEG URL transparently.

### 8.2 Medium difficulties (resolved Phase 0–2)

**A. Integrating blocking primitives with the Webots main loop** — primitives now internally drive `robot.step()` (model: existing `navigate_to()`). Main loop handles sensor sync + idle heartbeat + command dispatch.

**B. VLM HTTP call freezes simulation → latency masking** — `VLMWorker` runs in a daemon thread; main loop continues stepping Webots; idle animator runs during inference.

**C. Unpredictable action duration + interruption** — MVP: serial queue, no preemption. State-aware trigger blocks `EXECUTING` state from queuing new requests.

**D. Pinocchio state sync timing** — `_sync_sensors()` extracted as helper; called inside every primitive's step loop.

### 8.3 Minor difficulties

| Item | Effort | Status |
|---|---|---|
| E. Add `duration` dim to `move_arm` | Small | ✅ done — `move_arm_ik(side, xyz, duration)` |
| F. Inject joint limits into prompt | Small | ✅ done — `motor.getMin/MaxPosition()` scan injected |
| G. Inject `NaoVlmAPI` bound methods into sandbox | Small | ✅ done — `register_many()` |
| H. Joint-name typo tolerance | Small | ✅ done — `_canonicalize_joint_name()` via difflib |
| I. `trajectory='min_jerk'` (quintic polynomial) | Small-Medium | ✅ done in Phase 0; documented in prompt during cleanup |

### 8.4 Reusable existing assets

| Existing Code | Reuse Value |
|---|---|
| `NaoVlmAPI.move_arm()` Pinocchio IK | Direct kernel for `move_arm_ik` |
| `motor.getMin/MaxPosition()` clipping pattern | Reuse for safety layer |
| `navigate_to()` blocking + `robot.step()` pattern | Template for all primitives |
| `CameraTop` binding + `capture_camera_image()` | One-line substitution for webcam path |
| Supervisor mode | Reuse for evaluation reset |
| `nao_walk.py` CoM calculation | Move to stability checker |
| Sensor sync loop structure | No change needed |
| IK's `J[:, :6] = 0.0` floating-base-lock trick | **Must retain in every IK primitive** |

### 8.5 Gotchas

1. **NAO coordinate frame**: chest-centered, y-negative is right side, meters. Prompt explicitly tells VLM this.
2. **VLM latency 2-6 s**: handled by async daemon thread + idle animator overlay.
3. **Joint name CamelCase**: NAO uses `RShoulderPitch`. `_canonicalize_joint_name()` resolves snake_case / spaced / lowercase variants.
4. **Pinocchio FreeFlyer base (first 6 DoFs)**: kept the `J[:, :6] = 0.0` trick in every IK primitive.

---

## 9. Idle Design

### 9.1 Original state (now obsolete)
- `sandbox.py`'s `api_play_idle_animation()` was just `print(...)`, no actual motion.
- VLM prompt mentioned `'idle'` as a pose name — string only.
- `NaoVlmAPI` had no idle method.

### 9.2 As shipped (Phase 2)

**A. As a VLM-callable primitive (finite duration)** — `idle(duration)` does subtle breathing + small head scanning + gentle shoulder sway. Internal loop calls `robot.step()` so simulation advances during the primitive.

**B. As background overlay (latency masking)** — `IdleAnimator.tick(sim_time)` is called from the main loop every step. It writes small sinusoidal offsets to `HeadYaw` and the two shoulder pitches with low-pass blending so post-primitive transitions don't snap. When a primitive runs, the main loop is suspended inside the primitive's inner step loop, so idle naturally pauses; resumes automatically when the primitive returns.

### 9.3 Idle as fallback (three-tier per proposal — implemented during cleanup, see §16.3)
- Tier A (VLM timeout / parse failure) → re-kick VLM with same evidence.
- Tier B (retry exhausted) → replay last successful primitive composition from history.
- Tier C (no history) → run `idle(2.0)` as a graceful default.

---

## 10. Visual Input: Flexible Source Architecture

### 10.1 Deployment context
The controller may run on a **cloud Linux server** (preferred dev environment, better Pinocchio / Webots / dependency story than Windows). Developer's local machine is Windows with the physical webcam. Visual input must therefore support:

- Local webcam (when running fully on laptop)
- Remote webcam streamed from laptop to cloud (for live demo when controller runs on cloud)
- Pre-recorded video file (for reproducible evaluation)
- NAO's simulated `CameraTop` (pure-sim closed loop, optional)

### 10.2 Split the switch into two orthogonal variables

```python
# What the system conceptually reads from:
INPUT_MODE = 'webcam'    # 'webcam' | 'camera_top'

# Where the "webcam" actually comes from (only used when INPUT_MODE='webcam'):
WEBCAM_SOURCE = 0
# 0                                       → local device (webcam / OBS Virtual Cam)
# "http://localhost:5000/video_feed"      → remote MJPEG stream (SSH-tunneled)
# "videos/scenario_01.mp4"                → video file (evaluation)
```

The underlying `FrameBuffer(source=...)` passes directly to `cv2.VideoCapture`, which accepts all of these natively. **No special branching in FrameBuffer.**

### 10.3 Source matrix

| Scenario | `INPUT_MODE` | `WEBCAM_SOURCE` | Notes |
|---|---|---|---|
| Local dev (Windows laptop), live webcam | `webcam` | `0` | Plain webcam path |
| Local dev, loop a recorded video | `webcam` | `"videos/foo.mp4"` | No OBS needed |
| Cloud dev, live human demo | `webcam` | `"http://localhost:5000/video_feed"` | SSH reverse tunnel from laptop |
| Cloud evaluation | `webcam` | `"videos/scenario_XX.mp4"` (uploaded) | No streaming, no OBS |
| Pure-sim validation | `camera_top` | — | NAO's eye reads Webots scene |

### 10.4 Why pre-recorded video (not live humans) for evaluation
- Live gestures vary every run → VLM-as-Judge scores aren't reproducible → ablation impossible
- Rule-based baseline comparison needs identical inputs
- Proposal requires "benchmark for future updates" — **fixed inputs required**
- Pre-recorded video keeps human realism while staying reproducible

### 10.5 Video assets — 8 canonical scenarios
Record 5-10 s real human clips of:
1. Wave
2. Crossing arms
3. Leaning forward (approaching)
4. Walking away
5. Crouching
6. Reject gesture
7. Greeting / handshake gesture
8. Standing still (idle)

Document expected robot response per clip as ground truth.

> Note: `debug_video_samples/` already contains 10 short YouTube-sourced gesture clips (waving, pointing, beckon, finger_no, stop, thumbs_up, yes_nod, no_shake, clap, shrug). They cover ~5 of the canonical scenarios and add 5 extras useful for diversity testing. Phase 5 may use them as a pilot dataset before recording the strict canonical 8.

### 10.6 Cloud streaming setup (live-webcam demo from Windows laptop to cloud)

#### A. Local MJPEG server (runs on Windows laptop)
See `scripts/local_camera_server.py`. Flask MJPEG endpoint at `http://127.0.0.1:5000/video_feed`. Binds localhost only.

#### B. SSH reverse tunnel (while dev session active)
```bash
ssh -R 5000:localhost:5000 user@cloud-server
```
Maps laptop's port 5000 → cloud's `localhost:5000`. All traffic through SSH, zero public exposure.

#### C. Cloud-side controller
```python
WEBCAM_SOURCE = "http://localhost:5000/video_feed"
buffer = FrameBuffer(source=WEBCAM_SOURCE, fps=10, buffer_seconds=2.0)
```

OpenCV's `VideoCapture` handles MJPEG HTTP streams natively — no protocol-specific code on our side.

### 10.7 Cloud-specific logistics

| Concern | Notes |
|---|---|
| Webots display | Headless (`webots --no-rendering`) works for automated evaluation; for visual dev use VNC / X11 forwarding / noVNC |
| GPU | Not needed for Phase 0-5 (Pinocchio + OpenAI API is CPU). Only needed for Phase 6 local VLM (Qwen2-VL) |
| Streaming latency | ~100-300 ms (JPEG encode + network). Within 1.5 s multi-frame window, ~10-20% offset, acceptable |
| Bandwidth | 640×480 JPEG @ 10 fps, quality 70 ≈ 3-5 Mbps upload |
| Security | SSH reverse tunnel keeps webcam on localhost; do **not** expose the Flask server publicly |
| OBS on cloud | Does not work (no display device). Use direct file source instead of OBS Virtual Cam when on cloud |

---

## 11. Multi-frame Temporal Perception: Rolling Buffer + State-aware Trigger

### 11.1 Why single-frame is insufficient
Single frames lose temporal information, causing unresolvable action-level ambiguity:
- Waving hello vs waving goodbye (direction differs, single frame looks identical)
- Reaching out to reject vs reaching out to shake hands (same snapshot)
- Approaching vs retreating (velocity direction invisible)
- Instantaneous pose vs ongoing motion (can't tell "doing" from "just finished")

### 11.2 Architecture

#### A. Rolling `FrameBuffer` (background thread)
Source: `nao_VLM/controllers/nao_vlm_test/frame_buffer.py`. Background daemon thread, deque(maxlen=20) at 10 fps, motion score updated each frame, `sample_recent(n)` returns base64 JPEGs for VLM input.

#### B. GPT-4o multi-image call
The 5 sampled frames are sent as separate `image_url` content items in a single chat completion request. `detail='low'` keeps cost at ~$0.005 per call.

#### C. Prompt — temporal-aware
The prompt declares the inputs as a sequence and requires `motion_dynamics ∈ {oscillatory, approaching, retreating, raising, lowering, static}` in the JSON output, plus the new `robot_intent` field added during cleanup (§16.3).

### 11.3 State-aware Trigger

#### Motivation
A pure timed-fallback (e.g., "fire VLM every 3 s no matter what") burns API cost while nothing is happening. A pure motion-trigger can miss the human's subtle reaction right after the robot finishes an action. A **state machine** resolves both.

Three states:

| State | Meaning | Trigger Rule |
|---|---|---|
| `IDLE` | Robot standing by, no action running | **Motion only** + long-silence safety timeout |
| `EXECUTING` | VLM-generated code running | **Never trigger** |
| `POST_ACTION` | Action just finished | **Fixed-delay trigger once** (actively observe human reaction) |

#### Why this is better
- `IDLE` motion-only → human static ⇒ zero API cost (vs periodic-fallback burning every 3 s)
- `POST_ACTION` fixed-delay → captures subtle reactions (nod, smile, verbal response) that motion-threshold would miss. **This closes the perception-action-feedback loop.**
- `EXECUTING` no-trigger → more semantic than a cooldown timer

#### Implementation
Source: `nao_VLM/controllers/nao_vlm_test/vlm_trigger.py`. Public API: `consider_trigger() → Optional[reason]` (pure query) and `confirm_fire(reason)` (mutates counters/timestamps). The split avoids inflating fire counters when a kick is rejected upstream (e.g., FrameBuffer not yet populated).

#### Cost comparison (vs an old "timed every 3 s" baseline)

| Scenario | Timed | State-aware | Reduction |
|---|---|---|---|
| 1 min idle, no human | 20 calls | 2 calls | 90% |
| 1 min with one full interaction | ~15-20 calls | ~2-3 calls | ~83% |
| **Idle 1 hour** | ~$6 | ~$0.5 | **92%** |
| **Active 1 hour** | ~$6 | ~$1 | **83%** |

#### Tunable parameters (in `config.py`)
- `MOTION_THRESHOLD = 5.0` — empirical, tune against webcam + lighting.
- `POST_ACTION_DELAY = 2.0` — reaction observation window.
- `IDLE_SAFETY_TIMEOUT = 30.0` — robustness backstop.

### 11.4 Four-thread Architecture
```
Main loop             FrameBuffer thread       Idle thread overlay     VLM thread
───────────           ────────────────────     ────────────────────    ──────────
sensor sync            capture @ 10 fps         IdleAnimator.tick()     wait for kick
  ↓                    2 s rolling buffer       (called from main loop) ↓
trigger check ──trigger─→ sample_recent(5) ───────────────────────────→ OpenAI API
  ↓                                                                       ↓ (2-6 s)
await VLM result                                                        return code
  ↓ (main loop keeps stepping Webots, never freezes)                      ↓
exec VLM code → mark_action_done                          ←──── code ready
```

(Note: idle is implemented as a per-step overlay called from the main loop, not a separate thread, because Webots motor writes must stay single-threaded.)

### 11.5 Academic narrative
For the report / defense:
> The system performs zero-shot temporal reasoning via multi-frame VLM prompting, extracting motion dynamics (oscillatory / approaching / retreating) without any explicit motion classifier or fine-tuning. It maintains a reactive feedback loop — after executing each generative behavior, it actively samples a post-action reaction window to adapt subsequent responses, rather than polling on a fixed schedule. Combined with continuous parameterization of motion primitives (`oscillate_joint(frequency, amplitude, decay)` chosen by the VLM), this exceeds the rule-based animation baseline in both efficiency and adaptability.

This responds to both of the professor's critiques:
- "Rule-based animation" → multi-frame reasoning + generative primitive code is not rule-based
- "Simple intent triggers" → state-aware trigger + temporal dynamics extraction is not simple intent classification

---

## 12. Implementation Phases (overview)

### Phase 0: Module integration + visual input — ✅ done
1. Merged `vlm.py` and `sandbox.py` logic into the Webots controller
2. `FrameBuffer` background thread (with motion scoring)
3. `INPUT_MODE` + `WEBCAM_SOURCE` switches (see §10.2)
4. Multi-image VLM call
5. `scripts/local_camera_server.py` (Flask MJPEG server)

### Phase 1: Motion primitives — ✅ done
`move_joint`, `move_joints`, `move_arm_ik`, `oscillate_joint`, `hold` plus `min_jerk` quintic trajectory shape.

### Phase 2: Idle + multi-thread orchestration — ✅ done
`idle(duration)` primitive, `IdleAnimator` overlay, `VLMTrigger` state machine, daemon-threaded VLM worker.

### Phase 3 + 4: Prompt redesign + sandbox + fallback + AST validation — ✅ done
See §15 for the original plan and §16 for the as-shipped retrospective. Notably:
- Joint-name typo tolerance (difflib fuzzy match).
- Three-tier fallback (Tier A retry, Tier B replay, Tier C idle).
- AST-based pre-flight validator with joint-limit, IK-radius, oscillation-parameter, and lower-body-forbidden checks.
- Local VLM backend behind `VLM_BACKEND=local` switch (cloud is default).

### Phase 5: Evaluation framework — **active forward plan, see §17**

### Phase 6: Stretch goals (deferred)
Action preemption mechanism, MediaPipe keypoint overlay, naturalness-aware filtering refinements, possible local-VLM judge.

---

## 13. Critical Files (current layout)

| File | Role |
|---|---|
| `nao_VLM/controllers/nao_vlm_test/nao_vlm_test.py` | Main Webots controller. NaoVlmAPI + main loop + threading. |
| `nao_VLM/controllers/nao_vlm_test/config.py` | `INPUT_MODE`, `WEBCAM_SOURCE`, trigger thresholds, URDF path resolution, VLM backend selection. |
| `nao_VLM/controllers/nao_vlm_test/frame_buffer.py` | Rolling buffer + motion score + base64 sampling. |
| `nao_VLM/controllers/nao_vlm_test/vlm_client.py` | Multi-image GPT-4o (or local) client + system prompt + parser. |
| `nao_VLM/controllers/nao_vlm_test/sandbox_exec.py` | AST validator + restricted exec sandbox. |
| `nao_VLM/controllers/nao_vlm_test/fallback.py` | 3-tier FallbackPolicy with action history and fire counters. |
| `nao_VLM/controllers/nao_vlm_test/idle_animator.py` | Per-step idle overlay. |
| `nao_VLM/controllers/nao_vlm_test/vlm_trigger.py` | `VLMTrigger` state machine. |
| `nao_VLM/controllers/nao_vlm_test/runtime.ini.example` | Per-developer Python interpreter path template. Active `runtime.ini` is gitignored. |
| `scripts/local_camera_server.py` | Laptop-side Flask MJPEG server for cloud streaming. |
| `requirements.txt` | Cloud / Linux deps: cv2, numpy, pin (Pinocchio), openai, python-dotenv, requests, flask. |
| `videos/` | Pre-recorded scenario clips (committed; user-recorded for canonical 8). |
| `debug_video_samples/` | 10 short pilot gesture clips for diversity testing. |

---

## 14. Verification (current and Phase 5)

### Phase 0–4 end-to-end smoke (already passes)
1. Set `WEBCAM_SOURCE` to a video file.
2. Launch Webots `nao_VLM/worlds/nao_VLM.wbt`.
3. Confirm console shows: `[init] VLMClient ready`, `sandbox exposes` lists 8 primitives, `VLMTrigger` initialised, main loop entered.
4. Watch for `[step ...] VLM kick` lines firing on motion / post_action / safety.
5. Confirm NAO visibly moves — primitives execute, idle overlay during inference.
6. Confirm `[VLM]   robot: ...` log lines describe sensible response intents matching the executed code.

### Phase 5 automated evaluation (target)
- `python -m evaluation.run_benchmark --scenarios all --rounds 3 --method cap`
- `python -m evaluation.run_benchmark --scenarios all --rounds 3 --method rule_baseline`
- Metrics output: Execution Success Rate, Safety Adherence, Joint Jerk, CoM stability, VLM-as-Judge score, Fallback Activation Rate.

### Smoke-test metric targets (initial)
- Execution Success Rate > 80%
- Safety Adherence (joint limits) == 100%
- No CoM excursion outside support polygon during upper-body motion
- Fallback activation < 20% per scenario

---

## 15. Phase 3 + 4 Original Plan (historical reference)

> This section was the original Phase 3 + 4 plan. The actual implementation is reflected in §16. Kept here so the design intent stays visible.

### 15.A Prompt diversification (Phase 3)
Replace the single waving example with **3-4 diverse examples** spanning different intents and primitive combinations:

- Decaying wave back at a greeting (`trajectory='cubic'`).
- Cautious lean-back on hostile / approaching gesture (upper-body shoulder pitch composition; lower-body joints already forbidden).
- Idle acknowledgment when human is static (`move_head` + `speak`).
- Bowing greeting on slow approach (`move_joints` with `min_jerk`).

Each annotated with one-line "use when motion_dynamics ∈ {…}" guidance.

### 15.B Joint-name typo tolerance (Phase 3)
In sandbox-exposed primitives, if `name` is not in `self.motors`, try `difflib.get_close_matches(name, self.motors.keys(), n=1, cutoff=0.7)` and either auto-correct (with a warning) or reject with a clear error that fallback can catch.

### 15.C Three-tier fallback policy (Phase 4)
A `FallbackPolicy` class:

- **Tier A** retry once on transient failure (network/timeout/parse_incomplete/AST-validation failure). Re-kicks VLM with the same frames.
- **Tier B** replay last successful action when retry exhausted. Cache size = last 3 successful responses.
- **Tier C** idle when no successful response is cached. Run `vlm_api.idle(2.0)`.

### 15.D Sandbox joint-limit pre-check (Phase 4)
AST static validation in `SandboxExecutor`:

- Parse `code_str` with `ast.parse`. Walk all `Call` nodes whose function name is one of the registered primitives.
- For `move_joint`, `move_joints`, `oscillate_joint`: extract literal `name`, check it is in the known joint list. Extract literal `angle` / `center` / `amplitude` and verify within joint limits.
- For `move_arm_ik`: extract literal `xyz` and check norm is reasonable.
- Returns a `ValidationResult(ok, reasons)`. If invalid, sandbox refuses to exec and triggers fallback Tier A.

### 15.E (Deferred) Pinocchio self-collision check
Not in scope; revisit only if evaluation reveals self-collisions.

---

## 16. Phase 3 + 4 + Cleanup Iteration — Retrospective (DONE)

This section documents what is now on `main`. **All items below are merged.** The next teammate's job is Phase 5 (§17) — they should not need to revisit anything in this section unless something here looks broken.

### 16.1 Status of §15 line-items

| §15 item | Final status | Where it lives |
|---|---|---|
| §15.A Prompt diversification (3-4 examples) | ✅ Complete (added during cleanup iteration; was initially partial) | `vlm_client.py` `_BASE_SYSTEM_PROMPT` "Concrete examples" block |
| §15.B Joint-name typo tolerance | ✅ Complete | `nao_vlm_test.py:_canonicalize_joint_name()` (difflib fuzzy match), `_normalize_arm_side()` (alias map) |
| §15.C Tier A retry | ✅ Complete | `fallback.py` `FallbackPolicy.handle_failure()` returns `'retry'` with budget; main loop calls `worker.kick_with_frames(last_trigger_frames)` |
| §15.C Tier B replay | ✅ Complete (added during cleanup iteration) | `fallback.py` action-history `deque(maxlen=3)` + `_tier_b_used_this_cycle` one-shot guard; main loop runs cached `python_code` through executor |
| §15.C Tier C idle | ✅ Complete | `handle_failure()` calls `idle_fn(2.0)` then returns `action='idle'` |
| §15.D AST validator | ✅ Complete (and exceeds plan, see §16.2) | `sandbox_exec.py:validate()` |

### 16.2 AST validator features (more than §15.D required)

`SandboxExecutor.validate()` enforces all of the following at AST parse time, before any motor write:

- Unknown primitives → `unknown_primitive: <name>`.
- Joint-name lookup → unknown joints rejected; lower-body joints (`L/R + Hip/Knee/Ankle*`) blocked as `lower_body_joint_forbidden` invariant.
- `navigate_to(...)` blocked as `navigate_to_forbidden_for_demo`.
- `move_joint`/`move_joints` keyword whitelisting; angle scalar type check; angle in `[joint_lo, joint_hi]`.
- `oscillate_joint`: amplitude ≤ 0.7 rad, frequency ≤ 2.5 Hz, duration ≤ 3.0 s, oscillation envelope must lie within joint limits, max 3 oscillation calls per code block.
- `move_arm_ik`: `side ∈ {left, right}`; xyz radius in `[0.05, 0.6]` m from torso origin.
- `set_hand`: openness in `[0.0, 1.0]`.
- `move_head`: rejects literal-zero no-op (`move_head_no_op_zero_angles`) when both `|yaw|` and `|pitch|` < 0.05 rad.
- Universal: duration ∈ `[0.05, 3.0]` s.

### 16.3 Cleanup iteration completed before Phase 5 handover

After auditing Phase 3+4, a small cleanup iteration closed the remaining §15 gaps and locked in invariants Phase 5 evaluation depends on. **All items merged.**

| # | Change | Files | Notes |
|---|---|---|---|
| 1 | Removed legacy primitives from sandbox registration. Active set is **`move_joint`, `move_joints`, `move_arm_ik`, `move_head`, `set_hand`, `oscillate_joint`, `hold`, `idle`** (8 primitives). | `nao_vlm_test.py` | `look_at`/`move_arm`/`operate_gripper`/`set_posture` no longer reachable by the VLM. `set_posture` was the last bypass route to lower-body joints; removing it locks the forbidden-lower-body invariant. |
| 2 | Documented `move_head(yaw, pitch, duration, trajectory)` in the prompt's Motion Grammar section AND in the local-VLM repair / refinement template allowlists. | `vlm_client.py` | Was registered in sandbox but invisible to VLM until now. |
| 3 | Added 3 concrete worked examples to `_BASE_SYSTEM_PROMPT` covering the main `motion_dynamics` cases: oscillatory greeting (decaying wave), static curious (head-tilt), approaching cautious (upper-body lean + open palm). | `vlm_client.py` | Each annotated with the `motion_dynamics` it targets. Closes §15.A. |
| 4 | Implemented Tier B replay in `FallbackPolicy`: action-history `deque(maxlen=3)`, one-shot `_tier_b_used_this_cycle` guard, fire counters (`n_tier_a/b/c`), `stats()` dict for evaluation. Main loop wires `'replay'` branches for both call-failure and exec-failure paths. | `fallback.py` + `nao_vlm_test.py` | Empty/whitespace code never cached; `record_success(rsp)` resets full cycle state. Closes §15.C Tier B. |
| 5 | Added `move_head_no_op_zero_angles` validator rule to reject literal head no-ops where both `|yaw|` and `|pitch|` < 0.05 rad. | `sandbox_exec.py` | Caught a real failure mode where VLM was settling on `move_head(0.0, 0.0)` for static clips, parroting Example 2 structure but zeroing values. |
| 6 | Added `robot_intent` field to the JSON output schema. Log lines now show both human and robot intents on their own lines for readability. | `vlm_client.py` + `nao_vlm_test.py` | Phase 5 judge can cross-check stated intent vs executed code. |
| 7 | runtime.ini personalization: shipped `runtime.ini.example` as template; the active `runtime.ini` is now in `.gitignore` so each developer's Python interpreter path stays local. | `runtime.ini.example`, `.gitignore` | Each dev runs `cp runtime.ini.example runtime.ini` and edits `COMMAND`. |
| 8 | Added `requests>=2.28.0` to `requirements.txt`. The local-VLM HTTP backend code in `vlm_client.py` imports it at module top; without it, VLMClient fails to load even on cloud-only runs. | `requirements.txt` | |

### 16.4 Empirical observations from cleanup smoke testing

Running the controller with `debug_video_samples/` clips (`clap`, `thumbs_up`, etc.) under `WEBCAM_SOURCE`:

- `move_head` is now actively used by the VLM for static / acknowledgment-style responses; previously invisible to the model.
- `motion_dynamics='oscillatory'` clips like `clap`: VLM occasionally produces a genuine bilateral response (`move_arm_ik(left)` + `move_arm_ik(right)` + bilateral `oscillate_joint`), which is novel — not present in any prompt example.
- The same `oscillatory` clip can drift toward the Example 1 single-arm wave shape on subsequent post-action samples. Convergence-to-canonical-example is real but partial; Phase 5 should measure it (diversity metric).
- `move_head_no_op_zero_angles` rejection has been observed catching genuine VLM mistakes (zeroed head angles even when the clip was non-static).
- Anti-PII clause holds: zero GPT-4o content-policy refusals across cleanup-test runs.

### 16.5 Decisions still in effect for Phase 5

- **VLM backend for evaluation = cloud-only by default.** Set `VLM_BACKEND=openai` explicitly in `.env`. `VLM_BACKEND=auto` is a footgun on small VMs (falls to `local` and downloads multi-GB model when API key is missing).
- **Active primitive set** (sandbox-registered, prompt-documented): `move_joint`, `move_joints`, `move_arm_ik`, `move_head`, `set_hand`, `oscillate_joint`, `hold`, `idle`. Locomotion + lower-body are forbidden by the validator.
- **Scenario videos for evaluation = phone-recorded by user.** 5-10 s clips per §10.5, named `videos/scenario_NN_label.mp4`.
- **Pre-Phase 5 cleanup is done.** The next iteration is Phase 5 itself.

### 16.6 Bonus features the teammate added beyond §15 (now part of the system)

- `RUN_MODE='oneshot'` — sample once, kick VLM once, execute, exit. **Phase 5 evaluation entry point.**
- `RUN_MODE='replay'` — execute precomputed Python from file. Useful for re-running cached sessions deterministically.
- Local VLM backend (Qwen2-VL / SmolVLM2 with 4-bit loading) behind `VLM_BACKEND=local`. Phase 6 territory.
- `worker.kick_with_frames(frames)` for retrying with the same evidence (used by Tier A).
- Webcam backend selection + resolution config in `frame_buffer.py`.
- Naturalness-aware candidate filtering for local-VLM path (multiple samples + pick best).

---

## 17. Phase 5 Implementation Plan — Forward, Active

**Audience:** the teammate picking this up after the cleanup iteration. Everything in §16 is already on `main`; Phase 5 builds on top without modifying the existing modules' public APIs.

### 17.1 Goal
Reproducible evaluation pipeline that for each of 8 canonical scenarios:
1. Drives the controller with a fixed input video,
2. Logs joint trajectories + CoM + sandbox events,
3. Computes Jerk + CoM-stability + Execution-success metrics,
4. Asks an independent VLM-as-Judge whether the response was appropriate,
5. Compares against a rule-based baseline.

The harness wraps around the existing `RUN_MODE='oneshot'` controller — **no new controller modes need to be added.**

### 17.2 Modules to add

| Path | Role |
|---|---|
| `evaluation/__init__.py` | Empty package marker |
| `evaluation/scenarios.py` | Registry: `SCENARIOS = {id: ScenarioSpec(video_path, expected_intent, expected_motion_dynamics, ground_truth_response_label)}` |
| `evaluation/run_benchmark.py` | CLI: `python -m evaluation.run_benchmark --scenarios all --rounds 3 --method cap`. Iterates scenarios × rounds × method, launches Webots in oneshot mode for each, collects per-run JSON results into `artifacts/eval/<run_id>/<scenario>__<round>.json`. |
| `evaluation/metrics.py` | `compute_jerk(joint_log)`, `com_excursion(com_log, support_polygon)`, `success_rate(events)` |
| `evaluation/judge.py` | `judge_appropriateness(input_frames, robot_response_screenshot, intent) → (pass: bool, rationale: str)` using a separate GPT-4o call |
| `evaluation/rule_baseline.py` | Replacement for `vlm_client.VLMClient` when `--method rule_baseline`: same input pipeline, but VLM output is restricted to a single `intent` label, and a hardcoded `INTENT_TO_CODE` dict produces the response |
| `nao_VLM/controllers/nao_vlm_test/metrics_recorder.py` | Lightweight per-step JSONL logger. Append-only writer that the main loop calls each step when an env var (`METRICS_RUN_ID`) is set |
| `videos/scenario_01_wave.mp4` … `scenario_08_idle.mp4` | 8 phone-recorded canonical clips per §10.5. **User-provided.** |

### 17.3 Hooks needed in existing code (small additions only)

1. **Main loop** — when `os.getenv('METRICS_RUN_ID')` is set, every step append `{t, q, com_xyz}` to `artifacts/oneshot/<run_id>/joint_states.jsonl`. CoM via `pin.centerOfMass(model, data, q_current)`. ~5 LoC.
2. **Sandbox** — when `METRICS_RUN_ID` is set, append `{t, event, error}` (`event ∈ {validate_pass, validate_fail, exec_ok, exec_error}`) to `artifacts/oneshot/<run_id>/sandbox_events.jsonl`. ~5 LoC in `SandboxExecutor.run()`.
3. **FallbackPolicy** — already exposes `n_tier_a/b/c` via `stats()` from cleanup §16.3. Have the main loop dump `fallback.stats()` into the per-run `result.json` on exit. ~3 LoC.
4. **Controller exit (oneshot)** — write a single `result.json` with: `scenario_id`, `vlm_response` (`semantic_context` + `python_code` + `robot_intent`), `exec_outcome`, `fallback_stats`, paths to JSONL files. ~10 LoC.

**No public-API changes** to: `vlm_client.py`, `frame_buffer.py`, `idle_animator.py`, `vlm_trigger.py`, `fallback.py`.

### 17.4 Rule-based baseline design

Mirrors the CaP system's pipeline so comparisons are fair:

- Same FrameBuffer input.
- Same prompt to GPT-4o, but the prompt asks for **only an `intent` label** (no Python code). Fixed allowed labels: `{wave, reject, lean, walk_away, crouch, greet, idle}`.
- `INTENT_TO_CODE` dict in `rule_baseline.py` maps each label to a hardcoded primitive composition (e.g. `wave → "move_arm_ik('right', xyz=[0.15,-0.15,0.10],duration=0.4); oscillate_joint(...); move_arm_ik('right', xyz=[0.02,-0.10,-0.20],duration=0.4)"`).
- Sandbox path identical → safety, validator, metrics behave identically.

Rule-baseline benefits: (a) demonstrates the proposal's "intent classification → fixed behavior" point, (b) deterministic and fast (no second-pass VLM code generation), (c) good lower bound for comparison.

### 17.5 VLM-as-Judge design

Per proposal §3 evaluation framework:

- **Inputs to judge:** (a) first frame from input video (the human gesture), (b) screenshot of NAO at end of execution (the robot response), (c) one-line `robot_intent` text from `semantic_context`.
- **Judge prompt** (fixed): "Is this physical response socially appropriate for the human's gesture? Reply JSON with `{pass: bool, rationale: str (≤30 words)}`."
- **Model:** GPT-4o, temperature=0.0, two images per call.
- **Cache:** key by `(input_video_id, response_hash)`. Cached judgements during reruns are free.

Judge call cost: ~$0.01 per judgement at low-detail. 8 scenarios × 3 rounds × 2 methods = 48 judgements ≈ $0.50 per full benchmark.

### 17.6 Canonical scenarios

```python
SCENARIOS = {
    'wave':            ScenarioSpec(video='videos/scenario_01_wave.mp4',          intent='greeting',    expected_motion='oscillatory'),
    'cross_arms':      ScenarioSpec(video='videos/scenario_02_cross_arms.mp4',    intent='rejection',   expected_motion='static'),
    'lean_forward':    ScenarioSpec(video='videos/scenario_03_lean_forward.mp4',  intent='approach',    expected_motion='approaching'),
    'walk_away':       ScenarioSpec(video='videos/scenario_04_walk_away.mp4',     intent='leaving',     expected_motion='retreating'),
    'crouch':          ScenarioSpec(video='videos/scenario_05_crouch.mp4',        intent='lower_focus', expected_motion='lowering'),
    'reject_gesture':  ScenarioSpec(video='videos/scenario_06_reject.mp4',        intent='rejection',   expected_motion='static'),
    'greet_handshake': ScenarioSpec(video='videos/scenario_07_handshake.mp4',     intent='greeting',    expected_motion='raising'),
    'idle_standing':   ScenarioSpec(video='videos/scenario_08_idle.mp4',          intent='neutral',     expected_motion='static'),
}
```

> **Pilot dataset alternative.** Until the canonical 8 are recorded, the existing `debug_video_samples/` clips (`waving`, `pointing`, `beckon`, `finger_no`, `stop`, `thumbs_up`, `yes_nod`, `no_shake`, `clap`, `shrug`) can act as a 10-clip pilot benchmark. They cover ~5 of the 8 canonical intents and add 5 extras useful for diversity testing.

### 17.7 Metric targets

- Execution Success Rate > 80% (% scenarios where `exec_ok` and not all-Tier-C-idle).
- Safety Adherence == 100% (validator already enforces; sanity-check no validator bypass).
- CoM excursion outside support polygon == 0 during upper-body motion.
- Fallback Activation Rate < 20% per scenario.
- VLM-as-Judge Appropriateness ≥ 75% (CaP). Compare CaP vs rule_baseline: CaP should beat baseline by ≥ 10 percentage points; if not, the generative system isn't doing more than the lookup.

### 17.8 Verification commands

```bash
# Smoke single scenario in oneshot mode (sanity check the metrics hooks)
RUN_MODE=oneshot METRICS_RUN_ID=smoke_$(date +%s) \
    WEBCAM_SOURCE=videos/scenario_01_wave.mp4 \
    webots --no-rendering nao_VLM/worlds/nao_VLM.wbt
# expect: artifacts/oneshot/smoke_*/result.json + joint_states.jsonl + sandbox_events.jsonl

# Full benchmark (CaP)
python -m evaluation.run_benchmark --scenarios all --rounds 3 --method cap \
    --output artifacts/eval/cap_$(date +%Y%m%d_%H%M).json

# Full benchmark (rule baseline)
python -m evaluation.run_benchmark --scenarios all --rounds 3 --method rule_baseline \
    --output artifacts/eval/baseline_$(date +%Y%m%d_%H%M).json

# Judge + final report
python -m evaluation.judge artifacts/eval/cap_*.json artifacts/eval/baseline_*.json \
    > artifacts/eval/report.md
```

Per-scenario success criteria:
1. `RUN_MODE=oneshot` exits cleanly within 15 s of starting.
2. `result.json` valid; `joint_states.jsonl` has > 50 records.
3. Jerk computation finishes without numerical issues.
4. Judge call returns parseable JSON with `pass` and `rationale`.
5. Aggregate CaP success rate > rule-baseline by ≥ 10 percentage points.

### 17.9 Suggested implementation order

1. **Metrics recorder + JSONL hooks** in main loop and sandbox. Smoke-test on a single scenario by running oneshot mode and verifying the JSONL files are written.
2. **`metrics.py`** — pure offline computations on JSONL files. Unit-testable without Webots.
3. **`scenarios.py`** registry. Use `debug_video_samples/` as the pilot dataset until the canonical 8 are recorded.
4. **`run_benchmark.py`** harness — subprocess-launches Webots in oneshot mode per scenario, parses `result.json` outputs, aggregates a summary CSV/JSON.
5. **`rule_baseline.py`** — implement `INTENT_TO_CODE` and the intent-only prompt, plug into the harness as `--method rule_baseline`.
6. **`judge.py`** — separate GPT-4o judging pass over the aggregated outputs, with caching.
7. **Canonical 8 videos** — record/import once available; pilot benchmark works on `debug_video_samples/` in the meantime.

---

## 18. Out of scope (deferred or future work)

- **Pinocchio self-collision check** (still deferred from §15.E). Revisit only if Phase 5 evaluation reveals self-collisions.
- **Phase 6 stretches**: action preemption, MediaPipe keypoint overlay, naturalness-aware filtering refinement (already partially implemented for local-VLM path), local-VLM judge.
- **Live-webcam Phase 5 runs.** Live webcam is only used for demos; evaluation stays on pre-recorded videos for reproducibility.
- **ROS 2 `ros2_control` migration.** Document Webots Controller's role as the equivalent in the report; ROS 2 stays as future work.
