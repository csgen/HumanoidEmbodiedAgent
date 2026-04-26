# Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan

## Context

This project is a course group assignment for building a humanoid embodied agent. The original proposal (v2.1) planned a custom primitive-based URDF (9-DoF upper body + wheeled base) in Gazebo + ROS 2. However, the teammate responsible for URDF imported an open-source NAO H25 V5.0 model from the `ros-naoqi/nao_robot` package and runs it in Webots instead. This plan analyzes the gap, responds to the professor's feedback ("appears rule-based rather than true embodied intelligence"), and lays out a revised implementation path.

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

## 2. Already Implemented

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

### 3.2 VLM Prompt ↔ Actual API Mismatch
- `vlm.py` prompt exposes: `set_torso_pitch()`, `blend_arm_pose()`, `play_idle_animation()`
- `nao_vlm_test.py` actually implements: `look_at()`, `move_arm()`, `set_posture()`, `operate_gripper()`, `navigate_to()`
- **The two API sets do not overlap**. VLM-generated code cannot execute on Webots
- `sandbox.py` whitelist functions are mock prints, not wired to the controller

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
- Biggest technical debt: **ROS 2 architecture absence** and **VLM prompt ↔ actual API disconnect**

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
oscillate_joint(name, center, amplitude, frequency, duration, decay=0.0)
hold(duration)
```

**System-level (not part of VLM's everyday toolkit):**
```python
idle(duration)                      # finite-duration primitive composition
start_idle_loop() / stop_idle_loop()   # background latency masking
speak(text)
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

### 8.1 Critical: Three modules are disconnected

**Current state**:
- `vlm.py` uses `cv2.VideoCapture(0)` → **laptop webcam, not NAO's eye**
- `sandbox.py` is a standalone script with mocked functions
- `nao_vlm_test.py` is the Webots controller with the real robot/camera/motors
- **No data flow between the three**

**Prerequisite work**: merge VLM calling and sandbox execution into the Webots controller, share the same `robot` object, replace `cv2.VideoCapture(0)` with either NAO's `CameraTop` or a webcam adapter (see §11).

### 8.2 Medium difficulties

**A. Integrating blocking primitives with the Webots main loop**
- Current main loop is outer-driven: `while robot.step(): if step_count==200: ...` with non-blocking API calls
- Primitives must internally drive `robot.step()` (model: the existing `navigate_to()`)
- Main loop refactor required: **only handle sensor sync + idle heartbeat + command dispatch**

**B. VLM HTTP call freezes simulation → latency masking**
- OpenAI API blocks 2-6 s; direct call from the main loop freezes physics
- Solution: `threading.Thread` + `queue.Queue`, background VLM call, main loop plays idle
- Solving idle == solving latency masking (two-for-one)

**C. Unpredictable action duration + interruption**
- A VLM code block may take 5-10 s; a new command during execution — preempt or defer?
- MVP: no preemption, serial queue, check between segments
- Advanced: `threading.Event` flag for preemptable primitives

**D. Pinocchio state sync timing**
- Main loop currently syncs sensors → `q_pin` every step; if primitives internally call `robot.step()`, this sync is bypassed
- Extract `_sync_sensors()` helper and call it inside every primitive's step loop

### 8.3 Minor difficulties

| Item | Effort | Note |
|---|---|---|
| E. Add `duration` dim to `move_arm` | Small | Reuse Pinocchio IK kernel |
| F. Inject joint limits into prompt | Small | Scan `motor.getMinPosition()/getMaxPosition()` |
| G. Inject `NaoVlmAPI` bound methods into sandbox | Small | `safe_globals['move_joint'] = vlm_api.move_joint` |
| H. Joint-name typo tolerance | Small | `difflib.get_close_matches` or strict validation |
| I. `trajectory='min_jerk'` (quintic polynomial) | Small-Medium | Closed-form solution, MVP can do cubic first |

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

1. **NAO coordinate frame**: chest-centered, y-negative is right side, meters. `move_arm` test values `(0.05, -0.05, -0.05)` are small because units are meters. Prompt must tell VLM this, or IK won't converge
2. **VLM latency 2-6 s**: mandatory async + idle fallback; otherwise physics simulation stalls
3. **Joint name CamelCase**: NAO uses `RShoulderPitch`, VLM tends to emit snake_case
4. **Pinocchio FreeFlyer base (first 6 DoFs)**: keep the `J[:, :6] = 0.0` trick in every IK primitive to avoid the floating base absorbing the IK delta

---

## 9. Idle Design

### 9.1 Current state
- `sandbox.py`'s `api_play_idle_animation()` is just `print(...)`, no actual motion
- `vlm.py` prompt mentions `'idle'` as a pose name — string only
- **NaoVlmAPI has no idle method**
- Must be built from scratch

### 9.2 Two usage modes

**A. As a VLM-callable primitive (finite duration)**
```python
def idle(self, duration):
    """Subtle breathing + small head scanning + gentle shoulder sway"""
    steps = int(duration * 1000 / self.timestep)
    for i in range(steps):
        t = i * self.timestep / 1000.0
        self.motors['HeadYaw'].setPosition(0.1 * math.sin(0.3 * t))
        self.motors['LShoulderPitch'].setPosition(1.5 + 0.03 * math.sin(0.5 * t))
        self.motors['RShoulderPitch'].setPosition(1.5 + 0.03 * math.sin(0.5 * t + math.pi))
        self.robot.step(self.timestep)
```

**B. As background latency masking (continuous)**
- `threading.Thread` looping `idle(2.0)`
- `threading.Event` for pause/resume
- VLM inference starts → `start_idle`; VLM code arrives → `stop_idle` → execute → `start_idle`

### 9.3 Idle as fallback trigger (three-tier per proposal)
- Tier A (VLM timeout / parse failure) → keep playing idle + retry once
- Tier B (sandbox execution exception) → return to last successful action → idle
- Tier C (no history) → direct idle

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

### 10.6 Cloud streaming setup (live-webcam demo from Windows laptop to cloud)

#### A. Local MJPEG server (runs on Windows laptop)
```python
# scripts/local_camera_server.py
from flask import Flask, Response
import cv2, argparse

app = Flask(__name__)
cap = None

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret: continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='0')  # 0 or a video file path
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    app.run(host='127.0.0.1', port=args.port)  # localhost only, no public exposure
```

#### B. SSH reverse tunnel (while dev session active)
```bash
# On the Windows laptop:
ssh -R 5000:localhost:5000 user@cloud-server
```
Maps laptop's port 5000 → cloud's `localhost:5000`. All traffic through SSH, zero public exposure.

#### C. Cloud-side controller
```python
WEBCAM_SOURCE = "http://localhost:5000/video_feed"
buffer = FrameBuffer(source=WEBCAM_SOURCE, fps=10, buffer_seconds=2.0)
```

OpenCV's `VideoCapture` handles MJPEG HTTP streams natively — no protocol-specific code in our side.

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
```python
class FrameBuffer:
    def __init__(self, buffer_seconds=2.0, fps=10, source=0):
        self.buffer = collections.deque(maxlen=int(buffer_seconds * fps))
        self.fps = fps
        self.source = source
        self.last_motion_score = 0.0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
    
    def _capture_loop(self):
        cap = cv2.VideoCapture(self.source)
        prev_gray = None
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret: continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                self.last_motion_score = float(np.mean(diff))
            prev_gray = gray
            self.buffer.append((time.time(), frame))
            time.sleep(1.0 / self.fps)
    
    def sample_recent(self, n=5):
        """Uniformly sample n frames from the last `buffer_seconds` seconds"""
        if len(self.buffer) < n: return []
        indices = np.linspace(0, len(self.buffer)-1, n).astype(int)
        frames = [self.buffer[i][1] for i in indices]
        return [self._encode_base64(f) for f in frames]
    
    def _encode_base64(self, frame):
        _, buf = cv2.imencode('.jpg', frame)
        return base64.b64encode(buf).decode('utf-8')
```

Advantages:
- VLM can pull historical frames instantly — no 1.5 s sampling pause
- Background thread composes cleanly with idle and VLM threads
- Motion score comes for free, feeds the trigger

#### B. GPT-4o multi-image call
```python
def call_vlm_with_sequence(frames_b64, scenario_context=None):
    content = [{"type": "text", "text": user_prompt}]
    for img_b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}",
                "detail": "low"  # ~85 tokens per image, 5 images = ~425 tokens
            }
        })
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt_with_temporal},
            {"role": "user", "content": content}
        ],
        max_tokens=500,
        temperature=0.2
    )
```

#### C. Prompt — temporal-aware
Add to system prompt:
```
You are receiving a SEQUENCE of 5 frames captured over 1.5 seconds (chronological order).
Analyze the MOTION and INTENT across frames, not just the final pose:
- Arm trajectory direction (rising / lowering / oscillating)
- Body movement (approaching / retreating / static)
- Action phase (starting / ongoing / ending)

Your semantic context object must include:
{
    "intent": ...,
    "social_distance": ...,
    "affect": ...,
    "confidence": ...,
    "motion_dynamics": "oscillatory" | "approaching" | "retreating" | "raising" | "lowering" | "static"  # NEW
}
```

The new `motion_dynamics` field directly supports "continuous parameterization" — e.g., VLM classifies the input as `oscillatory` and chooses frequency/amplitude for `oscillate_joint()` accordingly.

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
```python
class VLMTrigger:
    STATE_IDLE = 'idle'
    STATE_EXECUTING = 'executing'
    STATE_POST_ACTION = 'post_action'
    
    def __init__(self, buffer,
                 motion_threshold=5.0,
                 post_action_delay=2.0,
                 idle_safety_timeout=30.0):
        self.buffer = buffer
        self.motion_threshold = motion_threshold
        self.post_action_delay = post_action_delay
        self.idle_safety_timeout = idle_safety_timeout
        self.state = self.STATE_IDLE
        self.state_enter_time = time.time()
    
    def mark_executing(self):
        self.state = self.STATE_EXECUTING
        self.state_enter_time = time.time()
    
    def mark_action_done(self):
        self.state = self.STATE_POST_ACTION
        self.state_enter_time = time.time()
    
    def mark_idle(self):
        self.state = self.STATE_IDLE
        self.state_enter_time = time.time()
    
    def should_trigger(self):
        now = time.time()
        elapsed = now - self.state_enter_time
        
        if self.state == self.STATE_EXECUTING:
            return False
        
        if self.state == self.STATE_POST_ACTION:
            if elapsed >= self.post_action_delay:
                self.mark_idle()   # consume the post-action event
                return True
            return False
        
        # STATE_IDLE
        if self.buffer.last_motion_score > self.motion_threshold:
            self.mark_idle()      # reset timing, avoid oscillation
            return True
        
        if elapsed > self.idle_safety_timeout:
            self.mark_idle()
            return True
        
        return False
```

#### Example timeline of a typical interaction
```
t=0      IDLE         —                                robot idle, FrameBuffer recording
t=5.2    IDLE         human enters, motion > threshold  ✅ trigger VLM (motion)
t=5.2    EXECUTING    VLM inference + motion execution  silent
t=8.7    POST_ACTION  motion done                       start 2 s reaction window
t=10.7   POST_ACTION  2 s elapsed                       ✅ trigger VLM (observe reaction)
...
t=40     IDLE         no human for a while              ✅ safety timeout fires
t=40     EXECUTING    VLM outputs idle command
t=41     IDLE         returns to idle
```

#### Executor integration
```python
# Main loop pseudocode
while robot.step(timestep) != -1:
    if trigger.should_trigger():
        trigger.mark_executing()
        frames = buffer.sample_recent(5)
        vlm_code = call_vlm(frames)               # background thread
        exec_in_sandbox(vlm_code, vlm_api)        # blocking, internally calls robot.step()
        trigger.mark_action_done()                # opens 2s observation window
```

#### Cost comparison
| Scenario | Old "timed + motion" | State-aware | Reduction |
|---|---|---|---|
| 1 min idle, no human | 20 calls (every 3 s) | 2 calls (30 s safety × 2) | 90% |
| 1 min with one full interaction | ~15-20 calls | ~2-3 calls | ~83% |
| GPT-4o `detail=low` × 5 frames | $0.005/call | same | — |
| **Idle 1 hour** | ~$6 | ~$0.5 | **92%** |
| **Active 1 hour** | ~$6 | ~$1 | **83%** |

#### Tunable parameters
- `motion_threshold=5.0` — empirical, tune against webcam + lighting. Initial target: natural waves trigger, breathing does not
- `post_action_delay=2.0` — reaction window. Too short misses the response; too long feels sluggish
- `idle_safety_timeout=30.0` — robustness backstop. Can be lengthened considerably since motion trigger handles the main path

### 11.4 Four-thread Architecture
```
Main loop             FrameBuffer thread       Idle thread          VLM thread
───────────           ────────────────────     ────────────────     ──────────
sensor sync            capture @ 10 fps         idle motion          wait for trigger
  ↓                    2 s rolling buffer       head oscillation     ↓
trigger check ──trigger─→ sample_recent(5) ─────────────────────→  OpenAI API
  ↓                                                                   ↓ (2-6 s)
await VLM result                                                    return code
  ↓ (main loop keeps stepping Webots, never freezes)                  ↓
stop idle → exec VLM code → restart idle              ←──── code ready
```

All four threads serve different purposes; the main loop never blocks.

### 11.5 Academic narrative
For the report / defense:
> The system performs zero-shot temporal reasoning via multi-frame VLM prompting, extracting motion dynamics (oscillatory / approaching / retreating) without any explicit motion classifier or fine-tuning. It maintains a reactive feedback loop — after executing each generative behavior, it actively samples a post-action reaction window to adapt subsequent responses, rather than polling on a fixed schedule. Combined with continuous parameterization of motion primitives (`oscillate_joint(frequency, amplitude, decay)` chosen by the VLM), this exceeds the rule-based animation baseline in both efficiency and adaptability.

This responds to both of the professor's critiques:
- "Rule-based animation" → multi-frame reasoning + generative primitive code is not rule-based
- "Simple intent triggers" → state-aware trigger + temporal dynamics extraction is not simple intent classification

---

## 12. Implementation Phases

### Phase 0: Module integration + visual input (prerequisite)
1. Merge `vlm.py` and `sandbox.py` logic into the Webots controller
2. Implement `FrameBuffer` background thread (with motion scoring)
3. `INPUT_MODE` + `WEBCAM_SOURCE` switches (see §10.2)
4. Convert `generate_embodied_behavior()` to a multi-image call
5. Write `scripts/local_camera_server.py` (Flask MJPEG server) for cloud-streaming demos
6. Validate in three modes:
   - `WEBCAM_SOURCE=0` (local webcam, local dev)
   - `WEBCAM_SOURCE="videos/foo.mp4"` (file source, works on cloud without streaming)
   - `WEBCAM_SOURCE="http://localhost:5000/video_feed"` (cloud + SSH reverse tunnel, live demo)

### Phase 1: Motion primitives
1. `move_joint(name, angle, duration, trajectory='cubic')`
2. `move_joints(dict, duration)`
3. Convert current `move_arm` into `move_arm_ik(side, xyz, duration)`
4. `oscillate_joint(name, center, amplitude, frequency, duration, decay)`
5. `hold(duration)`

### Phase 2: Idle + multi-thread orchestration
1. `idle(duration)` primitive
2. `start/stop_idle_loop()` background thread
3. `VLMTrigger` state-aware trigger (IDLE / EXECUTING / POST_ACTION)
4. Move VLM call to the background thread scaffolded in Phase 0

### Phase 3: Prompt redesign + sandbox hookup
1. Scan joint limits from `motor.getMin/MaxPosition()` → inject into prompt
2. Temporal-aware prompt (multi-frame + `motion_dynamics` field)
3. Prompt additions: NAO coordinate frame, primitive API spec, example code
4. Sandbox whitelist → `NaoVlmAPI` bound methods
5. Joint-name typo tolerance (`difflib` or strict validation)

### Phase 4: Fallback + safety
1. Three-tier fallback implementation
2. Joint-limit pre-check in sandbox (before execution)
3. (Optional) Pinocchio self-collision check

### Phase 5: Evaluation framework (pre-recorded video based)
1. Record 8 canonical scenario clips
2. OBS configuration script (video → Virtual Camera)
3. Evaluation harness: iterate 8 scenarios × N rounds, log joint states + VLM outputs
4. Joint Jerk computation (3rd-order finite differences)
5. CoM tracking + support polygon check
6. VLM-as-Judge (compare input first/last frame + robot response screenshot)
7. Rule-based baseline counterpart

### Phase 6: Stretch goals
1. `trajectory='min_jerk'` (quintic polynomial)
2. Action preemption mechanism
3. Local VLM deployment (Qwen2-VL, MiniCPM-V) — native video support
4. MediaPipe keypoint overlay injected into prompt

---

## 13. Critical Files to Modify

| File | Modification |
|---|---|
| `nao_VLM/controllers/nao_vlm_test/nao_vlm_test.py` | Add `FrameBuffer`, `VLMTrigger`, motion primitives; refactor main loop |
| `vlm.py` | Merge into controller; switch to multi-image call; rewrite prompt |
| `sandbox.py` | Merge into controller; inject `NaoVlmAPI` bound methods; add joint-limit precheck |
| `nao_VLM/worlds/nao_VLM.wbt` | (Optional) add billboard or Display node for `camera_top` mode |
| New: `config.py` | `INPUT_MODE`, `WEBCAM_SOURCE`, motion thresholds, API keys |
| New: `scripts/local_camera_server.py` | Flask MJPEG server for cloud-streaming the laptop webcam |
| New: `evaluation/` | Harness, metrics, VLM-as-Judge, 8 scenario configs |
| New: `videos/` | Pre-recorded canonical scenario clips (sync'd to cloud for evaluation) |

---

## 14. Verification

### End-to-end smoke test
1. Start OBS with `videos/scenario_01_wave.mp4` on Virtual Camera
2. Launch Webots world `nao_VLM.wbt`
3. Confirm FrameBuffer populates (debug print of `len(buffer)`)
4. Confirm motion trigger fires (debug log `"trigger: motion score=X"`)
5. Confirm VLM returns multi-image analysis with `motion_dynamics` field
6. Confirm NAO executes motion primitives in Webots (visual check)
7. Confirm 2-second post-action trigger fires → second VLM call
8. Confirm idle thread runs during VLM inference (no frozen robot)

### Automated evaluation run
- `python evaluation/run_benchmark.py --scenarios all --rounds 3 --method cap`
- `python evaluation/run_benchmark.py --scenarios all --rounds 3 --method rule_baseline`
- Metrics output: Execution Success Rate, Safety Adherence, Joint Jerk, CoM stability, VLM-as-Judge score, Fallback Activation Rate

### Smoke-test metrics targets (initial)
- Execution Success Rate > 80%
- Safety Adherence (joint limits) == 100%
- No CoM excursion outside support polygon during upper-body motion
- Fallback activation < 20% per scenario

---

## 15. Next Iteration — Phase 3 Finishing + Phase 4

### Context
After Phase 2 was integrated and the trigger counter bug was fixed, a real run
on a 20s phone-recorded clip surfaced two follow-up issues:

1. **VLM parrots the single in-prompt example.** Kicks 4–8 produced near-identical
   waving code (same xyz, same elbow oscillation), regardless of how the human
   varied gestures. Root cause: the prompt only contains one example (the
   "decaying wave"), so GPT-4o anchors to it and never explores other primitive
   compositions. `min_jerk` is documented but never picked, for the same reason.
2. **No real fallback chain.** When the VLM call fails (`parse_incomplete` /
   refusal / network error), the controller just returns to IDLE and waits for
   the next motion event. Two consecutive GPT-4o content-policy refusals were
   already observed in an earlier run. Proposal §3.4 specifies a three-tier
   degradation policy that we have not yet implemented.

These are the two remaining items in plan Phase 3 (§12) and the entirety of
plan Phase 4 (§12). Tackling them now closes the gap before Phase 5 evaluation.

### Scope of this iteration

#### A. Prompt diversification (Phase 3 §3.3 — finishing)
Replace the single waving example with **3-4 diverse examples** spanning
different intents and primitive combinations, so the VLM has a richer
template library:

- **Example 1: Decaying wave back at a greeting.** Keep current pattern but
  use `trajectory='cubic'` explicitly and add an `affect`-based amplitude
  guideline.
- **Example 2: Cautious lean-back on hostile / approaching gesture.** Uses
  `move_joints({'LHipPitch': ..., 'RHipPitch': ..., 'LKneePitch': ...,
  'RKneePitch': ...}, duration=0.6, trajectory='min_jerk')` to bend slightly
  backward (recovers proposal's "lean-back rejection" idea via biped joints).
- **Example 3: Idle acknowledgment when human is static.** Uses
  `move_joint('HeadYaw', ...)` for a small head tilt + `speak(...)` —
  demonstrates that not every response must be an arm gesture.
- **Example 4: Bowing greeting on slow approach.** Uses `move_joints` with
  `trajectory='min_jerk'` to demonstrate when min_jerk is appropriate.

Each example is annotated with a one-line comment about WHEN it would be
chosen, training the VLM to differentiate by `motion_dynamics` and `affect`.

#### B. Joint-name typo tolerance (Phase 3 §3.5)
In the sandbox-exposed primitives (`move_joint`, `move_joints`,
`oscillate_joint`), if `name` is not in `self.motors`, attempt
`difflib.get_close_matches(name, self.motors.keys(), n=1, cutoff=0.7)`
and either auto-correct (printing a warning) or reject with a clear error
that fallback can catch. Recommended: **strict reject** (better signal for
fallback policy) + log the candidate match for debugging.

#### C. Three-tier fallback policy (Phase 4 §4.1)
A new module `fallback.py` with a `FallbackPolicy` class:

- **Tier A — Retry once on transient failure.** Triggered by:
  network/timeout exception, `parse_incomplete`, sandbox AST-validation
  failure (added in §15.D below). Re-kicks VLM with the SAME frames (so
  ablations can compare inputs fairly). Retry budget per state-cycle = 1.
- **Tier B — Replay last successful action.** When the VLM has produced
  good code recently and the new attempt fails after retry, replay that
  cached `python_code` instead of generating new. Cache size = last 3
  successful responses; pick the most recent unless its semantic context
  was very different from the current one (rough cosine similarity on the
  semantic dict, or just "most recent" for MVP).
- **Tier C — Idle.** If no successful response is cached, run
  `vlm_api.idle(2.0)` and stay in IDLE.

Wire-in: in the main loop, replace `trigger.mark_idle()` on `rsp.ok == False`
with `fallback.handle_failure(reason, last_frames)` and
`fallback.record_success(rsp)` on `rsp.ok == True`. The policy returns the
new state to enter (`'retry'`, `'replay'`, `'idle'`).

#### D. Sandbox joint-limit pre-check (Phase 4 §4.2)
Currently `_clip_to_motor_limits` clips at primitive runtime. A pre-flight
**static AST validation** in `SandboxExecutor` would let us reject ill-formed
code BEFORE any joints move (cleaner fallback trigger):

- Parse `code_str` with `ast.parse`. Walk all `Call` nodes whose function
  name is one of the registered primitives.
- For `move_joint`, `move_joints`, `oscillate_joint`: extract literal
  `name` argument and check it is in the known joint list. Extract literal
  `angle` / `center` / `amplitude` and verify within joint limits. (Skip
  validation when arg is not a literal, e.g. inside a `for` loop
  parameterized by `amp`.)
- For `move_arm_ik`: extract literal `xyz` and check norm is reasonable
  (< 0.6 m from torso origin) — VLM occasionally generates absurd targets.
- Returns a `ValidationResult(ok, reasons)`. If invalid, sandbox refuses
  to exec and triggers fallback Tier A.

#### E. (Deferred — not in this iteration)
- Pinocchio self-collision (§4.3): nice-to-have but adds significant
  complexity. Defer unless we observe actual self-collisions in evaluation.

### Critical files

| File | Change |
|---|---|
| `nao_VLM/controllers/nao_vlm_test/vlm_client.py` | Replace single example with 3–4 diverse examples in `_BASE_SYSTEM_PROMPT`; add affect/distance modulation rules |
| `nao_VLM/controllers/nao_vlm_test/nao_vlm_test.py` | Add `difflib` typo lookup in `move_joint`/`move_joints`/`oscillate_joint`; integrate `FallbackPolicy` in main loop |
| `nao_VLM/controllers/nao_vlm_test/sandbox_exec.py` | Add `validate(code_str, joint_limits, motors)` AST-walk method returning `ValidationResult`; `run()` calls it before `exec` |
| **NEW**: `nao_VLM/controllers/nao_vlm_test/fallback.py` | `FallbackPolicy` class: retry budget, action history (deque(maxlen=3)), state machine for handle_failure/record_success |

### Reusable existing pieces

- `SandboxExecutor.run()` — extend (don't replace) to call `.validate()` first.
- `VLMResponse` dataclass — has `.python_code`, `.semantic_context` already; cache as-is in fallback action history.
- `VLMTrigger.mark_idle()` — fallback Tier C just calls this.
- `NaoVlmAPI.idle(duration)` — fallback Tier C uses this for the actual motion.
- The frames passed to the failed `worker.kick(frames)` are already the same buffer sample; for Tier A retry, just re-call `client.call(frames)` (no need to resample the buffer — same evidence, fresh attempt).

### Verification plan

End-to-end smoke run with the existing 20s phone clip:

1. **Diversity check**: across 8+ kicks, verify at least 3 distinct primitive
   compositions appear in `[VLM] code:` blocks (not all variants of the
   waving template). Check console for `trajectory='min_jerk'` appearing in
   at least one VLM-generated call.
2. **Typo tolerance**: temporarily inject `move_joint('rshoulderpitch', ...)`
   (lowercase) into a manually-crafted VLM response and confirm sandbox
   rejects it with a message naming `RShoulderPitch` as the close match.
3. **Tier A retry**: artificially make `parse_vlm_output` return empty code
   on every other call (toggle a debug flag), confirm the controller logs
   `[fallback] tier A retry (attempt 1/1)` and re-kicks.
4. **Tier B replay**: after at least one successful kick, force two
   consecutive failures. Confirm log shows `[fallback] tier B replay of cached
   action` and NAO re-runs the previously-cached primitive sequence.
5. **Tier C idle**: with no cache populated, force a failure on the very
   first kick. Confirm `[fallback] tier C idle` and NAO runs the breathing
   primitive instead of staying frozen.
6. **AST validation rejection**: craft a VLM response with
   `move_joint('RShoulderPitch', 99.0, 0.5)` (out of range) and verify the
   sandbox rejects pre-flight, fallback activates Tier A.
7. **No regression**: existing kicks for a normal waving video should still
   complete successfully with same observable behavior on NAO.

### Out of scope (still left after this iteration)
- Phase 5 evaluation harness (next iteration)
- Phase 6 stretches (action preemption, local VLM, MediaPipe overlay)
- Pinocchio self-collision check (deferred per §15.E)