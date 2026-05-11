# NAO VLM Embodied AI

This is a NAO robot Embodied AI project based on the Webots simulator and the Pinocchio dynamics engine. This project provides a highly encapsulated physical API layer for Vision Large Language Models (VLM), enabling the large model to directly take over the robot's movement and vision systems.

## 🌟 Core Features
- **Parallel Universe Architecture**: Webots handles the physical body and collision execution, while Pinocchio acts as the mathematical cerebellum, calculating inverse kinematics (IK) in real-time.
- **VLM-Friendly Motion Grammar**: Exposes low-level physical primitives such as `move_joint()`, `move_joints()`, `move_arm_ik()`, `move_head()`, `set_hand()`, `oscillate_joint()`, `hold()`, and `idle()`.
- **Safety Interception Armor**: Built-in joint limits, self-collision detection, and distance verification to prevent dangerous movements caused by VLM "hallucinations".

## 🛠️ Environment & Installation
- Ubuntu 22.04
- Webots R2025a (world file currently declares `#VRML_SIM R2025a`)
- Python 3.10+
- Pinocchio, NumPy, OpenCV

### 1. Install Webots
Download the official Webots Debian package for Ubuntu 22.04 from the [Cyberbotics GitHub Releases](https://github.com/cyberbotics/webots/releases). Open your terminal and install it via `apt`:
```bash
sudo apt install ./webots_2025a_amd64.deb
```

### 2. Install Python Dependencies
Install the necessary mathematical and visual processing libraries:
```bash
pip install -r requirements.txt
```

### 3. NAO URDF
The needed NAO URDF assets are already present under `nao_VLM/nao_robot/`. Pinocchio uses these files for IK and CoM calculations.

## 🚀 Quick Start

### 1. Launch Webots
You can quickly launch the simulator and load the specific world file directly from your terminal:
```bash
webots worlds/nao_VLM.wbt
```

### 2. Enable Supervisor Mode (Troubleshooting)
To allow the Python controller to execute macroscopic navigation (like sliding/walking via the `Maps_to` API), the NAO robot node **must** have supervisor privileges. 
- In the left **Scene Tree** of the Webots interface, double-click to expand the `Nao "NAO"` node.
- Scroll down, find the `supervisor` field, and change it from `FALSE` to `TRUE`.
- Save the world file (`Ctrl + S`).

### 3. Run the Controller
Ensure the `controller` field of the NAO node is set to your Python script (e.g., `nao_vlm_test`). Reset (⏪) and Play (▶️) the simulation. The Python controller will automatically start the VLM API polling test.

### 4. Real-time Webcam Mode
The default target workflow is real-time interaction:

- `INPUT_MODE=webcam`
- `WEBCAM_SOURCE=0`
- `RUN_MODE=periodic`
- `FRAMEBUFFER_BACKEND=auto`

This means the controller continuously samples recent frames from the local webcam,
queries the VLM with a general control-contract prompt, and executes the returned
robot motion sequence.

### 5. One-shot Example Video Demo
The recorded example video is only for debugging. To run a single-turn demo from
`example_video/webcam_20260425_072825.mp4`:

1. Create the conda environment:
```bash
conda env create -f environment.yml
```
2. Copy `.env.debug.example-video` to `.env`.
3. Point `nao_VLM/controllers/nao_vlm_test/runtime.ini` to the conda Python interpreter.
4. Launch Webots and run the `nao_vlm_test` controller.

In `oneshot` mode, the controller samples a short frame sequence from the video,
calls the VLM once, executes the returned Python primitive sequence once, saves
artifacts under `artifacts/oneshot/`, and exits.

### 6. Phase 5 Evaluation
Phase 5 adds a reproducible benchmark around `RUN_MODE=oneshot`.

```bash
python -m evaluation.run_benchmark --scenario-set pilot --rounds 1 --method cap
python -m evaluation.run_benchmark --scenario-set pilot --rounds 1 --method rule_baseline
python -m evaluation.judge artifacts/eval/*.json --output artifacts/eval/report.md
```

For Docker-based Linux/Webots evaluation from macOS:

```bash
docker build -f docker/Dockerfile -t humanoid-webots:phase5 .
docker run --rm -v "$PWD:/workspace" -e llm_api_key="$OPENAI_API_KEY" \
  humanoid-webots:phase5 \
  python3 -m evaluation.run_benchmark --scenario-set pilot --rounds 1 --method cap --headless
```

### 7. Live Camera Demo
Native Linux webcam:

```bash
INPUT_MODE=webcam WEBCAM_SOURCE=0 RUN_MODE=periodic webots nao_VLM/worlds/nao_VLM.wbt
```

Mac camera into Docker or a remote Linux session:

```bash
python3 scripts/local_camera_server.py --source 0 --port 5000 --fps 10
# Docker controller: WEBCAM_SOURCE=http://host.docker.internal:5000/video_feed
# SSH-tunneled remote/HPC controller: WEBCAM_SOURCE=http://127.0.0.1:5000/video_feed
```

Then launch the live controller with:

```bash
WEBCAM_SOURCE=0 bash scripts/run_live_camera_demo.sh
```

### 8. 协作与日常使用
如果你是协作者，先看 `COLLABORATOR_GUIDE.md`。里面包括当前阶段、常用脚本、以及只推送 `artifacts/screen_recordings_matched/` 的约定。

## 📚 Helpful Resources and Links

- [Webots Official Website](https://cyberbotics.com)
- [Webots GitHub Repository](https://github.com/cyberbotics/webots)
- [Webots Installation Guide](https://cyberbotics.com/doc/guide/installation-procedure#from-the-installation-file)

