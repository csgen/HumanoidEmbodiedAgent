# NAO VLM Embodied AI

This is a NAO robot Embodied AI project based on the Webots simulator and the Pinocchio dynamics engine. This project provides a highly encapsulated physical API layer for Vision Large Language Models (VLM), enabling the large model to directly take over the robot's movement and vision systems.

## 🌟 Core Features
- **Parallel Universe Architecture**: Webots handles the physical body and collision execution, while Pinocchio acts as the mathematical cerebellum, calculating inverse kinematics (IK) in real-time.
- **VLM-Friendly API**: Abstracts complex low-level controls into natural language-level, high-level functions such as `look_at()`, `move_arm()`, and `Maps_to()`.
- **Safety Interception Armor**: Built-in joint limits, self-collision detection, and distance verification to prevent dangerous movements caused by VLM "hallucinations".

## 🛠️ Environment & Installation
- Ubuntu 22.04
- Webots R2023b (or compatible versions)
- Python 3.10+
- Pinocchio, NumPy, OpenCV

### 1. Install Webots
Download the official Webots Debian package for Ubuntu 22.04 from the [Cyberbotics GitHub Releases](https://github.com/cyberbotics/webots/releases). Open your terminal and install it via `apt`:
```bash
sudo apt install ./webots_2023b_amd64.deb
```

### 2. Install Python Dependencies
Install the necessary mathematical and visual processing libraries:
```bash
pip install -r requirements.txt
```

### 3. Fetch the Official NAO URDF (Important!)
Since Pinocchio requires the official NAO physical model to calculate the Center of Mass (CoM) and IK matrices, you must clone the official NAO repository into your project root folder:
```bash
git clone [https://github.com/ros-naoqi/nao_robot.git](https://github.com/ros-naoqi/nao_robot.git)
```

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
