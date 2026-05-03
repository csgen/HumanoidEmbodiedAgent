"""
Central configuration for the NAO VLM embodied agent.

Imported by the Webots controller and (indirectly) by evaluation scripts.
Environment variables take precedence over the defaults below, so that
evaluation harnesses can override without editing this file.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Visual input
# ---------------------------------------------------------------------------
# Conceptual source of perception:
#   'webcam'     - local webcam / video file / remote MJPEG stream
#   'camera_top' - NAO's simulated CameraTop device (pure-sim closed loop)
INPUT_MODE = os.getenv('INPUT_MODE', 'webcam')

# Concrete source handed to cv2.VideoCapture (only used when INPUT_MODE=='webcam'):
#   0                                         -> local webcam device (or OBS Virtual Cam)
#   "videos/scenario_01_wave.mp4"             -> pre-recorded video file (relative to repo root)
#   "/abs/path/to/video.mp4"                  -> pre-recorded video file (absolute)
#   "http://localhost:5000/video_feed"        -> MJPEG stream via SSH reverse tunnel
_source_env = os.getenv('WEBCAM_SOURCE', '0')
if _source_env.isdigit():
    WEBCAM_SOURCE = int(_source_env)
else:
    # Strings that are URLs or absolute paths are passed through untouched.
    # Bare relative paths are resolved against the repo root, because the
    # Webots controller launches with the controller directory as CWD and
    # users typically drop their video files at <repo>/videos/.
    _is_url = '://' in _source_env
    _is_abs = Path(_source_env).is_absolute()
    if _is_url or _is_abs:
        WEBCAM_SOURCE = _source_env
    else:
        _resolved = (Path(__file__).resolve().parent.parent.parent.parent / _source_env)
        WEBCAM_SOURCE = str(_resolved)

# FrameBuffer parameters
FRAME_BUFFER_SECONDS = float(os.getenv('FRAME_BUFFER_SECONDS', '2.0'))
FRAME_BUFFER_FPS = int(os.getenv('FRAME_BUFFER_FPS', '10'))
VLM_FRAME_COUNT = int(os.getenv('VLM_FRAME_COUNT', '5'))
VLM_WINDOW_SECONDS = float(os.getenv('VLM_WINDOW_SECONDS', '1.5'))
FRAMEBUFFER_BACKEND = os.getenv('FRAMEBUFFER_BACKEND', 'auto').strip().lower()
FRAMEBUFFER_WIDTH = int(os.getenv('FRAMEBUFFER_WIDTH', '1280'))
FRAMEBUFFER_HEIGHT = int(os.getenv('FRAMEBUFFER_HEIGHT', '720'))

# Controller run mode:
#   'periodic' - original Phase-0 loop, kick VLM every fixed interval
#   'oneshot'  - sample once from webcam/video, call VLM once, execute once, then exit
#   'replay'   - skip VLM call, execute precomputed low-level code from file
RUN_MODE = os.getenv('RUN_MODE', 'periodic').strip().lower()
REPLAY_CODE_PATH = os.getenv('REPLAY_CODE_PATH', '').strip()
REPLAY_START_DELAY = float(os.getenv('REPLAY_START_DELAY', '0.0'))
ONE_SHOT_BUFFER_TIMEOUT = float(os.getenv('ONE_SHOT_BUFFER_TIMEOUT', '8.0'))
ONE_SHOT_VLM_TIMEOUT = float(os.getenv('ONE_SHOT_VLM_TIMEOUT', '90.0'))
ONE_SHOT_EXIT_AFTER_EXECUTE = os.getenv('ONE_SHOT_EXIT_AFTER_EXECUTE', '1').strip().lower() not in {
    '0', 'false', 'no'
}
ONE_SHOT_VIDEO_SETTLE_SECONDS = float(os.getenv('ONE_SHOT_VIDEO_SETTLE_SECONDS', '0.0'))
ONE_SHOT_VIDEO_CAPTURE_MODE = os.getenv('ONE_SHOT_VIDEO_CAPTURE_MODE', 'recent').strip().lower()

# ---------------------------------------------------------------------------
# VLM trigger (state-aware)
# ---------------------------------------------------------------------------
MOTION_THRESHOLD = 5.0           # mean absolute frame diff to count as "someone moved"
POST_ACTION_DELAY = 2.0          # seconds after an action completes before observing reaction
IDLE_SAFETY_TIMEOUT = 30.0       # backstop trigger when IDLE for too long (robustness)

# ---------------------------------------------------------------------------
# VLM API
# ---------------------------------------------------------------------------
LLM_API_KEY = os.getenv('llm_api_key', '')
LLM_BASE_URL = os.getenv('base_url', '')
VLM_BACKEND = os.getenv('VLM_BACKEND', 'auto').strip().lower()
VLM_MODEL = os.getenv('VLM_MODEL', 'gpt-4o')
VLM_MAX_TOKENS = int(os.getenv('VLM_MAX_TOKENS', '700'))
VLM_TEMPERATURE = 0.2
VLM_IMAGE_DETAIL = 'low'         # 'low' ≈ 85 tokens/image, 'high' ≈ 1100+ tokens
VLM_SCENARIO_HINT = os.getenv('VLM_SCENARIO_HINT', '').strip()
LOCAL_VLM_MODEL = os.getenv('LOCAL_VLM_MODEL', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct').strip()
LOCAL_VLM_LOAD_IN_4BIT = os.getenv('LOCAL_VLM_LOAD_IN_4BIT', 'auto').strip().lower()
LOCAL_VLM_SERVER_URL = os.getenv('LOCAL_VLM_SERVER_URL', '').strip()
LOCAL_VLM_NUM_CANDIDATES = max(1, int(os.getenv('LOCAL_VLM_NUM_CANDIDATES', '3')))
LOCAL_VLM_TEMPERATURE = float(os.getenv('LOCAL_VLM_TEMPERATURE', '0.65'))
LOCAL_VLM_TOP_P = float(os.getenv('LOCAL_VLM_TOP_P', '0.92'))
LOCAL_VLM_DEBUG = os.getenv('LOCAL_VLM_DEBUG', '1').strip().lower() not in {'0', 'false', 'no'}

# ---------------------------------------------------------------------------
# Robot control
# ---------------------------------------------------------------------------
# Joints whose state we mirror back into Pinocchio every step.
# Finger phalanges are bound dynamically in the controller (too many to list).
TRACKED_JOINTS = [
    'HeadYaw', 'HeadPitch',
    'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
    'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw',
    'LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll',
    'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll',
]

# Neutral / "standing with arms at natural rest" pose.
# Used by idle primitive and fallback recovery. Tuned for NAO H25.
NEUTRAL_POSE = {
    'HeadYaw': 0.0, 'HeadPitch': 0.0,
    'LShoulderPitch': 1.5, 'LShoulderRoll': 0.15,
    'LElbowYaw': -1.2, 'LElbowRoll': -0.5, 'LWristYaw': 0.0,
    'RShoulderPitch': 1.5, 'RShoulderRoll': -0.15,
    'RElbowYaw': 1.2, 'RElbowRoll': 0.5, 'RWristYaw': 0.0,
}

# ---------------------------------------------------------------------------
# URDF path resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CONTROLLERS_DIR = _HERE.parent
_NAO_VLM_DIR = _CONTROLLERS_DIR.parent
_REPO_ROOT = _NAO_VLM_DIR.parent

# Ordered list of URDF candidate paths (first existing wins)
URDF_CANDIDATES = [
    _NAO_VLM_DIR / 'nao_robot' / 'nao_description' / 'urdf' / 'naoV50_generated_urdf' / 'nao.urdf',
    _NAO_VLM_DIR / 'nao_robot' / 'nao_description' / 'urdf' / 'nao.urdf',
    Path(os.path.expanduser('~/nao_VLM/nao_robot/nao_description/urdf/naoV50_generated_urdf/nao.urdf')),
    Path(os.path.expanduser('~/nao_VLM/nao_robot/nao_description/urdf/nao.urdf')),
]


def find_urdf_path():
    """Return the first existing URDF candidate path, or None if none exist."""
    for p in URDF_CANDIDATES:
        if p.exists():
            return str(p)
    return None


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = _REPO_ROOT
VIDEOS_DIR = _REPO_ROOT / 'videos'
LOG_DIR = _REPO_ROOT / 'logs'
ARTIFACTS_DIR = _REPO_ROOT / 'artifacts' / 'oneshot'
DEMO_RECORDINGS_DIR = _REPO_ROOT / 'artifacts' / 'screen_recordings'
