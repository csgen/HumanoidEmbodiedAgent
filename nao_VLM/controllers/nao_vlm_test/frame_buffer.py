"""
FrameBuffer: a thread-safe rolling window of recent webcam frames plus a
continuously-updated motion score.

Design:
- A background daemon thread opens `source` with cv2.VideoCapture and pushes
  frames into a bounded deque at ~`fps` Hz.
- Each iteration also computes an absolute-difference motion score between
  consecutive grayscale frames. That score is exposed via `last_motion_score`
  so the VLMTrigger can read it without grabbing any lock.
- `sample_recent(n)` returns n uniformly-spaced, base64-encoded JPEG frames
  drawn from the buffer, suitable for GPT-4o multi-image input.

The buffer is safe to call `sample_recent` from another thread while the
capture loop runs; the deque operations and the single-float score are each
atomic in CPython.
"""
from __future__ import annotations

import base64
import collections
import platform
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np


_DEFAULT_LINUX_RESOLUTIONS = [
    (1280, 720),
    (640, 480),
]


class FrameBuffer:
    def __init__(
        self,
        source=0,
        buffer_seconds: float = 2.0,
        fps: int = 10,
        jpeg_quality: int = 80,
        downscale_width: Optional[int] = None,
        backend: str = 'auto',
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        source
            Anything cv2.VideoCapture accepts: int device index, file path,
            or MJPEG URL (e.g. "http://localhost:5000/video_feed").
        buffer_seconds
            Length of the rolling window in seconds.
        fps
            Target capture rate. Drives both the buffer cadence and the
            motion-score update rate.
        jpeg_quality
            JPEG quality for base64 encoding (sent to VLM).
        downscale_width
            If set, frames are resized so width == downscale_width before being
            stored. Useful to cut token cost on cloud VLM calls.
        """
        self.source = source
        self.fps = max(1, int(fps))
        self.jpeg_quality = int(jpeg_quality)
        self.downscale_width = downscale_width
        self.backend = str(backend or 'auto').strip().lower()
        self.frame_width = int(frame_width) if frame_width else None
        self.frame_height = int(frame_height) if frame_height else None

        maxlen = max(2, int(buffer_seconds * self.fps))
        # Each entry: (timestamp_float, bgr_ndarray)
        self.buffer: "collections.deque[Tuple[float, np.ndarray]]" = collections.deque(maxlen=maxlen)

        self.last_motion_score: float = 0.0
        self._frame_count: int = 0
        self._stop_event = threading.Event()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, name="FrameBuffer", daemon=True
        )

    def _resolve_backends(self):
        backends = {
            'any': cv2.CAP_ANY,
            'v4l2': getattr(cv2, 'CAP_V4L2', cv2.CAP_ANY),
            'gstreamer': getattr(cv2, 'CAP_GSTREAMER', cv2.CAP_ANY),
        }
        if self.backend != 'auto':
            return [(self.backend, backends.get(self.backend, cv2.CAP_ANY))]
        if platform.system().lower() == 'linux' and isinstance(self.source, int):
            return [
                ('v4l2', backends['v4l2']),
                ('any', backends['any']),
                ('gstreamer', backends['gstreamer']),
            ]
        return [('any', backends['any'])]

    def _resolution_candidates(self):
        requested = []
        if self.frame_width and self.frame_height:
            requested.append((self.frame_width, self.frame_height))
        if platform.system().lower() == 'linux' and isinstance(self.source, int):
            for item in _DEFAULT_LINUX_RESOLUTIONS:
                if item not in requested:
                    requested.append(item)
        return requested or [(0, 0)]

    def _open_capture(self):
        errors = []
        for backend_name, backend_value in self._resolve_backends():
            for width, height in self._resolution_candidates():
                cap = cv2.VideoCapture(self.source, backend_value)
                if not cap.isOpened():
                    errors.append(f'backend={backend_name} open failed')
                    cap.release()
                    continue

                if width > 0 and height > 0:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)

                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap, errors
                errors.append(f'backend={backend_name} resolution={width}x{height} no frames')
                cap.release()
        return None, errors

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "FrameBuffer":
        """Start the capture thread. Safe to call once."""
        if not self._thread.is_alive():
            self._thread.start()
            # Wait briefly for the capture device to come up so sample_recent
            # from the caller can immediately see a frame.
            self._started.wait(timeout=2.0)
        return self

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ capture loop

    def _capture_loop(self) -> None:
        cap, errors = self._open_capture()
        if cap is None:
            detail = '; '.join(errors) if errors else 'no more detail'
            print(f"[FrameBuffer] ERROR: failed to open source {self.source!r} ({detail})")
            self._started.set()  # unblock start() even on failure
            return

        interval = 1.0 / self.fps
        prev_gray: Optional[np.ndarray] = None
        first_ok = False

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret or frame is None:
                    # MJPEG streams can hiccup; keep retrying
                    time.sleep(0.02)
                    continue

                if self.downscale_width and frame.shape[1] > self.downscale_width:
                    h, w = frame.shape[:2]
                    new_h = int(h * self.downscale_width / w)
                    frame = cv2.resize(frame, (self.downscale_width, new_h))

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    diff = cv2.absdiff(gray, prev_gray)
                    # Atomic write of a float - no lock required in CPython
                    self.last_motion_score = float(np.mean(diff))
                prev_gray = gray

                self.buffer.append((time.time(), frame))
                self._frame_count += 1

                if not first_ok:
                    first_ok = True
                    self._started.set()

                # Pace to target fps
                sleep_for = interval - (time.time() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            cap.release()
            if not self._started.is_set():
                self._started.set()

    # ------------------------------------------------------------------ sampling

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive() and not self._stop_event.is_set()

    def sample_recent(self, n: int = 5) -> List[str]:
        """
        Return up to n base64-encoded JPEG frames, uniformly spaced across the
        current buffer contents (chronological order, oldest first).

        Returns an empty list if the buffer has fewer than n frames.
        """
        n = max(1, int(n))
        snap = list(self.buffer)  # snapshot; deque is not sliceable safely under writes
        if len(snap) < n:
            return []

        # Uniform indices across snapshot
        indices = np.linspace(0, len(snap) - 1, n).astype(int)
        frames = [snap[i][1] for i in indices]
        return [self._encode(f) for f in frames]

    def latest(self) -> Optional[np.ndarray]:
        """Return the most recent BGR frame, or None."""
        if not self.buffer:
            return None
        return self.buffer[-1][1]

    def _encode(self, frame: np.ndarray) -> str:
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return ''
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def stats(self) -> dict:
        """Debug snapshot."""
        return {
            'source': self.source,
            'fps': self.fps,
            'buffer_len': len(self.buffer),
            'buffer_max': self.buffer.maxlen,
            'frames_captured': self._frame_count,
            'last_motion_score': self.last_motion_score,
            'alive': self.is_alive,
        }
