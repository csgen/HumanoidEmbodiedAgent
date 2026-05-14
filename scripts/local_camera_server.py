"""
Local camera server — streams the host machine's webcam (or a video file) as
MJPEG over HTTP, so a controller on another machine or VM can read it.

Two intended setups:
  1. VirtualBox host -> guest: run on the host with --host 0.0.0.0; the guest
     reads it at http://10.0.2.2:5000/video_feed  (10.0.2.2 is the host as seen
     from a VirtualBox NAT guest).
  2. Laptop -> cloud: run on the laptop (default --host 127.0.0.1) and reach it
     from the cloud over an SSH reverse tunnel:
     ssh -R 5000:localhost:5000 user@cloud

The controller reads the stream via
`cv2.VideoCapture("http://<host>:5000/video_feed")`.

Usage:
    # live webcam, localhost only (use with an SSH tunnel)
    python scripts/local_camera_server.py

    # live webcam, reachable by a VirtualBox guest on a trusted local network
    python scripts/local_camera_server.py --source 0 --host 0.0.0.0

    # loop a prerecorded clip
    python scripts/local_camera_server.py --source videos/scenario_01_wave.mp4 --loop

    # custom port
    python scripts/local_camera_server.py --port 5001

Security note:
    The server binds to 127.0.0.1 by default (not reachable off-machine).
    --host 0.0.0.0 exposes it on all interfaces — fine on a trusted local
    network (e.g. a VirtualBox host/guest pair), but don't use it on an
    untrusted network; prefer an SSH reverse tunnel there.
"""
import argparse
import sys
import time

import cv2

try:
    from flask import Flask, Response
except ImportError:
    print("ERROR: Flask is not installed. Run: pip install flask", file=sys.stderr)
    sys.exit(1)


app = Flask(__name__)

# Set by main() before app.run()
_cap = None
_is_video_file = False
_loop = True
_jpeg_quality = 70
_target_fps = 30


def _open_capture(source):
    """Open a cv2.VideoCapture. Accepts int (device index) or str (file path)."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source!r}")
    return cap


def _generate_frames():
    """Generator yielding MJPEG frames."""
    global _cap
    frame_interval = 1.0 / _target_fps
    last_emit = 0.0

    while True:
        ret, frame = _cap.read()
        if not ret:
            if _is_video_file and _loop:
                # Rewind the video and keep looping
                _cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # No more frames (non-looping file) or webcam error
            time.sleep(0.05)
            continue

        # Throttle to target FPS (webcams often free-run at 30+ fps)
        now = time.time()
        if now - last_emit < frame_interval:
            time.sleep(frame_interval - (now - last_emit))
        last_emit = time.time()

        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, _jpeg_quality])
        if not ok:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """MJPEG endpoint. Consumed by cv2.VideoCapture on the cloud side."""
    return Response(
        _generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/health')
def health():
    """Simple health check."""
    return {'status': 'ok', 'source': str(_cap is not None)}


@app.route('/')
def index():
    """Tiny browser-friendly preview (for local debugging)."""
    return (
        '<html><body style="margin:0;background:#111;">'
        '<img src="/video_feed" style="max-width:100%;max-height:100vh;">'
        '</body></html>'
    )


def main():
    global _cap, _is_video_file, _loop, _jpeg_quality, _target_fps

    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--source', default='0',
                    help='Device index (e.g. 0) or path to a video file. Default: 0')
    ap.add_argument('--port', type=int, default=5000, help='Listening port (default 5000)')
    ap.add_argument('--host', default='127.0.0.1',
                    help='Bind address. Default 127.0.0.1 (localhost only). Use 0.0.0.0 to let '
                         'a VirtualBox guest or other host on a trusted LAN reach the stream.')
    ap.add_argument('--loop', action='store_true',
                    help='Loop the video file when it ends (ignored for webcam)')
    ap.add_argument('--fps', type=int, default=30, help='Target frame rate for the stream')
    ap.add_argument('--quality', type=int, default=70, help='JPEG quality 1-100')
    args = ap.parse_args()

    # Parse source: "0" -> 0, anything else -> file path (or device string)
    src = int(args.source) if args.source.isdigit() else args.source
    _is_video_file = not isinstance(src, int)
    _loop = args.loop or not _is_video_file  # webcams are inherently continuous
    _jpeg_quality = max(1, min(100, args.quality))
    _target_fps = max(1, args.fps)

    print(f"[local_camera_server] opening source: {src!r}")
    _cap = _open_capture(src)

    w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = _cap.get(cv2.CAP_PROP_FPS)
    print(f"[local_camera_server] source ready: {w}x{h} @ ~{src_fps:.1f} fps")
    print(f"[local_camera_server] serving MJPEG at http://{args.host}:{args.port}/video_feed")
    print(f"[local_camera_server] preview in browser: http://{args.host}:{args.port}/")
    if args.host != '127.0.0.1':
        print(f"[local_camera_server] NOTE: host is {args.host}, not loopback — the "
              f"stream is reachable off-machine. Fine on a trusted LAN (e.g. a "
              f"VirtualBox host/guest pair, where the guest reads "
              f"http://10.0.2.2:{args.port}/video_feed). On an untrusted network, "
              f"prefer an SSH reverse tunnel instead.")

    # threaded=True lets the generator keep streaming while the view handler returns
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == '__main__':
    main()
