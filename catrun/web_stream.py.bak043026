#!/usr/bin/env python3
"""
web_stream.py
Phone-accessible web UI for the cat robot. Replaces catstream.py.

- Live annotated camera feed (from /cat_detection/image when available,
  falls back to raw GStreamer capture)
- Find Eevee / Pichu / Raichu / Stop buttons
- Shows current target and last seen cat
- Publishes commands to /cat_target

Run:
    export DISPLAY=:0
    python3 web_stream.py
Then open http://100.64.68.68:5000 on your phone.
"""

import subprocess
import threading
import time
import socket
import os

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# ── config ────────────────────────────────────────────────────────────────────
PORT    = 5000
W, H    = 1280, 720
GST_PIPELINE = (
    f'gst-launch-1.0 -q '
    f'nvarguscamerasrc sensor-id=0 ! '
    f'"video/x-raw(memory:NVMM),width={W},height={H},framerate=30/1" ! '
    f'nvvidconv flip-method=2 ! '
    f'"video/x-raw,format=BGRx" ! '
    f'videoconvert ! '
    f'"video/x-raw,format=BGR" ! '
    f'fdsink fd=1'
)
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# ── shared state ──────────────────────────────────────────────────────────────
latest_frame   = None        # JPEG bytes
frame_lock     = threading.Lock()
ros_node       = None
bridge         = CvBridge()
current_target = 'none'
last_seen_cat  = 'none'
frame_count    = 0
start_time     = time.time()
# ─────────────────────────────────────────────────────────────────────────────


# ── camera capture (GStreamer subprocess fallback) ────────────────────────────

def gst_capture_loop():
    """
    Captures raw BGR frames from GStreamer and stores as JPEG.
    Used when /cat_detection/image is not available.
    Only used if ROS annotated frames stop coming.
    """
    global latest_frame, frame_count
    enc = [cv2.IMWRITE_JPEG_QUALITY, 80]
    print('[camera] Starting GStreamer pipeline...')

    while True:
        try:
            proc = subprocess.Popen(
                GST_PIPELINE, shell=True,
                stdout=subprocess.PIPE, bufsize=W * H * 3)
            frame_size = W * H * 3

            while True:
                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                ok, jpg = cv2.imencode('.jpg', frame, enc)
                if ok:
                    with frame_lock:
                        # Only update from GStreamer if ROS hasn't given us a frame recently
                        latest_frame = jpg.tobytes()
                    frame_count += 1

            proc.kill()
        except Exception as e:
            print(f'[camera] Error: {e}')

        print('[camera] Pipeline died, restarting in 2s...')
        time.sleep(2)


# ── ROS node ──────────────────────────────────────────────────────────────────

class WebStreamNode(Node):

    def __init__(self):
        super().__init__('web_stream')

        self.cmd_pub = self.create_publisher(String, '/cat_target', 10)

        # Subscribe to annotated frames from cat_detector
        # PRIMARY: annotated frames from cat_detector (PLAY mode)
        self.create_subscription(
            Image, '/cat_detection/image', self.annotated_cb, 10)

        # FALLBACK: raw frames from camera_node (WATCH mode)
        self.create_subscription(
            Image, '/camera/catrun', self.raw_cb, 10)

        # Subscribe to identity updates to show on UI
        self.create_subscription(
            String, '/cat_identity', self.identity_cb, 10)

        self.last_annotated_time = 0.0
        self.ANNOTATED_TIMEOUT = 2.0  # seconds
        self.get_logger().info('WebStreamNode ready.')

    def annotated_cb(self, msg: Image):
        """Annotated frames from cat_detector — PLAY mode"""
        global latest_frame, frame_count
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            _, buf = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # Was 80
            with frame_lock:
                latest_frame = buf.tobytes()
            frame_count += 1
            self.last_annotated_time = time.time()
        except Exception as e:
            self.get_logger().error(f'Annotated frame error: {e}')

    def raw_cb(self, msg: Image):
        """Raw frames from camera_node — WATCH mode fallback"""
        global latest_frame, frame_count
        # Only use raw if annotated frames stopped coming
        if time.time() - self.last_annotated_time < self.ANNOTATED_TIMEOUT:
            return
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50]) # was 80
            with frame_lock:
                latest_frame = buf.tobytes()
            frame_count += 1
        except Exception as e:
            self.get_logger().error(f'Raw frame error: {e}')

    def identity_cb(self, msg: String):
        global last_seen_cat
        last_seen_cat = msg.data.strip()

    def send_command(self, cmd: str):
        global current_target
        current_target = cmd
        self.cmd_pub.publish(String(data=cmd))
        self.get_logger().info(f'Command: {cmd}')


# ── ROS thread ────────────────────────────────────────────────────────────────

def ros_thread():
    global ros_node
    rclpy.init()
    ros_node = WebStreamNode()
    rclpy.spin(ros_node)
    rclpy.shutdown()


# ── MJPEG generator ───────────────────────────────────────────────────────────

def generate_mjpeg():
    while True:
        # New
        now = time.time()
        # ✅ Cap at 10fps for web stream — reduces bandwidth
        if now - last_sent < 0.1:
            time.sleep(0.02)
            continue
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n')
        else:
            time.sleep(0.02)


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/video')
def video():
    # return Response(generate_mjpeg(),
    #                mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'Access-Control-Allow-Origin': '*',
        }
    )

@app.route('/snapshot')
def snapshot():
    """Single JPEG frame — for polling fallback"""
    with frame_lock:
        frame = latest_frame
    if frame:
        return Response(frame, mimetype='image/jpeg',
                       headers={'Cache-Control': 'no-cache'})
    return '', 204

@app.route('/command', methods=['POST'])
def command():
    cmd = request.json.get('cmd', '').strip().lower()
    if ros_node:
         # New
         # ✅ Run in thread so Flask doesn't block
        threading.Thread(
            target=ros_node.send_command, 
            args=(cmd,), 
            daemon=True
        ).start()
        # was: ros_node.send_command(cmd)
        return jsonify({'status': 'ok', 'cmd': cmd})
    return jsonify({'status': 'error', 'msg': 'ROS not ready'}), 500


@app.route('/status')
def status():
    fps = frame_count / max(time.time() - start_time, 1)
    return jsonify({
        'target':    current_target,
        'last_seen': last_seen_cat,
        'fps':       round(fps, 1),
        'frames':    frame_count,
    })


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cat Robot</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
    * { margin:0; padding:0; box-sizing:border-box; }
    body {
      background:#0d0d0d; color:#eee;
      font-family:'Space Mono', monospace;
      display:flex; flex-direction:column;
      align-items:center; padding:12px; gap:10px;
      min-height:100vh;
    }
    h2 { color:#ff9f43; font-size:1.1rem; letter-spacing:0.1em; margin-top:6px; }
    #feed {
      width:100%; max-width:960px;
      border-radius:8px; border:2px solid #222;
      background:#111;
    }
    .status-bar {
      display:flex; gap:12px; flex-wrap:wrap;
      justify-content:center; font-size:0.7rem; color:#555;
    }
    .live { color:#2ecc71; }
    .live::before {
      content:''; display:inline-block;
      width:7px; height:7px; border-radius:50%;
      background:#2ecc71; margin-right:5px;
      animation:blink 1.4s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
    .buttons {
      display:grid; grid-template-columns:1fr 1fr;
      gap:8px; width:100%; max-width:400px;
    }
    button {
      padding:14px 10px; border:none; border-radius:8px;
      font-family:inherit; font-size:0.9rem; font-weight:700;
      cursor:pointer; transition:opacity 0.15s;
    }
    button:active { opacity:0.7; }
    .btn-eevee  { background:#c8a96e; color:#1a1a1a; }
    .btn-pichu  { background:#f7d02c; color:#1a1a1a; }
    .btn-raichu { background:#e8631a; color:#fff; }
    .btn-stop   { background:#e74c3c; color:#fff; grid-column:1/-1; }
    #msg {
      font-size:0.75rem; color:#2ecc71;
      min-height:18px; text-align:center;
    }
  </style>
</head>
<body>
  <h2>🐱 Cat Robot</h2>
  <img id="feed" alt="Live feed"
     style="width:100%; max-width:960px; border-radius:8px; 
            border:2px solid #222; background:#111; min-height:240px;">
  <div class="status-bar">
    <span class="live">LIVE</span>
    <span id="target-label">target: none</span>
    <span id="seen-label">last seen: none</span>
    <span id="fps-label">-- fps</span>
  </div>
  <div id="msg"></div>
  <div class="buttons">
    <button class="btn-eevee"  onclick="sendCmd('eevee')">🦊 Find Eevee</button>
    <button class="btn-pichu"  onclick="sendCmd('pichu')">⚡ Find Pichu</button>
    <button class="btn-raichu" onclick="sendCmd('raichu')">🔥 Find Raichu</button>
    <button class="btn-stop"   onclick="sendCmd('stop')">⛔ Stop</button>
  </div>
  <script>
  const feed = document.getElementById('feed');

  // Always use polling — most reliable on mobile
  function pollSnapshot() {
    fetch('/snapshot?' + Date.now())
      .then(r => {
        if (!r.ok) throw new Error('no frame');
        return r.blob();
      })
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const old = feed.src;
        feed.src = url;
        if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
      })
      .catch(() => {})
      .finally(() => setTimeout(pollSnapshot, 80)); // ~12fps
  }

  pollSnapshot();

  function sendCmd(cmd) {
    document.getElementById('msg').innerText = 'Sending: ' + cmd + '...';
    fetch('/command', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({cmd: cmd})
    })
    .then(r => r.json())
    .then(d => {
      document.getElementById('msg').innerText = '✅ ' + (d.cmd || d.msg);
    })
    .catch(() => {
      document.getElementById('msg').innerText = '❌ Failed to send';
    });
  }

  function pollStatus() {
    fetch('/status')
      .then(r => r.json())
      .then(d => {
        document.getElementById('target-label').innerText = 'target: ' + d.target;
        document.getElementById('seen-label').innerText   = 'last seen: ' + d.last_seen;
        document.getElementById('fps-label').innerText    = d.fps + ' fps';
      })
      .catch(() => {});
  }
  setInterval(pollStatus, 2000);
  pollStatus();
</script>
</body>
</html>'''


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = '?.?.?.?'

    # Start ROS
    threading.Thread(target=ros_thread, daemon=True).start()

    print(f'''
┌─────────────────────────────────────┐
│  🐱  Cat Robot Web UI               │
├─────────────────────────────────────┤
│  Local:     http://{ip}:{PORT}
│  Tailscale: http://100.64.68.68:{PORT}
│                                     │
│  Buttons: Find Eevee/Pichu/Raichu   │
│  Stop button halts the robot        │
└─────────────────────────────────────┘
''')

    app.run(host='0.0.0.0', port=PORT, threaded=True)
