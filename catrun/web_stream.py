#!/usr/bin/env python3
"""
web_stream.py - Two-mode Cat Robot Web UI
Play mode:  manual drive controls + live camera
Watch mode: live camera + cat target buttons
"""

import threading
import time
import socket

import cv2
from flask import Flask, Response, request, jsonify

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

PORT = 5000

app        = Flask(__name__)
latest_frame   = None
frame_lock     = threading.Lock()
ros_node       = None
bridge         = CvBridge()
current_target = 'none'
last_seen_cat  = 'none'
frame_count    = 0
start_time     = time.time()


class WebStreamNode(Node):
    def __init__(self):
        super().__init__('web_stream')

        self.cmd_pub  = self.create_publisher(String, '/cat_target', 10)
        self.vel_pub  = self.create_publisher(Twist,  '/cmd_vel',    10)

        self.create_subscription(
            Image, '/cat_detection/image', self.annotated_cb, 10)
        self.create_subscription(
            Image, '/camera/catrun', self.raw_cb, 10)
        self.create_subscription(
            String, '/cat_identity', self.identity_cb, 10)

        self.last_annotated_time = 0.0
        self.get_logger().info('WebStreamNode ready.')

    def annotated_cb(self, msg):
        global latest_frame, frame_count
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            with frame_lock:
                latest_frame = buf.tobytes()
            frame_count += 1
            self.last_annotated_time = time.time()
        except Exception as e:
            self.get_logger().error(f'Annotated frame error: {e}')

    def raw_cb(self, msg):
        global latest_frame, frame_count
        if time.time() - self.last_annotated_time < 2.0:
            return
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            with frame_lock:
                latest_frame = buf.tobytes()
            frame_count += 1
        except Exception as e:
            self.get_logger().error(f'Raw frame error: {e}')

    def identity_cb(self, msg):
        global last_seen_cat
        last_seen_cat = msg.data.strip()

    def send_command(self, cmd):
        global current_target
        current_target = cmd
        threading.Thread(
            target=lambda: self.cmd_pub.publish(String(data=cmd)),
            daemon=True).start()

    def send_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x  = float(linear)
        twist.angular.z = float(angular)
        threading.Thread(
            target=lambda: self.vel_pub.publish(twist),
            daemon=True).start()


def ros_thread():
    global ros_node
    rclpy.init()
    ros_node = WebStreamNode()
    rclpy.spin(ros_node)
    rclpy.shutdown()


@app.route('/snapshot')
def snapshot():
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
        ros_node.send_command(cmd)
        return jsonify({'status': 'ok', 'cmd': cmd})
    return jsonify({'status': 'error', 'msg': 'ROS not ready'}), 500


@app.route('/drive', methods=['POST'])
def drive():
    data    = request.json
    linear  = data.get('linear',  0.0)
    angular = data.get('angular', 0.0)
    if ros_node:
        ros_node.send_velocity(linear, angular)
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'msg': 'ROS not ready'}), 500


@app.route('/status')
def status():
    fps = frame_count / max(time.time() - start_time, 1)
    return jsonify({
        'target':    current_target,
        'last_seen': last_seen_cat,
        'fps':       round(fps, 1),
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

    /* Mode toggle */
    .mode-bar {
      display:flex; gap:0; border-radius:8px; overflow:hidden;
      border:2px solid #333; width:100%; max-width:400px;
    }
    .mode-btn {
      flex:1; padding:10px; border:none; cursor:pointer;
      font-family:inherit; font-size:0.85rem; font-weight:700;
      transition:all 0.2s;
    }
    .mode-btn.active { background:#ff9f43; color:#1a1a1a; }
    .mode-btn:not(.active) { background:#1a1a1a; color:#555; }

    /* Camera */
    #feed {
      width:100%; max-width:960px; border-radius:8px;
      border:2px solid #222; background:#111; min-height:200px;
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
    #msg { font-size:0.75rem; color:#2ecc71; min-height:18px; text-align:center; }

    /* ── WATCH mode buttons ── */
    #watch-panel .buttons {
      display:grid; grid-template-columns:1fr 1fr;
      gap:8px; width:100%; max-width:400px;
    }
    #watch-panel button {
      padding:14px 10px; border:none; border-radius:8px;
      font-family:inherit; font-size:0.9rem; font-weight:700;
      cursor:pointer; transition:opacity 0.15s;
    }
    #watch-panel button:active { opacity:0.7; }
    .btn-eevee  { background:#c8a96e; color:#1a1a1a; }
    .btn-pichu  { background:#f7d02c; color:#1a1a1a; }
    .btn-raichu { background:#e8631a; color:#fff; }
    .btn-stop   { background:#e74c3c; color:#fff; grid-column:1/-1; }

    /* ── PLAY mode d-pad ── */
    #play-panel { width:100%; max-width:400px; }
    .dpad {
      display:grid;
      grid-template-columns: 1fr 1fr 1fr;
      grid-template-rows:    1fr 1fr 1fr;
      gap:8px; width:220px; margin:0 auto;
    }
    .dpad button {
      padding:18px 0; border:none; border-radius:10px;
      font-size:1.4rem; font-weight:700; cursor:pointer;
      background:#1e1e2e; color:#eee; border:2px solid #333;
      transition:background 0.1s, transform 0.1s;
      user-select:none; -webkit-user-select:none;
    }
    .dpad button:active, .dpad button.pressed {
      background:#ff9f43; color:#1a1a1a; transform:scale(0.93);
    }
    .dpad .empty { background:transparent; border:none; cursor:default; }
    .drive-stop {
      display:block; width:100%; margin-top:10px;
      padding:14px; border:none; border-radius:10px;
      background:#e74c3c; color:#fff;
      font-family:inherit; font-size:1rem; font-weight:700;
      cursor:pointer;
    }
    .speed-row {
      display:flex; align-items:center; gap:10px;
      margin-top:10px; font-size:0.8rem; color:#888;
    }
    .speed-row input { flex:1; accent-color:#ff9f43; }

    /* hidden panels */
    .panel { display:none; flex-direction:column; align-items:center; gap:10px; width:100%; }
    .panel.active { display:flex; }
  </style>
</head>
<body>
  <h2>🐱 Cat Robot</h2>

  <!-- Mode toggle -->
  <div class="mode-bar">
    <button class="mode-btn active" id="btn-watch" onclick="setMode('watch')">👁 Watch</button>
    <button class="mode-btn"        id="btn-play"  onclick="setMode('play')">🎮 Play</button>
  </div>

  <!-- Camera feed -->
  <img id="feed" alt="Live feed">

  <!-- Status -->
  <div class="status-bar">
    <span class="live">LIVE</span>
    <span id="target-label">target: none</span>
    <span id="seen-label">last seen: none</span>
    <span id="fps-label">-- fps</span>
  </div>
  <div id="msg"></div>

  <!-- WATCH panel -->
  <div class="panel active" id="watch-panel">
    <div class="buttons">
      <button class="btn-eevee"  onclick="sendCmd('eevee')">🦊 Find Eevee</button>
      <button class="btn-pichu"  onclick="sendCmd('pichu')">⚡ Find Pichu</button>
      <button class="btn-raichu" onclick="sendCmd('raichu')">🔥 Find Raichu</button>
      <button class="btn-stop"   onclick="sendCmd('stop')">⛔ Stop</button>
    </div>
  </div>

  <!-- PLAY panel -->
  <div class="panel" id="play-panel">
    <div class="dpad">
      <div class="empty"></div>
      <button id="btn-fwd"  ontouchstart="drive(1,0)"  ontouchend="stopDrive()" onmousedown="drive(1,0)"  onmouseup="stopDrive()">▲</button>
      <div class="empty"></div>
      <button id="btn-left" ontouchstart="drive(0,1)"  ontouchend="stopDrive()" onmousedown="drive(0,1)"  onmouseup="stopDrive()">◄</button>
      <button id="btn-stop-c" ontouchstart="stopDrive()" onmousedown="stopDrive()">■</button>
      <button id="btn-right" ontouchstart="drive(0,-1)" ontouchend="stopDrive()" onmousedown="drive(0,-1)" onmouseup="stopDrive()">►</button>
      <div class="empty"></div>
      <button id="btn-back" ontouchstart="drive(-1,0)" ontouchend="stopDrive()" onmousedown="drive(-1,0)" onmouseup="stopDrive()">▼</button>
      <div class="empty"></div>
    </div>
    <div class="speed-row">
      <span>Speed:</span>
      <input type="range" id="speed" min="0.1" max="0.8" step="0.1" value="0.4">
      <span id="speed-val">0.4</span>
    </div>
  </div>

  <script>
    // ── Camera polling ────────────────────────────────────────────
    const feed = document.getElementById('feed');
    function pollSnapshot() {
      fetch('/snapshot?' + Date.now())
        .then(r => { if (!r.ok) throw new Error(); return r.blob(); })
        .then(blob => {
          const url = URL.createObjectURL(blob);
          const old = feed.src;
          feed.src = url;
          if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
        })
        .catch(() => {})
        .finally(() => setTimeout(pollSnapshot, 80));
    }
    pollSnapshot();

    // ── Mode toggle ───────────────────────────────────────────────
    function setMode(mode) {
      document.getElementById('watch-panel').classList.toggle('active', mode === 'watch');
      document.getElementById('play-panel').classList.toggle('active',  mode === 'play');
      document.getElementById('btn-watch').classList.toggle('active', mode === 'watch');
      document.getElementById('btn-play').classList.toggle('active',  mode === 'play');
      stopDrive();
    }

    // ── Watch mode commands ───────────────────────────────────────
    function sendCmd(cmd) {
      document.getElementById('msg').innerText = 'Sending: ' + cmd + '...';
      fetch('/command', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({cmd})
      })
      .then(r => r.json())
      .then(d => { document.getElementById('msg').innerText = '✅ ' + (d.cmd || d.msg); })
      .catch(() => { document.getElementById('msg').innerText = '❌ Failed'; });
    }

    // ── Play mode drive ───────────────────────────────────────────
    let driveInterval = null;

    function getSpeed() {
      return parseFloat(document.getElementById('speed').value);
    }

    function drive(linear, angular) {
      stopDrive();
      const spd = getSpeed();
      function send() {
        fetch('/drive', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({
            linear:  linear  * spd,
            angular: angular * spd * 2
          })
        }).catch(() => {});
      }
      send();
      driveInterval = setInterval(send, 150);
    }

    function stopDrive() {
      if (driveInterval) { clearInterval(driveInterval); driveInterval = null; }
      fetch('/drive', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({linear: 0, angular: 0})
      }).catch(() => {});
    }

    // Speed slider
    document.getElementById('speed').addEventListener('input', function() {
      document.getElementById('speed-val').innerText = this.value;
    });

    // ── Status polling ────────────────────────────────────────────
    function pollStatus() {
      fetch('/status')
        .then(r => r.json())
        .then(d => {
          document.getElementById('target-label').innerText = 'target: '    + d.target;
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


if __name__ == '__main__':
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = '?.?.?.?'

    threading.Thread(target=ros_thread, daemon=True).start()
    time.sleep(2)

    print(f'''
┌─────────────────────────────────────┐
│  🐱  Cat Robot Web UI               │
├─────────────────────────────────────┤
│  http://{ip}:{PORT}
│                                     │
│  WATCH: Find cats + live feed       │
│  PLAY:  Drive robot + live feed     │
└─────────────────────────────────────┘
''')
    app.run(host='0.0.0.0', port=PORT, threaded=True)