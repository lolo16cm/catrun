#!/usr/bin/env python3
"""
web_stream.py
Flask web server that:
- Streams live annotated camera feed to phone browser
- Accepts "find eevee" / "find pichu" / "find raichu" / "stop" commands
- Publishes commands to /cat_target ROS topic
"""

from flask import Flask, Response, request, jsonify
import cv2
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

app = Flask(__name__)

# ── shared state ──────────────────────────────────────────────────────────────
latest_frame = None
frame_lock   = threading.Lock()
ros_node     = None
bridge       = CvBridge()
# ─────────────────────────────────────────────────────────────────────────────


class WebStreamNode(Node):
    def __init__(self):
        super().__init__('web_stream')

        # Publish commands to cat_detector and seek_cat
        self.cmd_pub = self.create_publisher(String, '/cat_target', 10)

        # Subscribe to annotated image from cat_detector
        self.create_subscription(
            Image,
            '/cat_detection/image',
            self.image_cb,
            10)

        self.get_logger().info('Web stream node ready!')

    def image_cb(self, msg):
        global latest_frame
        try:
            frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with frame_lock:
                latest_frame = buffer.tobytes()
        except Exception as e:
            self.get_logger().error(f'Frame error: {e}')

    def send_command(self, cmd: str):
        self.cmd_pub.publish(String(data=cmd))
        self.get_logger().info(f'Command sent: {cmd}')


# ── ROS thread ────────────────────────────────────────────────────────────────

def ros_thread():
    global ros_node
    rclpy.init()
    ros_node = WebStreamNode()
    rclpy.spin(ros_node)
    rclpy.shutdown()


# ── Flask routes ──────────────────────────────────────────────────────────────

def generate():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/command', methods=['POST'])
def command():
    cmd = request.json.get('cmd', '').strip().lower()
    if ros_node:
        ros_node.send_command(cmd)
        return jsonify({'status': 'ok', 'cmd': cmd})
    return jsonify({'status': 'error', 'msg': 'ROS not ready'}), 500


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Cat Robot</title>
        <style>
            body {
                background: #111;
                color: white;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 10px;
                margin: 0;
            }
            h2 { color: #00ff88; margin: 10px 0; }
            img { width: 100%; max-width: 640px; border: 2px solid #00ff88; }
            .buttons { margin: 15px 0; }
            button {
                background: #00ff88;
                color: black;
                border: none;
                padding: 12px 20px;
                margin: 6px;
                border-radius: 8px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                min-width: 120px;
            }
            button.stop {
                background: #ff4444;
                color: white;
            }
            #status {
                color: #00ff88;
                font-size: 14px;
                margin: 8px;
                min-height: 20px;
            }
        </style>
    </head>
    <body>
        <h2>🐱 Cat Robot</h2>
        <img src="/video"/>
        <div id="status">Ready</div>
        <div class="buttons">
            <button onclick="sendCmd('find eevee')">Find Eevee</button>
            <button onclick="sendCmd('find pichu')">Find Pichu</button>
            <button onclick="sendCmd('find raichu')">Find Raichu</button>
            <button class="stop" onclick="sendCmd('stop')">⛔ Stop</button>
        </div>
        <script>
            function sendCmd(cmd) {
                document.getElementById('status').innerText = 'Sending: ' + cmd + '...';
                fetch('/command', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({cmd: cmd})
                })
                .then(r => r.json())
                .then(d => {
                    document.getElementById('status').innerText = '✅ ' + d.cmd;
                })
                .catch(() => {
                    document.getElementById('status').innerText = '❌ Failed to send';
                });
            }
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    # Start ROS in background thread
    t = threading.Thread(target=ros_thread, daemon=True)
    t.start()

    # Start Flask on all interfaces so phone can connect
    print('Starting web server on http://192.168.1.188:5000')
    print('Open this URL on your phone!')
    app.run(host='0.0.0.0', port=5000, threaded=True)