#!/usr/bin/env python3
import subprocess
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

W, H = 1280, 720
PIPELINE = (
    'gst-launch-1.0 -q '
    'nvarguscamerasrc sensor-id=0 ! '
    '"video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1" ! '
    'nvvidconv flip-method=2 ! '
    '"video/x-raw,format=BGRx" ! '
    'videoconvert ! '
    '"video/x-raw,format=BGR" ! '
    'fdsink fd=1'
)

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/catrun', 10)
        self.get_logger().info('Starting GStreamer pipeline...')
        threading.Thread(target=self.capture_loop, daemon=True).start()

    def capture_loop(self):
        frame_size = W * H * 3
        while True:
            try:
                proc = subprocess.Popen(
                    PIPELINE, shell=True,
                    stdout=subprocess.PIPE, bufsize=frame_size)
                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        break
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
                    msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
                    msg.header.stamp = self.get_clock().now().to_msg()
                    msg.header.frame_id = 'camera'
                    self.pub.publish(msg)
                proc.kill()
            except Exception as e:
                self.get_logger().error(f'Pipeline error: {e}')
            self.get_logger().warn('Pipeline died, restarting...')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
