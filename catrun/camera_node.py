#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, '/camera/catrun', 10)

        pipeline = (
            'nvarguscamerasrc sensor-id=0 ! '
            'video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1 ! '
            'nvvidconv flip-method=2 ! '
            'video/x-raw, format=BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=BGR ! '
            'appsink drop=1'
        )

        self.get_logger().info('Opening camera...')
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error('❌ Camera failed to open!')
            return

        self.get_logger().info('✅ Camera opened! Publishing to /camera/catrun')
        self.create_timer(1.0/30.0, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn('Failed to grab frame')
            return
        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()