#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class CatDetectionNode(Node):
    def __init__(self):
        super().__init__('cat_detection')
        
        # YOLO model (downloads automatically first time)
        self.model = YOLO('yolov8n.pt')
        self.bridge = CvBridge()

        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        # Publish cat position (x, y, z=distance)
        self.cat_pub = self.create_publisher(Point, '/cat_position', 10)
        # Publish annotated image for monitoring
        self.img_pub = self.create_publisher(Image, '/cat_detection/image', 10)

        self.get_logger().info('Cat Detection Node started!')

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        h, w = frame.shape[:2]

        # Run YOLO detection
        results = self.model(frame, verbose=False)

        cat_detected = False

        for result in results:
            for box in result.boxes:
                # class 15 = cat in COCO dataset
                if int(box.cls) == 15:
                    cat_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2  # center x
                    cy = (y1 + y2) // 2  # center y

                    # Publish cat center position
                    pt = Point()
                    pt.x = float(cx)
                    pt.y = float(cy)
                    pt.z = 0.0  # depth added later
                    self.cat_pub.publish(pt)

                    # Draw box on frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, 'CAT', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if not cat_detected:
            self.get_logger().info('No cat detected', throttle_duration_sec=2.0)

        # Publish annotated image
        self.img_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = CatDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()