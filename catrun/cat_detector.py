import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class CatDetector(Node):
    def __init__(self):
        super().__init__('cat_detector')
        
        self.model = YOLO('yolov8n.pt')
        self.model.to('cpu')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.detection_count = 0
        
        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/catrun',
            self.image_callback,
            10
        )
        
        # Publish cat position
        self.pub = self.create_publisher(
            PointStamped,
            '/cat_position',
            10
        )
        
        # Timer to print status every 3 seconds
        self.timer = self.create_timer(3.0, self.status_callback)
        
        self.get_logger().info('='*40)
        self.get_logger().info('Cat detector node started!')
        self.get_logger().info('Waiting for camera topic: /camera/catrun')
        self.get_logger().info('='*40)

    def status_callback(self):
        if self.frame_count == 0:
            self.get_logger().warn(
                '⚠️  No frames received yet! Check camera topic:'
                '\n  ros2 topic list | grep camera'
                '\n  ros2 topic hz /camera/catrun'
            )
        else:
            self.get_logger().info(
                f'📷 Status: frames={self.frame_count} | '
                f'detections={self.detection_count} | '
                f'cats_found={"YES 🐱" if self.detection_count > 0 else "NO"}'
            )

    def image_callback(self, msg):
        self.frame_count += 1
        
        # Confirm camera is working every 30 frames
        if self.frame_count == 1:
            self.get_logger().info(
                f'✅ Camera connected! '
                f'Resolution: {msg.width}x{msg.height}'
            )
        
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'📷 Receiving frames... ({self.frame_count} total)'
            )

        # Convert ROS image to OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'❌ cv_bridge failed: {e}')
            return

        # Run YOLOv8
        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            self.get_logger().error(f'❌ YOLO failed: {e}')
            return

        cat_found = False
        for result in results:
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                label = self.model.names[cls]
                
                # Log ALL detections so you can see what YOLO finds
                if conf > 0.3:
                    self.get_logger().info(
                        f'🔍 Detected: {label} (class {cls}) '
                        f'confidence={conf:.2f}'
                    )
                
                # Cat is class 15
                if cls == 15 and conf > 0.5:
                    cat_found = True
                    self.detection_count += 1
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    self.get_logger().info(
                        f'CAT DETECTED! '
                        f'center=({cx:.0f},{cy:.0f}) '
                        f'conf={conf:.2f}'
		    )

                    # Publish position
                    point = PointStamped()
                    point.header = msg.header
                    point.point.x = cx
                    point.point.y = cy
                    point.point.z = 0.0
                    self.pub.publish(point)

        if not cat_found and self.frame_count % 30 == 0:
            self.get_logger().info('No cat detected in current frame')

def main(args=None):
    rclpy.init(args=args)
    node = CatDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
