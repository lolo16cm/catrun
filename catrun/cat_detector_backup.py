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
        
        # Load YOLOv8 model
        self.model = YOLO('yolov8n.pt')  # nano = fastest on Jetson
        self.bridge = CvBridge()
        
        # Subscribe to camera
        self.sub = self.create_subscription(
            Image,
            '/camera/catrun',  # change to your camera topic
            self.image_callback,
            10
        )
        
        # Publish cat position
        self.pub = self.create_publisher(
            PointStamped,
            '/cat_position',
            10
        )
        
        self.get_logger().info('Cat detector started!')

    def image_callback(self, msg):
        # Convert ROS image to OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLOv8
        results = self.model(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # Class 15 = cat in COCO dataset
                if int(box.cls) == 15:
                    # Get bounding box center
                    x1, y1, x2, y2 = box.xyxy[0]
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)
                    conf = float(box.conf)
                    
                    self.get_logger().info(
                        f'Cat detected! center=({cx:.0f},{cy:.0f}) conf={conf:.2f}'
                    )
                    
                    # Publish position
                    point = PointStamped()
                    point.header = msg.header
                    point.point.x = cx
                    point.point.y = cy
                    point.point.z = 0.0
                    self.pub.publish(point)
                    
                    # Draw box on frame
                    cv2.rectangle(frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)),
                        (0, 255, 0), 2)
                    cv2.putText(frame, f'CAT {conf:.2f}',
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
        
        cv2.imshow('Cat Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = CatDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
