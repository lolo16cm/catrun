#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torchvision.transforms as T
from torchvision import models
from torch import nn
import os

# ── config ───────────────────────────────────────────────────────────────────
YOLO_MODEL_PATH   = 'yolov8n.pt'
MOBILENET_MODEL   = os.path.expanduser('~/cat_ws/models/cat_classifier.pth')
CAT_CLASSES       = ['eevee', 'pichu', 'raichu']
YOLO_CAT_CLASS_ID = 15
YOLO_CONF_THRESH  = 0.5
MN_CONF_THRESH    = 0.70
# ─────────────────────────────────────────────────────────────────────────────

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


class CatDetector(Node):

    def __init__(self):
        super().__init__('cat_detector')

        # YOLOv8
        self.get_logger().info('Loading YOLOv8...')
        self.model = YOLO(YOLO_MODEL_PATH)
        self.model.to('cpu')

        # MobileNetV2
        self.classifier = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if os.path.exists(MOBILENET_MODEL):
            self.get_logger().info(f'Loading MobileNetV2 from {MOBILENET_MODEL}')
            m = models.mobilenet_v2(weights=None)
            m.classifier[1] = nn.Linear(1280, len(CAT_CLASSES))
            m.load_state_dict(torch.load(MOBILENET_MODEL, map_location=self.device))
            m.to(self.device)
            m.eval()
            self.classifier = m
            self.get_logger().info(f'MobileNetV2 ready! Classes: {CAT_CLASSES}')
        else:
            self.get_logger().warn(
                f'cat_classifier.pth not found at {MOBILENET_MODEL}\n'
                'Copy it there to enable identity recognition.\n'
                'Running YOLO-only for now.')

        self.bridge = CvBridge()
        self.target_cat = None
        self.frame_count = 0
        self.detection_count = 0

        # Subscribers
        self.create_subscription(Image,  '/camera/catrun', self.image_callback, 10)
        self.create_subscription(String, '/cat_target',    self.target_callback, 10)

        # Publishers
        self.pub          = self.create_publisher(PointStamped, '/cat_position', 10)
        self.pub_identity = self.create_publisher(String,       '/cat_identity', 10)

        self.create_timer(3.0, self.status_callback)

        self.get_logger().info('='*40)
        self.get_logger().info('Cat detector node started!')
        self.get_logger().info('Waiting for camera topic: /camera/catrun')
        self.get_logger().info('='*40)

    def status_callback(self):
        if self.frame_count == 0:
            self.get_logger().warn(
                '⚠️  No frames received yet! Check camera topic:'
                '\n  ros2 topic list | grep camera'
                '\n  ros2 topic hz /camera/catrun')
        else:
            self.get_logger().info(
                f'📷 Status: frames={self.frame_count} | '
                f'detections={self.detection_count} | '
                f'target={self.target_cat or "none"} | '
                f'cats_found={"YES 🐱" if self.detection_count > 0 else "NO"}')

    def target_callback(self, msg: String):
        text = msg.data.strip().lower()
        for name in CAT_CLASSES:
            if name in text:
                self.target_cat = name
                self.get_logger().info(f'🎯 Target cat set to: {self.target_cat}')
                return
        self.target_cat = None
        self.get_logger().info('Target cleared.')

    def image_callback(self, msg):
        self.frame_count += 1

        if self.frame_count == 1:
            self.get_logger().info(
                f'✅ Camera connected! Resolution: {msg.width}x{msg.height}')

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'📷 Receiving frames... ({self.frame_count} total)')

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'❌ cv_bridge failed: {e}')
            return

        h, w = frame.shape[:2]

        try:
            results = self.model(frame, verbose=False)
        except Exception as e:
            self.get_logger().error(f'❌ YOLO failed: {e}')
            return

        cat_found = False
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls)
                conf = float(box.conf)
                label = self.model.names[cls]

                if conf > 0.3:
                    self.get_logger().info(
                        f'🔍 Detected: {label} (class {cls}) confidence={conf:.2f}')

                if cls != YOLO_CAT_CLASS_ID or conf < YOLO_CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Stage 2: MobileNetV2 identity
                if self.classifier is not None:
                    cat_name, id_conf = self._classify(frame[y1:y2, x1:x2])
                else:
                    cat_name, id_conf = 'cat', conf

                cat_found = True
                self.detection_count += 1
                cx = float((x1 + x2) / 2)
                cy = float((y1 + y2) / 2)

                self.get_logger().info(
                    f'🐱 CAT DETECTED! name={cat_name} '
                    f'conf={id_conf:.2f} center=({cx:.0f},{cy:.0f})')

                self.pub_identity.publish(String(data=cat_name))

                point = PointStamped()
                point.header = msg.header
                point.point.x = cx
                point.point.y = cy
                point.point.z = 0.0
                self.pub.publish(point)

        if not cat_found and self.frame_count % 30 == 0:
            self.get_logger().info('No cat detected in current frame')

    def _classify(self, crop_bgr: np.ndarray):
        if crop_bgr.size == 0:
            return 'unknown', 0.0

        pil_img = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        tensor  = TRANSFORM(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out   = self.classifier(tensor)
            probs = torch.softmax(out, dim=1)[0]
            idx   = int(probs.argmax())
            conf  = float(probs[idx])

        if conf < MN_CONF_THRESH:
            return 'unknown', conf

        return CAT_CLASSES[idx], conf


def main(args=None):
    rclpy.init(args=args)
    node = CatDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()