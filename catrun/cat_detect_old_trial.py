#!/usr/bin/env python3
"""
cat_detection.py
Stage 1 — YOLOv8 detects cats (class 15)
Stage 2 — MobileNetV2 (PyTorch) identifies which cat: eevee / pichu / raichu
"""

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
CAT_CLASSES       = ['eevee', 'pichu', 'raichu']   # must match training order
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


class CatDetectionNode(Node):

    def __init__(self):
        super().__init__('cat_detection')

        # ── YOLOv8 ───────────────────────────────────────────────────────────
        self.get_logger().info('Loading YOLOv8...')
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.yolo.to('cpu')

        # ── MobileNetV2 ──────────────────────────────────────────────────────
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
            self.get_logger().info(f'MobileNetV2 ready on {self.device}')
        else:
            self.get_logger().warn(
                f'cat_classifier.pth not found at {MOBILENET_MODEL}. '
                'Copy it there to enable identity. Running YOLO-only for now.')

        self.bridge     = CvBridge()
        self.target_cat = None
        self.frame_count     = 0
        self.detection_count = 0

        # ── subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Image,  '/camera/catrun', self.image_cb, 10)
        self.create_subscription(String, '/cat_target',    self.target_cb, 10)

        # ── publishers ───────────────────────────────────────────────────────
        self.pub_position = self.create_publisher(PointStamped, '/cat_position', 10)
        self.pub_identity = self.create_publisher(String,       '/cat_identity', 10)
        self.pub_img      = self.create_publisher(Image,        '/cat_detection/image', 10)

        # ── status timer ─────────────────────────────────────────────────────
        self.create_timer(3.0, self.status_cb)

        self.get_logger().info('=' * 40)
        self.get_logger().info('Cat Detection Node started!')
        self.get_logger().info(f'Classes: {CAT_CLASSES}')
        self.get_logger().info('Waiting for /camera/catrun ...')
        self.get_logger().info('=' * 40)

    # ── status timer ─────────────────────────────────────────────────────────

    def status_cb(self):
        if self.frame_count == 0:
            self.get_logger().warn(
                'No frames yet! Check:\n'
                '  ros2 topic list | grep camera\n'
                '  ros2 topic hz /camera/catrun')
        else:
            self.get_logger().info(
                f'frames={self.frame_count} | '
                f'detections={self.detection_count} | '
                f'target={self.target_cat or "none"} | '
                f'cats_found={"YES" if self.detection_count > 0 else "NO"}')

    # ── target callback ───────────────────────────────────────────────────────

    def target_cb(self, msg: String):
        text = msg.data.strip().lower()
        for name in CAT_CLASSES:
            if name in text:
                self.target_cat = name
                self.get_logger().info(f'Target cat → {self.target_cat}')
                return
        self.target_cat = None
        self.get_logger().info('Target cleared.')

    # ── main image callback ───────────────────────────────────────────────────

    def image_cb(self, msg: Image):
        self.frame_count += 1

        if self.frame_count == 1:
            self.get_logger().info(
                f'Camera connected! {msg.width}x{msg.height}')

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge: {e}')
            return

        h, w      = frame.shape[:2]
        annotated = frame.copy()
        best_box  = None

        try:
            results = self.yolo(frame, verbose=False)[0]
        except Exception as e:
            self.get_logger().error(f'YOLO: {e}')
            return

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = self.yolo.names[cls_id]

            # log all detections above 0.3 so you can see what YOLO finds
            if conf > 0.3:
                self.get_logger().info(
                    f'Detected: {label} (cls {cls_id}) conf={conf:.2f}')

            if cls_id != YOLO_CAT_CLASS_ID or conf < YOLO_CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Stage 2: identity
            if self.classifier is not None:
                cat_name, id_conf = self._classify(frame[y1:y2, x1:x2])
            else:
                cat_name, id_conf = 'cat', conf

            self.detection_count += 1
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            self.get_logger().info(
                f'CAT DETECTED! name={cat_name} conf={id_conf:.2f} '
                f'center=({cx:.0f},{cy:.0f})')

            self.pub_identity.publish(String(data=cat_name))

            is_target = (self.target_cat is not None and cat_name == self.target_cat)
            color = (0, 255, 0) if is_target else (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f'{cat_name} {id_conf:.2f}',
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if best_box is None or is_target:
                best_box = (cx, cy, id_conf, msg.header)

        # publish position of best detection
        if best_box:
            cx, cy, id_conf, header = best_box
            pt = PointStamped()
            pt.header = header
            pt.point.x = cx
            pt.point.y = cy
            pt.point.z = 0.0
            self.pub_position.publish(pt)
        elif self.frame_count % 30 == 0:
            self.get_logger().info('No cat in current frame')

        # publish annotated image
        try:
            self.pub_img.publish(
                self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
        except Exception as e:
            self.get_logger().error(f'img publish: {e}')

    # ── MobileNetV2 classifier ────────────────────────────────────────────────

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


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = CatDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()