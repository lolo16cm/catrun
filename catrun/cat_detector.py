#!/usr/bin/env python3
"""
cat_detector.py
Runs YOLOv8 + MobileNetV2 to identify cats (eevee / pichu / raichu).

Detection is triggered two ways:
  1. /camera_check_trigger  - from seek_cat when LiDAR sees a cat-sized moving object
  2. Every ALWAYS_ON_INTERVAL seconds - fallback for still/hidden cats

Publishes:
  /cat_identity   (std_msgs/String)       - confirmed cat name
  /cat_position   (geometry_msgs/PointStamped) - normalised [0,1] cx in frame
  /cat_detection/image (sensor_msgs/Image) - annotated frame for web_stream
"""

import os
import sys
import time
import threading

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image as PILImage
from torch import nn
from torchvision import models
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

# config
YOLO_MODEL_PATH    = 'yolov8n.pt'
MOBILENET_MODEL    = os.path.expanduser('~/cat_ws/models/cat_classifier.pth')
CAT_CLASSES        = ['eevee', 'pichu', 'raichu']
YOLO_CAT_CLASS_ID  = 15          # COCO class id for 'cat'
YOLO_CONF_THRESH   = 0.5
MN_CONF_THRESH     = 0.70
ALWAYS_ON_INTERVAL = 0.1

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


class CatDetector(Node):

    def __init__(self):
        super().__init__('cat_detector')

        # models
        self.get_logger().info('Loading YOLOv8...')
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.yolo.to('cpu')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier = self._load_classifier()

        # state
        self.bridge          = CvBridge()
        self.target_cat      = None
        self.latest_frame    = None
        self.frame_lock      = threading.Lock()
        self.triggered       = False
        self.last_run        = 0.0
        self.frame_count     = 0
        self.detection_count = 0

        # subscribers
        self.create_subscription(Image,  '/camera/catrun',          self.image_cb,   10)
        self.create_subscription(String, '/cat_target',             self.target_cb,  10)
        self.create_subscription(String, '/camera_check_trigger',   self.trigger_cb, 10)

        # publishers
        self.pub_position  = self.create_publisher(PointStamped, '/cat_position',        10)
        self.pub_identity  = self.create_publisher(String,       '/cat_identity',         10)
        self.pub_annotated = self.create_publisher(Image,        '/cat_detection/image',  10)

        # timers
        self.create_timer(0.05, self.detection_loop)
        self.create_timer(3.0,  self.status_cb)

        self.get_logger().info('=' * 45)
        self.get_logger().info('CatDetector ready!')
        self.get_logger().info(f'  Classes : {CAT_CLASSES}')
        self.get_logger().info(f'  Device  : {self.device}')
        self.get_logger().info(f'  Trigger : /camera_check_trigger')
        self.get_logger().info('=' * 45)

    def _load_classifier(self):
        if not os.path.exists(MOBILENET_MODEL):
            self.get_logger().warn(
                f'cat_classifier.pth not found at {MOBILENET_MODEL}\n'
                '  YOLO-only mode (no identity recognition).\n'
                '  Run train_classifier.py to create the model.')
            return None

        self.get_logger().info(f'Loading MobileNetV2 from {MOBILENET_MODEL}')
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, len(CAT_CLASSES))
        m.load_state_dict(torch.load(MOBILENET_MODEL, map_location=self.device))
        m.to(self.device)
        m.eval()
        self.get_logger().info(f'MobileNetV2 ready! Classes: {CAT_CLASSES}')
        return m

    # callbacks
    def target_cb(self, msg: String):
        text = msg.data.strip().lower()
        for name in CAT_CLASSES:
            if name in text:
                self.target_cat = name
                self.get_logger().info(f'Target cat: {self.target_cat}')
                return
        self.target_cat = None
        self.get_logger().info('Target cleared.')

    def trigger_cb(self, msg: String):
        self.get_logger().debug(
            f'[trigger] Camera check requested for: {msg.data}')
        self.triggered = True

    def image_cb(self, msg: Image):
        self.frame_count += 1
        if self.frame_count == 1:
            self.get_logger().info(
                f'Camera connected! {msg.width}x{msg.height}')
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')

    # detection loop
    def detection_loop(self):
        now = time.time()
        should_run = (
            self.triggered or
            (now - self.last_run) >= ALWAYS_ON_INTERVAL
        )
        if not should_run:
            return

        with self.frame_lock:
            frame = self.latest_frame

        if frame is None:
            return

        self.triggered = False
        self.last_run  = now
        self._run_detection(frame)

    def _run_detection(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        cat_found = False

        try:
            results = self.yolo(frame, verbose=False, stream=True, imgsz=320)
        except Exception as e:
            self.get_logger().error(f'YOLO error: {e}')
            return

        for result in results:
            for box in result.boxes:
                cls  = int(box.cls)
                conf = float(box.conf)

                if cls != YOLO_CAT_CLASS_ID or conf < YOLO_CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(w, x2); y2 = min(h, y2)

                if self.classifier is not None:
                    cat_name, id_conf = self._classify(frame[y1:y2, x1:x2])
                else:
                    cat_name, id_conf = 'cat', conf

                if self.target_cat and cat_name != self.target_cat:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (100, 100, 100), 2)
                    cv2.putText(annotated, f'{cat_name} (not target)',
                                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (100, 100, 100), 1)
                    continue

                cat_found = True
                self.detection_count += 1
                cx_norm = float((x1 + x2) / 2) / w
                cy_norm = float((y1 + y2) / 2) / h

                self.get_logger().info(
                    f'CAT FOUND: {cat_name} conf={id_conf:.2f} '
                    f'cx={cx_norm:.2f} cy={cy_norm:.2f}')

                self.pub_identity.publish(String(data=cat_name))

                pt = PointStamped()
                pt.header.stamp    = self.get_clock().now().to_msg()
                pt.header.frame_id = 'camera'
                pt.point.x = cx_norm
                pt.point.y = cy_norm
                pt.point.z = 0.0
                self.pub_position.publish(pt)

                color = (0, 255, 100)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f'{cat_name} {id_conf:.2f}'
                cv2.putText(annotated, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_annotated.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Annotated publish error: {e}')

        if not cat_found:
            self.get_logger().debug('No target cat in frame.')

    def _classify(self, crop_bgr: np.ndarray):
        if crop_bgr.size == 0:
            return 'unknown', 0.0

        pil = PILImage.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        t   = TRANSFORM(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out   = self.classifier(t)
            probs = torch.softmax(out, dim=1)[0]
            idx   = int(probs.argmax())
            conf  = float(probs[idx])

        if conf < MN_CONF_THRESH:
            return 'unknown', conf

        return CAT_CLASSES[idx], conf

    def status_cb(self):
        if self.frame_count == 0:
            self.get_logger().warn(
                'No frames yet! Check: ros2 topic hz /camera/catrun')
        else:
            self.get_logger().info(
                f'frames={self.frame_count} | '
                f'detections={self.detection_count} | '
                f'target={self.target_cat or "any"} | '
                f'classifier={"ON" if self.classifier else "OFF (YOLO only)"}')


def _safe_cleanup(node):
    """Release ML models, OpenCV windows, and CUDA memory."""
    if node is not None:
        try:
            if hasattr(node, 'classifier') and node.classifier is not None:
                del node.classifier
                node.classifier = None
        except Exception:
            pass
        try:
            if hasattr(node, 'yolo') and node.yolo is not None:
                del node.yolo
                node.yolo = None
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = CatDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[cat_detector] error: {e}", file=sys.stderr)
    finally:
        _safe_cleanup(node)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[cat_detector] clean shutdown complete")


if __name__ == '__main__':
    main()
