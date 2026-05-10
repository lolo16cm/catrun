#!/usr/bin/env python3
"""
cat_detector.py
Runs YOLOv8 + MobileNetV2 to identify cats (eevee / pichu / raichu).

Detection is triggered two ways:
  1. /camera_check_trigger  - from seek_cat when LiDAR sees a cat-sized moving object
  2. Every ALWAYS_ON_INTERVAL seconds - fallback for still/hidden cats

Publishes:
  /cat_identity   (std_msgs/String)             - confirmed cat name
  /cat_position   (geometry_msgs/PointStamped) - normalised [0,1] cx in frame
  /cat_detection/image (sensor_msgs/Image)      - annotated frame for web_stream

Multi-frame confirmation:
  A detection only triggers /cat_identity + /cat_position when the same
  cat name appears in CONFIRM_HITS of the last CONFIRM_WINDOW frames.
  This filters out glitched frames, motion-blurred frames, and momentary
  mis-classifications. Single-frame false positives can no longer trigger
  follow mode, and single-frame misses no longer break detection.
"""

import os
import sys
import time
import threading
from collections import deque

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

# Multi-frame confirmation
# Detection must appear in CONFIRM_HITS of the last CONFIRM_WINDOW frames
# before /cat_identity and /cat_position are published.
# Tuning:
#   - Increase CONFIRM_HITS to require more agreement (fewer false positives,
#     but slower response)
#   - Decrease to be more permissive (faster response, but more noise)
#   - CONFIRM_WINDOW should be at least 1 + CONFIRM_HITS
CONFIRM_WINDOW = 5    # how many recent frames to consider
CONFIRM_HITS   = 3    # how many of those frames must agree on the same cat

# How long an entry stays "fresh" in the rolling buffer. If detections come
# in sparsely (e.g. during a still pose), we don't want a hit from 30 seconds
# ago contributing to today's vote.
ENTRY_TTL_SEC  = 2.0

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
        self.confirmed_count = 0   # how many times we've published a confirmed ID

        # Multi-frame confirmation buffer.
        # Each entry: dict with keys {ts, name, cx, cy, conf}
        # name == None means "this frame had no cat detection"
        self.detection_history = deque(maxlen=CONFIRM_WINDOW)

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
        self.get_logger().info(f'  Confirm : {CONFIRM_HITS} of last {CONFIRM_WINDOW} frames '
                               f'(TTL {ENTRY_TTL_SEC}s)')
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
                self.detection_history.clear()  # reset confirmation buffer
                self.get_logger().info(f'Target cat: {self.target_cat}')
                return
        self.target_cat = None
        self.detection_history.clear()
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
        detected_this_frame = None  # what (if anything) did we detect
        cx_norm = cy_norm = 0.0
        id_conf = 0.0

        try:
            results = self.yolo(frame, verbose=False, stream=True, imgsz=320)
        except Exception as e:
            self.get_logger().error(f'YOLO error: {e}')
            return

        # Pick the SINGLE best cat detection in this frame (by confidence).
        # Previously the code looped over all boxes and immediately published
        # for each — meaning two cats in frame would publish twice for one
        # frame. We want one detection per frame for the rolling buffer.
        best_box = None
        best_conf = 0.0
        for result in results:
            for box in result.boxes:
                cls  = int(box.cls)
                conf = float(box.conf)
                if cls != YOLO_CAT_CLASS_ID or conf < YOLO_CONF_THRESH:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_box  = box

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)

            if self.classifier is not None:
                cat_name, id_conf = self._classify(frame[y1:y2, x1:x2])
            else:
                cat_name, id_conf = 'cat', best_conf

            cx_norm = float((x1 + x2) / 2) / w
            cy_norm = float((y1 + y2) / 2) / h

            # Annotate bounding box
            color = (0, 255, 100)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{cat_name} {id_conf:.2f}'
            cv2.putText(annotated, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detected_this_frame = cat_name

        # ── Append this frame's result to the rolling history ──────────────
        now = time.time()
        self.detection_history.append({
            'ts':   now,
            'name': detected_this_frame,   # may be None
            'cx':   cx_norm,
            'cy':   cy_norm,
            'conf': id_conf,
        })

        # ── Multi-frame confirmation: only publish if N of last M agree ────
        # Drop expired entries from voting (TTL)
        fresh = [e for e in self.detection_history
                 if (now - e['ts']) <= ENTRY_TTL_SEC]

        # Count hits per name, ignoring None entries
        name_counts = {}
        for e in fresh:
            if e['name'] is None:
                continue
            name_counts[e['name']] = name_counts.get(e['name'], 0) + 1

        confirmed_name = None
        for name, count in name_counts.items():
            if count >= CONFIRM_HITS:
                confirmed_name = name
                break

        # Filter by target if one is set
        if self.target_cat and confirmed_name != self.target_cat:
            # Either nothing confirmed, or confirmed but not the target
            if detected_this_frame and detected_this_frame != self.target_cat:
                # Annotate non-target detections in grey (still useful to see)
                pass  # already drawn above in green; downgrade color:
                # (we'll just leave them green for visibility - it's fine)

            # Add a "voting" overlay so user sees what's happening
            self._add_vote_overlay(annotated, name_counts, fresh)

            # publish annotated frame (no identity/position)
            self._publish_annotated(annotated)
            return

        if confirmed_name is None:
            # Nothing reached the confirmation threshold
            self._add_vote_overlay(annotated, name_counts, fresh)
            self._publish_annotated(annotated)
            return

        # ── Confirmed! Publish identity + position ─────────────────────────
        # Use the average position from the fresh hits of the confirmed name
        hits = [e for e in fresh if e['name'] == confirmed_name]
        avg_cx   = sum(e['cx']   for e in hits) / len(hits)
        avg_cy   = sum(e['cy']   for e in hits) / len(hits)
        avg_conf = sum(e['conf'] for e in hits) / len(hits)

        self.confirmed_count += 1
        self.detection_count += 1

        self.get_logger().info(
            f'CONFIRMED: {confirmed_name} '
            f'({len(hits)}/{len(fresh)} frames) '
            f'avg_conf={avg_conf:.2f} cx={avg_cx:.2f}')

        # Publish identity
        self.pub_identity.publish(String(data=confirmed_name))

        # Publish position
        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        pt.header.frame_id = 'camera'
        pt.point.x = avg_cx
        pt.point.y = avg_cy
        pt.point.z = 0.0
        self.pub_position.publish(pt)

        # Add big confirmation overlay
        cv2.putText(annotated,
                    f'CONFIRMED {confirmed_name} ({len(hits)}/{len(fresh)})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._publish_annotated(annotated)

    def _add_vote_overlay(self, img, name_counts, fresh):
        """Overlay current voting status on the annotated image."""
        y = 30
        cv2.putText(img, f'frames in window: {len(fresh)}/{CONFIRM_WINDOW}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        if name_counts:
            for name, count in name_counts.items():
                color = (0, 255, 0) if count >= CONFIRM_HITS else (0, 200, 200)
                cv2.putText(img,
                            f'  {name}: {count}/{CONFIRM_HITS} needed',
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
        else:
            cv2.putText(img, '  (no detections in window)',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (150, 150, 150), 1)

    def _publish_annotated(self, annotated):
        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_annotated.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Annotated publish error: {e}')

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
                f'confirmed={self.confirmed_count} | '
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
