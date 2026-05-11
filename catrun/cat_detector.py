#!/usr/bin/env python3
"""
cat_detector.py
Runs YOLOv8 + MobileNetV2 to identify cats (eevee / pichu / raichu).

Publishes:
  /cat_identity        - confirmed cat name (multi-frame voted)
  /cat_position        - x = normalised cx [0,1]
                         y = normalised cy [0,1]
                         z = ESTIMATED DISTANCE in meters (from bbox width)
  /cat_detection/image - annotated frame

The same detector serves BOTH watch and play modes; only the camera
feeding /camera/catrun changes (front CSI in watch, rear USB in play).
The `mode` parameter is purely cosmetic - it tags the logs so it's
obvious which mode the detector is currently serving.

Distance estimation:
  No depth sensor available, so we use the same trick the ball follower
  reference uses: a known-width object's bbox-pixel-width is inversely
  proportional to its distance.
    distance = (real_width_m * focal_px) / bbox_width_px
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

# ─── config ──────────────────────────────────────────────────────────────
YOLO_MODEL_PATH    = 'yolov8n.pt'
MOBILENET_MODEL    = os.path.expanduser('~/cat_ws/models/cat_classifier.pth')
CAT_CLASSES        = ['eevee', 'pichu', 'raichu']
YOLO_CAT_CLASS_ID  = 15
YOLO_CONF_THRESH   = 0.5
MN_CONF_THRESH     = 0.70
# YOLO inference resolution. Smaller = much faster on Jetson CPU.
# At 192 a cat plushie is still ~30-60 px wide so detection is reliable.
YOLO_IMGSZ         = 192
# Time between YOLO runs when no trigger is pending. Lower = snappier
# detection but more CPU. 0.25 = 4 Hz which is fine for cats and lets
# the Jetson CPU breathe.
ALWAYS_ON_INTERVAL = 0.25

# Multi-frame confirmation
CONFIRM_WINDOW = 5
CONFIRM_HITS   = 3
ENTRY_TTL_SEC  = 2.0

# Distance estimation from bounding box width
# distance = (real_width * focal_px) / bbox_width_px
#
# CALIBRATION:
#   1. Place the cat plushie EXACTLY 1 meter from the camera
#   2. Watch the annotated stream - note the bbox width shown as "bbox=NNN"
#   3. Set focal_px (ROS param) = bbox_width_at_1m / CAT_REAL_WIDTH_M
#
# NOTE: The front CSI cam (1280x720) and the rear USB cam (640x480) almost
# certainly have different focal-px values. Calibrate each one and pass
# the right value via the launch file's `focal_px` parameter.
CAT_REAL_WIDTH_M = 0.25      # ~25 cm wide plushie - measure your actual one
CAMERA_FOCAL_PX  = 600.0     # default; override with ROS param `focal_px`
DIST_MIN_M       = 0.10
DIST_MAX_M       = 5.00

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


def estimate_distance_m(bbox_width_px, focal_px):
    if bbox_width_px <= 1:
        return None
    d = (CAT_REAL_WIDTH_M * float(focal_px)) / float(bbox_width_px)
    return max(DIST_MIN_M, min(DIST_MAX_M, d))


class CatDetector(Node):

    def __init__(self):
        super().__init__('cat_detector')

        # ─── parameters ───────────────────────────────────────────────
        # `mode` is just a label used in logs so you can tell which
        # behavior is currently consuming the feed (watch vs play).
        self.declare_parameter('mode', 'watch')
        self.declare_parameter('focal_px', CAMERA_FOCAL_PX)

        self.mode     = str(self.get_parameter('mode').value).lower().strip()
        self.focal_px = float(self.get_parameter('focal_px').value)
        if self.mode not in ('watch', 'play'):
            self.get_logger().warn(
                f"Unknown mode '{self.mode}', defaulting to 'watch'")
            self.mode = 'watch'

        self.get_logger().info('Loading YOLOv8...')
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.yolo.to('cpu')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Play mode only needs YOLO ("is it a cat?") - skip MobileNetV2
        # entirely to save CPU/RAM on the Orin Nano. Watch mode still
        # needs the classifier to tell eevee/pichu/raichu apart.
        if self.mode == 'play':
            self.get_logger().info(
                '[play] Skipping MobileNetV2 - YOLO-only detection')
            self.classifier = None
        else:
            self.classifier = self._load_classifier()

        self.bridge          = CvBridge()
        self.target_cat      = None
        self.latest_frame    = None
        self.frame_lock      = threading.Lock()
        self.triggered       = False
        self.last_run        = 0.0
        self.frame_count     = 0
        self.detection_count = 0
        self.confirmed_count = 0

        # Multi-frame confirmation buffer.
        # Each entry: dict {ts, name, cx, cy, dist, bbox_w, conf}
        self.detection_history = deque(maxlen=CONFIRM_WINDOW)

        # ─── Dual-camera support for play mode ────────────────────────
        # In play mode the launch file starts TWO camera nodes:
        #   /camera/front  - front CSI (csi0)  - default in watch & most play states
        #   /camera/rear   - rear USB  (usb1)  - used during FLEE_RUN to watch pursuer
        # In watch mode only /camera/catrun is published (legacy topic),
        # and we subscribe to it too. We process frames from the camera
        # that matches the CURRENT robot state.
        # The 'active' camera is selected based on the latest /seek_status:
        #   'fleeing' / 'checking'  -> rear (watch the threat behind)
        #   anything else            -> front
        self.latest_status      = ''
        self.latest_front_frame = None    # last image from /camera/front
        self.latest_rear_frame  = None    # last image from /camera/rear
        self.latest_legacy_frame = None   # last image from /camera/catrun (watch mode)

        # The "active" feed is the one we actually run YOLO on AND
        # mirror to /camera/catrun for the web UI.
        self.create_subscription(Image,  '/camera/front',           self._front_cb,   10)
        self.create_subscription(Image,  '/camera/rear',            self._rear_cb,    10)
        self.create_subscription(Image,  '/camera/catrun',          self._legacy_cb,  10)
        self.create_subscription(String, '/seek_status',            self._status_cb,  10)
        self.create_subscription(String, '/cat_target',             self.target_cb,  10)
        self.create_subscription(String, '/camera_check_trigger',   self.trigger_cb, 10)

        # Mirror the active camera onto /camera/catrun so web_stream
        # (and any other legacy subscriber) keeps working. We only
        # mirror in play mode; in watch mode /camera/catrun is already
        # published directly by camera_node so we skip mirroring.
        self.pub_catrun_mirror = self.create_publisher(Image, '/camera/catrun', 10)

        self.pub_position  = self.create_publisher(PointStamped, '/cat_position',        10)
        self.pub_identity  = self.create_publisher(String,       '/cat_identity',         10)
        self.pub_annotated = self.create_publisher(Image,        '/cat_detection/image',  10)

        self.create_timer(0.05, self.detection_loop)
        self.create_timer(3.0,  self.status_cb)

        self.get_logger().info('=' * 50)
        self.get_logger().info(f'CatDetector ready! [mode={self.mode}]')
        self.get_logger().info(f'  Classes  : {CAT_CLASSES}')
        self.get_logger().info(f'  Device   : {self.device}')
        self.get_logger().info(f'  Confirm  : {CONFIRM_HITS} of {CONFIRM_WINDOW}')
        self.get_logger().info(f'  Distance : real_w={CAT_REAL_WIDTH_M}m '
                               f'focal={self.focal_px}px')
        self.get_logger().info('=' * 50)

    def _load_classifier(self):
        if not os.path.exists(MOBILENET_MODEL):
            self.get_logger().warn(
                f'cat_classifier.pth not found at {MOBILENET_MODEL} - YOLO-only mode')
            return None

        self.get_logger().info(f'Loading MobileNetV2 from {MOBILENET_MODEL}')
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, len(CAT_CLASSES))
        m.load_state_dict(torch.load(MOBILENET_MODEL, map_location=self.device))
        m.to(self.device)
        m.eval()
        self.get_logger().info(f'MobileNetV2 ready! Classes: {CAT_CLASSES}')
        return m

    def target_cb(self, msg: String):
        text = msg.data.strip().lower()

        # Play mode: classifier disabled - we don't care WHICH cat the
        # user named, any cat triggers flee. Just remember we're armed.
        if self.mode == 'play':
            if text in ('', 'stop', 'none'):
                self.target_cat = None
                self.detection_history.clear()
                self.get_logger().info('[play] Disarmed.')
            else:
                self.target_cat = 'any'
                self.detection_history.clear()
                self.get_logger().info('[play] Armed - any cat triggers flee.')
            return

        # Watch mode: identify the specific cat the user named.
        for name in CAT_CLASSES:
            if name in text:
                self.target_cat = name
                self.detection_history.clear()
                self.get_logger().info(
                    f'[{self.mode}] Target cat: {self.target_cat}')
                return
        self.target_cat = None
        self.detection_history.clear()
        self.get_logger().info(f'[{self.mode}] Target cleared.')

    def trigger_cb(self, msg: String):
        self.triggered = True

    # ─── Camera feed callbacks ────────────────────────────────────────
    # Three subscriptions:
    #   /camera/front  - dual-cam play mode: front CSI
    #   /camera/rear   - dual-cam play mode: rear USB
    #   /camera/catrun - single-cam watch mode (legacy)
    # Each callback stores its frame; the active-feed selector decides
    # which one to feed to YOLO and mirror to /camera/catrun.

    def _status_cb(self, msg: String):
        self.latest_status = msg.data.strip().lower()

    def _active_feed_name(self):
        """Pick which feed YOLO should process this tick. Default is
        front. Switch to rear only when the robot is in one of these
        states (set by flee_behavior via /seek_status):
          - 'check_rear'  : LiDAR saw motion behind, verify with camera
          - 'fleeing'     : running away forward, watch the pursuer
          - 'checking'    : just stopped after a flee, scan behind first
        Otherwise (ambushing, seeking, idle, etc.) -> front.
        """
        rear_states = ('check_rear', 'fleeing', 'checking')
        want_rear = self.latest_status in rear_states

        if want_rear and self.latest_rear_frame is not None:
            return 'rear'
        if not want_rear and self.latest_front_frame is not None:
            return 'front'

        # Fallback to whatever we have
        if self.latest_front_frame is not None:
            return 'front'
        if self.latest_rear_frame is not None:
            return 'rear'
        return 'legacy'

    def _front_cb(self, msg: Image):
        self.frame_count += 1
        if self.frame_count == 1:
            self.get_logger().info(
                f'[{self.mode}] FRONT camera connected! '
                f'{msg.width}x{msg.height}')
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_front_frame = frame
                if self._active_feed_name() == 'front':
                    self.latest_frame = frame
            # Mirror to /camera/catrun if this is the active feed
            if self._active_feed_name() == 'front':
                self.pub_catrun_mirror.publish(msg)
        except Exception as e:
            self.get_logger().error(f'cv_bridge error (front): {e}')

    def _rear_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_rear_frame = frame
                if self._active_feed_name() == 'rear':
                    self.latest_frame = frame
            if self._active_feed_name() == 'rear':
                self.pub_catrun_mirror.publish(msg)
        except Exception as e:
            self.get_logger().error(f'cv_bridge error (rear): {e}')

    def _legacy_cb(self, msg: Image):
        """Watch mode (and any other single-camera config) publishes
        directly to /camera/catrun. Process that too, but never mirror
        (the publisher IS the source, so we'd loop)."""
        # Don't process if we're already receiving a front/rear stream
        # (avoids double-processing in dual-cam mode).
        if (self.latest_front_frame is not None
                or self.latest_rear_frame is not None):
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.frame_lock:
                self.latest_legacy_frame = frame
                self.latest_frame = frame
            if self.frame_count == 0:
                self.get_logger().info(
                    f'[{self.mode}] Legacy camera connected! '
                    f'{msg.width}x{msg.height}')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f'cv_bridge error (legacy): {e}')

    def detection_loop(self):
        now = time.time()
        if not (self.triggered or (now - self.last_run) >= ALWAYS_ON_INTERVAL):
            return
        self.triggered = False
        self.last_run  = now

        # Only process the currently-active feed. flee_behavior decides
        # which feed is active via the /seek_status topic:
        #   front: ambushing, seeking, idle (default)
        #   rear : check_rear, fleeing, checking
        active = self._active_feed_name()
        with self.frame_lock:
            if active == 'rear':
                frame = self.latest_rear_frame
            elif active == 'legacy':
                frame = self.latest_legacy_frame
            else:
                frame = self.latest_front_frame

        if frame is None:
            return

        self._run_detection(frame, camera=active)

    def _run_detection(self, frame: np.ndarray, camera: str = 'front'):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        detected_this_frame = None
        cx_norm = cy_norm = 0.0
        dist_m  = 0.0
        id_conf = 0.0
        bbox_w  = 0

        try:
            results = self.yolo(frame, verbose=False, stream=True, imgsz=YOLO_IMGSZ)
        except Exception as e:
            self.get_logger().error(f'YOLO error: {e}')
            return

        # Pick the SINGLE best cat detection (highest confidence)
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
            bbox_w  = x2 - x1
            dist_m  = estimate_distance_m(bbox_w, self.focal_px) or 0.0

            color = (0, 255, 100)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{cat_name} {id_conf:.2f} ~{dist_m:.2f}m bbox={bbox_w}px'
            cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detected_this_frame = cat_name

        # Append to rolling history
        now = time.time()
        self.detection_history.append({
            'ts':     now,
            'name':   detected_this_frame,
            'cx':     cx_norm,
            'cy':     cy_norm,
            'dist':   dist_m,
            'bbox_w': bbox_w,
            'conf':   id_conf,
        })

        # Multi-frame confirmation
        fresh = [e for e in self.detection_history
                 if (now - e['ts']) <= ENTRY_TTL_SEC]

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

        # Filter by target if set. The special value 'any' (used in play
        # mode) accepts any confirmed cat - we don't care which one.
        if (self.target_cat
                and self.target_cat != 'any'
                and confirmed_name != self.target_cat):
            self._add_vote_overlay(annotated, name_counts, fresh)
            self._publish_annotated(annotated)
            return

        if confirmed_name is None:
            self._add_vote_overlay(annotated, name_counts, fresh)
            self._publish_annotated(annotated)
            return

        # Confirmed - publish averaged position + distance
        hits = [e for e in fresh if e['name'] == confirmed_name]
        avg_cx   = sum(e['cx']   for e in hits) / len(hits)
        avg_cy   = sum(e['cy']   for e in hits) / len(hits)
        # Average bbox width then re-estimate distance (more stable than
        # averaging distances directly because distance is 1/x in width)
        avg_bbox = sum(e['bbox_w'] for e in hits) / len(hits)
        avg_dist = estimate_distance_m(avg_bbox, self.focal_px) or 0.0
        avg_conf = sum(e['conf'] for e in hits) / len(hits)

        self.confirmed_count += 1
        self.detection_count += 1

        self.get_logger().info(
            f'[{self.mode}] CONFIRMED: {confirmed_name} '
            f'({len(hits)}/{len(fresh)}) '
            f'cx={avg_cx:.2f} dist~{avg_dist:.2f}m bbox={avg_bbox:.0f}px '
            f'conf={avg_conf:.2f}')

        self.pub_identity.publish(String(data=confirmed_name))

        pt = PointStamped()
        pt.header.stamp    = self.get_clock().now().to_msg()
        # Encode the camera that saw the cat in frame_id. flee_behavior
        # uses this to know whether the cat is in front or behind.
        pt.header.frame_id = f'camera_{camera}'   # 'camera_front' or 'camera_rear'
        pt.point.x = avg_cx
        pt.point.y = avg_cy
        pt.point.z = avg_dist     # actual distance in meters
        self.pub_position.publish(pt)

        cv2.putText(annotated,
                    f'[{self.mode}] CONFIRMED {confirmed_name} dist~{avg_dist:.2f}m',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self._publish_annotated(annotated)

    def _add_vote_overlay(self, img, name_counts, fresh):
        y = 30
        cv2.putText(img, f'[{self.mode}] frames: {len(fresh)}/{CONFIRM_WINDOW}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        if name_counts:
            for name, count in name_counts.items():
                color = (0, 255, 0) if count >= CONFIRM_HITS else (0, 200, 200)
                cv2.putText(img, f'  {name}: {count}/{CONFIRM_HITS}',
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
        else:
            cv2.putText(img, '  (no detections)',
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
                f'[{self.mode}] No frames yet! '
                f'Check: ros2 topic hz /camera/front '
                f'or ros2 topic hz /camera/catrun')
        else:
            active = self._active_feed_name()
            self.get_logger().info(
                f'[{self.mode}] frames={self.frame_count} | '
                f'detections={self.detection_count} | '
                f'confirmed={self.confirmed_count} | '
                f'target={self.target_cat or "any"} | '
                f'active={active} (status={self.latest_status or "n/a"})')


def _safe_cleanup(node):
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