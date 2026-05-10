#!/usr/bin/env python3
"""
camera_node.py
==============
Single camera publisher node that supports BOTH cameras on the robot:

  - Front CSI camera (sensor-id=0)  -> used in WATCH mode (seek the cat)
  - Rear  USB camera (/dev/video1)  -> used in PLAY  mode (flee from cat)

Only ONE camera is opened per process (saves CPU). The launch file picks
which one by setting the `camera_source` parameter.

Both cameras publish to the SAME topic `/camera/catrun`, so the cat
detector and behavior nodes don't need to change topic names between
modes.

Parameters
----------
  camera_source : str   'csi0' | 'csi1' | 'usb1' | 'usb0'   (default: csi0)
  flip_method   : int   nvvidconv flip-method for CSI cams  (default: 2)
                        0=none, 2=180deg, 4=horiz flip, 6=vert flip ...
  width         : int   capture width  (default: 1280 for CSI, 640 for USB)
  height        : int   capture height (default: 720  for CSI, 480 for USB)
  framerate     : int   target fps (default: 30; publish rate is fps/2)
  device_path   : str   override for usbN, e.g. '/dev/video1' (default: derived)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


def build_csi_pipeline(sensor_id, width, height, framerate, flip_method):
    """GStreamer pipeline for a CSI camera via nvarguscamerasrc.

    Low-latency tuning:
      - appsink: max-buffers=1, drop=1, sync=false -> keep only newest
      - leaky queue between stages so old frames are dropped if a stage stalls
    """
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} ! '
        f'video/x-raw(memory:NVMM), width={width}, height={height}, '
        f'framerate={framerate}/1 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, format=BGRx ! '
        f'queue leaky=downstream max-size-buffers=1 ! '
        f'videoconvert ! '
        f'video/x-raw, format=BGR ! '
        f'appsink drop=1 sync=false max-buffers=1'
    )


def build_usb_pipeline(device_path, width, height, framerate):
    """GStreamer pipeline for a USB / V4L2 camera (e.g. /dev/video1).

    Same low-latency tuning as the CSI pipeline.
    """
    return (
        f'v4l2src device={device_path} io-mode=2 ! '
        f'video/x-raw, width={width}, height={height}, '
        f'framerate={framerate}/1 ! '
        f'queue leaky=downstream max-size-buffers=1 ! '
        f'videoconvert ! '
        f'video/x-raw, format=BGR ! '
        f'appsink drop=1 sync=false max-buffers=1'
    )


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # ─── parameters ────────────────────────────────────────────────
        self.declare_parameter('camera_source', 'csi0')
        self.declare_parameter('flip_method', 2)
        self.declare_parameter('width', 0)        # 0 = pick a sensible default
        self.declare_parameter('height', 0)
        self.declare_parameter('framerate', 30)
        self.declare_parameter('device_path', '')

        source      = self.get_parameter('camera_source').value.lower().strip()
        flip_method = int(self.get_parameter('flip_method').value)
        width       = int(self.get_parameter('width').value)
        height      = int(self.get_parameter('height').value)
        framerate   = int(self.get_parameter('framerate').value)
        device_path = self.get_parameter('device_path').value.strip()

        self.bridge = CvBridge()
        self.pub    = self.create_publisher(Image, '/camera/catrun', 10)

        # ─── pick the right pipeline ──────────────────────────────────
        if source in ('csi0', 'csi1'):
            sensor_id = 0 if source == 'csi0' else 1
            if width  == 0: width  = 1280
            if height == 0: height = 720
            pipeline = build_csi_pipeline(
                sensor_id, width, height, framerate, flip_method)
            self.get_logger().info(
                f'[Camera] CSI sensor-id={sensor_id} '
                f'{width}x{height}@{framerate} flip={flip_method}')

        elif source in ('usb0', 'usb1') or source.startswith('/dev/video'):
            if not device_path:
                if source == 'usb0':
                    device_path = '/dev/video0'
                elif source == 'usb1':
                    device_path = '/dev/video1'
                else:
                    device_path = source        # already a /dev/videoN string
            if width  == 0: width  = 640
            if height == 0: height = 480
            pipeline = build_usb_pipeline(
                device_path, width, height, framerate)
            self.get_logger().info(
                f'[Camera] USB {device_path} '
                f'{width}x{height}@{framerate}')

        else:
            self.get_logger().error(
                f"Unknown camera_source '{source}'. "
                f"Use one of: csi0, csi1, usb0, usb1, /dev/videoN")
            return

        # ─── open the camera ──────────────────────────────────────────
        self.get_logger().info('Opening camera...')
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            self.get_logger().error(
                f"Camera failed to open! source={source} "
                f"device={device_path or 'n/a'}")
            # Fallback for USB: try opening the device index directly without
            # GStreamer, in case the v4l2src plugin isn't available.
            if source.startswith('usb') or source.startswith('/dev/video'):
                idx = 1 if (source == 'usb1' or device_path.endswith('1')) else 0
                self.get_logger().warn(
                    f'Falling back to cv2.VideoCapture({idx}) without GStreamer')
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    self.cap.set(cv2.CAP_PROP_FPS,          framerate)
            if not self.cap.isOpened():
                return

        self.get_logger().info(
            f'Camera opened! Publishing to /camera/catrun ({source})')

        # Force OpenCV's internal buffer to size 1 so cv2.VideoCapture.read()
        # returns the newest frame instead of an old one from the queue.
        # This is essential for low latency.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Publish at the same rate the camera produces frames. Publishing
        # slower than the camera's framerate causes lag because frames
        # accumulate in the GStreamer/OpenCV buffer between reads.
        publish_rate = framerate
        self.create_timer(1.0 / publish_rate, self.timer_callback)
        self._last_publish_warn_t = 0.0

    def timer_callback(self):
        # Drain any stale frames from the buffer so we publish the
        # newest one. With max-buffers=1 in the GStreamer appsink and
        # CAP_PROP_BUFFERSIZE=1 in OpenCV this should usually be a
        # single read, but the loop is defensive against driver-level
        # buffering on USB cameras.
        latest = None
        for _ in range(3):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                latest = frame
            else:
                break

        if latest is None:
            self.get_logger().warn('Failed to grab frame',
                                   throttle_duration_sec=2.0)
            return

        msg = self.bridge.cv2_to_imgmsg(latest, 'bgr8')
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'
        self.pub.publish(msg)

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()