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
  width         : int   output (downscaled) width            (default: CSI 640, USB 640)
  height        : int   output (downscaled) height           (default: CSI 360, USB 480)
  framerate     : int   publish rate in Hz                   (default: 15)
                        Note: CSI capture is ALWAYS native 1920x1080@60.
                        nvvidconv downscales on GPU to width/height.
                        The python timer publishes at `framerate` Hz.
  device_path   : str   override for usbN, e.g. '/dev/video1' (default: derived)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


# CSI capture mode: we always capture at native 1920x1080@60 because the
# IMX477 supports this fast mode reliably (confirmed via gst-launch).
# Asking nvarguscamerasrc for arbitrary downscaled resolutions can force
# a slow path or fail. nvvidconv downscales on the GPU for free.
CSI_CAPTURE_W   = 1920
CSI_CAPTURE_H   = 1080
CSI_CAPTURE_FPS = 60


def build_csi_pipeline(sensor_id, out_w, out_h, flip_method):
    """GStreamer pipeline for a CSI camera via nvarguscamerasrc.

    Strategy:
      - Capture at native IMX477 1920x1080@60 (known-good fast mode).
      - Downscale to (out_w, out_h) on the GPU using nvvidconv.
      - appsink drops all but the newest frame: drop=1, max-buffers=1.

    The Python timer in CameraNode controls publish rate independently
    of the 60 Hz capture - publishing at 15 Hz simply means we discard
    3 of every 4 captured frames at the appsink boundary.
    """
    return (
        f'nvarguscamerasrc sensor-id={sensor_id} ! '
        f'video/x-raw(memory:NVMM), width={CSI_CAPTURE_W}, '
        f'height={CSI_CAPTURE_H}, framerate={CSI_CAPTURE_FPS}/1, '
        f'format=NV12 ! '
        f'nvvidconv flip-method={flip_method} ! '
        f'video/x-raw, width={out_w}, height={out_h}, format=BGRx ! '
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
        self.declare_parameter('framerate', 15)   # publish rate in Hz
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
            if width  == 0: width  = 640
            if height == 0: height = 360
            pipeline = build_csi_pipeline(
                sensor_id, width, height, flip_method)
            self.get_logger().info(
                f'[Camera] CSI sensor-id={sensor_id} '
                f'capture={CSI_CAPTURE_W}x{CSI_CAPTURE_H}@{CSI_CAPTURE_FPS} '
                f'-> out {width}x{height} publish={framerate}Hz '
                f'flip={flip_method}')

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
                f'[Camera] USB {device_path} {width}x{height}@{framerate}')

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

        # Force OpenCV's internal buffer to size 1 so cap.read() returns
        # the newest frame, not a queued old one.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Publish rate timer. Capture is at 60fps, publish at framerate.
        # With `drop=1 max-buffers=1` in appsink, each read returns the
        # latest captured frame - older ones are discarded automatically.
        self.create_timer(1.0 / framerate, self.timer_callback)

    def timer_callback(self):
        # ONE read per tick = the newest frame. The previous version did
        # a 3x "drain" loop, but with appsink drop=1 max-buffers=1 the
        # second read BLOCKS waiting for a new frame from GStreamer,
        # which capped publish rate at ~10 Hz. Single read is correct
        # AND fast.
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warn('Failed to grab frame',
                                   throttle_duration_sec=2.0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
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