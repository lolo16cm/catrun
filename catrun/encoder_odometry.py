#!/usr/bin/env python3
"""
encoder_odometry.py
Reads quadrature encoders from 4 motors (2 left, 2 right in parallel)
and publishes /odom topic for Nav2.

Motor specs:
  - 7 PPR base encoder
  - 1:150 gear reduction
  - Total: 1050 pulses per revolution

Robot specs:
  - Wheel diameter: 65mm
  - Track width: 110mm
"""

import math
import time
import threading

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import Jetson.GPIO as GPIO

# ── Encoder pins (BOARD numbering) ───────────────────────────────────────────
# Front Left
FL_A = 11
FL_B = 13
# Front Right
FR_A = 15
FR_B = 16
# Rear Left
RL_A = 19
RL_B = 21
# Rear Right
RR_A = 22
RR_B = 23

# ── Robot specs ───────────────────────────────────────────────────────────────
WHEEL_DIAMETER  = 0.065          # meters
WHEEL_RADIUS    = WHEEL_DIAMETER / 2.0
TRACK_WIDTH     = 0.110          # meters
PPR_BASE        = 7              # pulses per revolution (base)
GEAR_RATIO      = 150            # gear reduction ratio
PPR_TOTAL       = PPR_BASE * GEAR_RATIO * 4  # x4 for quadrature = 4200
WHEEL_CIRCUM    = math.pi * WHEEL_DIAMETER   # meters per revolution

# Distance per pulse
DIST_PER_PULSE  = WHEEL_CIRCUM / PPR_TOTAL   # meters


class EncoderOdometry(Node):

    def __init__(self):
        super().__init__('encoder_odometry')

        # ── Robot pose ────────────────────────────────────────────────────────
        self.x     = 0.0
        self.y     = 0.0
        self.theta = 0.0

        # ── Encoder counts ────────────────────────────────────────────────────
        # Left = FL + RL averaged (parallel wiring)
        # Right = FR + RR averaged (parallel wiring)
        self.fl_count = 0
        self.fr_count = 0
        self.rl_count = 0
        self.rr_count = 0

        self.last_left_count  = 0
        self.last_right_count = 0
        self.last_time        = self.get_clock().now()

        self.lock = threading.Lock()

        # ── GPIO setup ────────────────────────────────────────────────────────
        GPIO.setmode(GPIO.BOARD)
        pins = [FL_A, FL_B, FR_A, FR_B, RL_A, RL_B, RR_A, RR_B]
        for pin in pins:
            GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Quadrature encoder callbacks
        GPIO.add_event_detect(FL_A, GPIO.BOTH,
            callback=lambda ch: self._encoder_cb('FL', ch))
        GPIO.add_event_detect(FR_A, GPIO.BOTH,
            callback=lambda ch: self._encoder_cb('FR', ch))
        GPIO.add_event_detect(RL_A, GPIO.BOTH,
            callback=lambda ch: self._encoder_cb('RL', ch))
        GPIO.add_event_detect(RR_A, GPIO.BOTH,
            callback=lambda ch: self._encoder_cb('RR', ch))

        # ── Publishers ────────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Timer: publish odometry at 20Hz ───────────────────────────────────
        self.create_timer(0.05, self.publish_odom)

        self.get_logger().info('EncoderOdometry started!')
        self.get_logger().info(
            f'PPR total (quadrature): {PPR_TOTAL} | '
            f'Dist/pulse: {DIST_PER_PULSE*1000:.4f}mm')

    # ── Encoder callback ──────────────────────────────────────────────────────

    def _encoder_cb(self, motor: str, channel: int):
        """
        Quadrature decode: read Phase B to determine direction.
        If A rising and B low  → forward (+1)
        If A rising and B high → backward (-1)
        """
        a_state = GPIO.input(channel)

        if motor == 'FL':
            b_state = GPIO.input(FL_B)
            direction = 1 if (a_state == GPIO.HIGH and b_state == GPIO.LOW) \
                          or (a_state == GPIO.LOW  and b_state == GPIO.HIGH) \
                        else -1
            with self.lock:
                self.fl_count += direction

        elif motor == 'FR':
            b_state = GPIO.input(FR_B)
            direction = 1 if (a_state == GPIO.HIGH and b_state == GPIO.LOW) \
                          or (a_state == GPIO.LOW  and b_state == GPIO.HIGH) \
                        else -1
            with self.lock:
                self.fr_count += direction

        elif motor == 'RL':
            b_state = GPIO.input(RL_B)
            direction = 1 if (a_state == GPIO.HIGH and b_state == GPIO.LOW) \
                          or (a_state == GPIO.LOW  and b_state == GPIO.HIGH) \
                        else -1
            with self.lock:
                self.rl_count += direction

        elif motor == 'RR':
            b_state = GPIO.input(RR_B)
            direction = 1 if (a_state == GPIO.HIGH and b_state == GPIO.LOW) \
                          or (a_state == GPIO.LOW  and b_state == GPIO.HIGH) \
                        else -1
            with self.lock:
                self.rr_count += direction

    # ── Odometry calculation ──────────────────────────────────────────────────

    def publish_odom(self):
        now = self.get_clock().now()
        dt  = (now - self.last_time).nanoseconds / 1e9
        if dt <= 0:
            return
        self.last_time = now

        with self.lock:
            # Average front and rear for each side (parallel wiring)
            left_count  = (self.fl_count + self.rl_count) // 2
            right_count = (self.fr_count + self.rr_count) // 2

        # Delta counts since last update
        d_left  = (left_count  - self.last_left_count)  * DIST_PER_PULSE
        d_right = (right_count - self.last_right_count) * DIST_PER_PULSE

        self.last_left_count  = left_count
        self.last_right_count = right_count

        # Differential drive kinematics
        d_center = (d_left + d_right) / 2.0
        d_theta  = (d_right - d_left) / TRACK_WIDTH

        # Update pose
        self.x     += d_center * math.cos(self.theta + d_theta / 2.0)
        self.y     += d_center * math.sin(self.theta + d_theta / 2.0)
        self.theta += d_theta
        self.theta  = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Velocities
        vx = d_center / dt
        wz = d_theta  / dt

        # Quaternion from yaw
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)

        # ── Publish TF ────────────────────────────────────────────────────────
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = now.to_msg()
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id  = 'base_link'
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.z    = qz
        tf_msg.transform.rotation.w    = qw
        self.tf_broadcaster.sendTransform(tf_msg)

        # ── Publish Odometry ──────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp            = now.to_msg()
        odom.header.frame_id         = 'odom'
        odom.child_frame_id          = 'base_link'
        odom.pose.pose.position.x    = self.x
        odom.pose.pose.position.y    = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x    = vx
        odom.twist.twist.angular.z   = wz

        # Covariance (diagonal)
        odom.pose.covariance[0]  = 0.01
        odom.pose.covariance[7]  = 0.01
        odom.pose.covariance[35] = 0.01
        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[35] = 0.01

        self.odom_pub.publish(odom)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EncoderOdometry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()