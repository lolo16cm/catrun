#!/usr/bin/env python3
"""
motor_control.py
================
Bridge between /cmd_vel (Twist) and the L298 H-bridge driving the
4WD skid-steer chassis. Key responsibilities:

  - Translate (linear.x, angular.z) from /cmd_vel into wheel directions
  - Provide a hard LiDAR safety cutoff: refuse forward motion when
    something is closer than SAFE_DISTANCE in the front sector
  - Filter out LiDAR self-occlusion (chassis returns at 11-13 cm)

Major fixes vs. previous version:
  * Self-occlusion filter (LIDAR_MIN_RANGE) - was missing, caused the
    robot to think a wall was ALWAYS in front (its own chassis)
  * Sensible SAFE_DISTANCE (0.25 m, matching seek_cat) - was 0.05 m,
    so avoidance never fired in practice
  * No blocking time.sleep() in callbacks - was freezing the node for
    2+ seconds during recovery, missing scan/cmd_vel updates
  * Skid-steer combined motion - can drive forward-AND-turn at once
    via PWM-style alternation, so seek_cat's curving cmd_vel works
"""

import math
import os
import sys
import signal
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import Jetson.GPIO as GPIO

# ─── Pins (BOARD numbering) ──────────────────────────────────────────────
IN1 = 29   # left motor: HIGH=backward
IN2 = 31   # left motor: HIGH=forward
IN3 = 32   # right motor: HIGH=forward
IN4 = 33   # right motor: HIGH=backward

# ─── Tunables ────────────────────────────────────────────────────────────
# Front-sector LiDAR check. Matches seek_cat's SAFE_DISTANCE.
SAFE_DISTANCE        = 0.25   # m - hard stop if obstacle this close in front
LIDAR_MIN_RANGE      = 0.20   # m - ignore returns closer than this (self-occlusion)
SAFE_BACK_DISTANCE   = 0.20   # m - hard stop if obstacle this close behind

# Command thresholds (Twist values smaller than this round to zero)
LIN_DEADBAND         = 0.03   # m/s
ANG_DEADBAND         = 0.05   # rad/s

# Skid-steer combined motion: when both linear.x and angular.z are
# nonzero, alternate between "forward" and "turn" each tick. The mix
# parameter controls the duty cycle: 1.0 = pure forward, 0.0 = pure turn.
# We compute mix from the magnitudes of linear vs angular.
TICK_HZ              = 20.0   # how often we re-evaluate motor state
TICK_PERIOD          = 1.0 / TICK_HZ


def _safe_min(indices, ranges, eff_min, eff_max):
    """Min of LiDAR ranges in `indices`, filtering self-occlusion + invalids."""
    vals = []
    for i in indices:
        r = ranges[i]
        if eff_min < r < eff_max and math.isfinite(r):
            vals.append(r)
    return min(vals) if vals else float('inf')


class MotorControlNode(Node):

    def __init__(self):
        super().__init__('motor_control')

        # ─── Pinmux setup (Jetson Orin Nano) ──────────────────────────
        os.system("sudo busybox devmem 0x2430068 w 0x5")  # pin 29 IN1
        os.system("sudo busybox devmem 0x2430070 w 0x5")  # pin 31 IN2
        os.system("sudo busybox devmem 0x2434080 w 0x5")  # pin 32 IN3
        os.system("sudo busybox devmem 0x2434040 w 0x5")  # pin 33 IN4
        # Other pins from your previous config (kept in case other code
        # depends on them)
        os.system("sudo busybox devmem 0x2430050 w 0x5")  # PIN 11
        os.system("sudo busybox devmem 0x2430058 w 0x5")  # PIN 13
        os.system("sudo busybox devmem 0x2440020 w 0x5")  # PIN 15
        os.system("sudo busybox devmem 0x243D020 w 0x5")  # PIN 16
        os.system("sudo busybox devmem 0x243D040 w 0x5")  # PIN 19
        os.system("sudo busybox devmem 0x2434090 w 0x5")  # PIN 21
        os.system("sudo busybox devmem 0x2434088 w 0x5")  # PIN 22
        os.system("sudo busybox devmem 0x24340A0 w 0x5")  # PIN 23
        time.sleep(0.5)

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        self._stop_pins()

        # ─── State ────────────────────────────────────────────────────
        self.front_distance = float('inf')
        self.left_distance  = float('inf')
        self.right_distance = float('inf')
        self.back_distance  = float('inf')

        # Latest commanded twist
        self.cmd_linear  = 0.0
        self.cmd_angular = 0.0
        self.cmd_time    = 0.0   # when we last got a /cmd_vel msg

        # For PWM-style mixing of forward+turn
        self._tick_phase = 0     # alternates 0/1 each tick

        # ─── ROS interface ────────────────────────────────────────────
        self.create_subscription(Twist,     '/cmd_vel', self._cmd_cb,  10)
        self.create_subscription(LaserScan, '/scan',    self._scan_cb, 10)

        # Tick timer drives motor state. Decoupling motor state from
        # cmd_vel callbacks means a slow /cmd_vel doesn't make us
        # run-on, and lets us implement skid-steer mixing cleanly.
        self.create_timer(TICK_PERIOD, self._tick)

        self.get_logger().info('=' * 50)
        self.get_logger().info('Motor Control Node started')
        self.get_logger().info(f'  SAFE_DISTANCE     : {SAFE_DISTANCE} m')
        self.get_logger().info(f'  LIDAR_MIN_RANGE   : {LIDAR_MIN_RANGE} m '
                               f'(self-occlusion filter)')
        self.get_logger().info(f'  Tick rate         : {TICK_HZ} Hz')
        self.get_logger().info('=' * 50)

    # ─── Subscribers ──────────────────────────────────────────────────
    def _cmd_cb(self, msg: Twist):
        self.cmd_linear  = msg.linear.x
        self.cmd_angular = msg.angular.z
        self.cmd_time    = time.time()

    def _scan_cb(self, msg: LaserScan):
        ranges = msg.ranges
        n      = len(ranges)
        eff_min = max(msg.range_min, LIDAR_MIN_RANGE)
        eff_max = msg.range_max

        front_idx = list(range(0, n//12)) + list(range(11*n//12, n))
        left_idx  = list(range(n//6, n//3))
        right_idx = list(range(2*n//3, 5*n//6))
        back_idx  = list(range(5*n//12, 7*n//12))

        self.front_distance = _safe_min(front_idx, ranges, eff_min, eff_max)
        self.left_distance  = _safe_min(left_idx,  ranges, eff_min, eff_max)
        self.right_distance = _safe_min(right_idx, ranges, eff_min, eff_max)
        self.back_distance  = _safe_min(back_idx,  ranges, eff_min, eff_max)

    # ─── Tick loop: this is where motor state is decided ──────────────
    def _tick(self):
        # Failsafe: if we haven't heard /cmd_vel in a while, stop.
        # Prevents runaway if seek_cat / web_stream crashes.
        if time.time() - self.cmd_time > 0.5:
            self._stop_pins()
            return

        lin = self.cmd_linear
        ang = self.cmd_angular
        self._tick_phase = 1 - self._tick_phase

        # Apply deadbands so tiny noise doesn't drive motors
        if abs(lin) < LIN_DEADBAND: lin = 0.0
        if abs(ang) < ANG_DEADBAND: ang = 0.0

        # ─── Hard safety: refuse forward when something is in front ──
        if lin > 0 and self.front_distance < SAFE_DISTANCE:
            # Don't go forward. If a turn was requested, do that
            # in place; otherwise just stop and let the brain
            # figure out what to do next.
            self.get_logger().warn(
                f'BLOCKED forward (front={self.front_distance:.2f}m '
                f'< {SAFE_DISTANCE}m)',
                throttle_duration_sec=1.0)
            lin = 0.0

        # ─── Hard safety: refuse backward when something is behind ───
        if lin < 0 and self.back_distance < SAFE_BACK_DISTANCE:
            self.get_logger().warn(
                f'BLOCKED backward (back={self.back_distance:.2f}m '
                f'< {SAFE_BACK_DISTANCE}m)',
                throttle_duration_sec=1.0)
            lin = 0.0

        # ─── Decide motor state from (lin, ang) ──────────────────────
        # Skid-steer: pure turn, pure straight, or mix.
        if lin == 0.0 and ang == 0.0:
            self._stop_pins()
            return

        if lin == 0.0:
            # Pure rotation
            if ang > 0:
                self._turn_left()
            else:
                self._turn_right()
            return

        if ang == 0.0:
            # Pure straight
            if lin > 0:
                self._forward()
            else:
                self._backward()
            return

        # Mixed: PWM-style. Spend some ticks driving straight
        # and some ticks turning, proportional to magnitudes.
        # A simple ratio: turn_share in [0..1] based on angular
        # magnitude relative to typical max.
        ANG_NORM = 0.6   # rad/s that maps to "100% turn"
        turn_share = min(1.0, abs(ang) / ANG_NORM)
        # turn_share=0 -> always straight; turn_share=1 -> always turn
        # We alternate based on phase so the average behavior matches.
        if self._tick_phase < turn_share * 2:    # heuristic alternation
            # Turn this tick
            if ang > 0:
                self._turn_left()
            else:
                self._turn_right()
        else:
            if lin > 0:
                self._forward()
            else:
                self._backward()

    # ─── Low-level motor primitives (no logging in hot path) ──────────
    def _forward(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def _backward(self):
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def _turn_left(self):
        # left wheel backward, right wheel forward = rotate CCW
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def _turn_right(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def _stop_pins(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)

    def destroy_node(self):
        try:
            self._stop_pins()
        except Exception:
            pass
        super().destroy_node()


def _safe_cleanup(node):
    if node is not None:
        try:
            node._stop_pins()
        except Exception:
            pass
    try:
        GPIO.cleanup()
    except Exception:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = None

    def _sigterm_handler(signum, frame):
        _safe_cleanup(node)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        node = MotorControlNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[motor_control] error: {e}", file=sys.stderr)
    finally:
        _safe_cleanup(node)
        if node is not None:
            try:
                Node.destroy_node(node)
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[motor_control] clean shutdown complete")


if __name__ == '__main__':
    main()
