#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import Jetson.GPIO as GPIO
import os
import time
import math

# Board pin numbers
IN1 = 29
IN2 = 31
IN3 = 7    # moved from 32
IN4 = 11    # moved from 33
ENA = 32    # moved from 15 (hardware PWM)
ENB = 33    # moved from 16 (hardware PWM)

SAFE_DISTANCE = 0.05
SLOW_DISTANCE = 0.08
PWM_FREQ      = 1000   # Hz

# Speed levels (0-100)
FULL_SPEED = 100
SLOW_SPEED = 50
TURN_SPEED = 70


class MotorControlNode(Node):

    def __init__(self):
        super().__init__('motor_control')

        # Setup pinmux
        os.system("sudo busybox devmem 0x2430068 w 0x5")   # pin 29 IN1
        os.system("sudo busybox devmem 0x2430070 w 0x5")   # pin 31 IN2
        os.system("sudo busybox devmem 0x02430038 w 0x5") # pin 7  IN3
        os.system("sudo busybox devmem 0x02430090 w 0x5") # pin 11 IN4
        os.system("sudo busybox devmem 0x2434080 w 0x5")   # pin 32 (ENA)
        os.system("sudo busybox devmem 0x2434040 w 0x5")   # pin 33 (ENB)
        time.sleep(0.5)

        # Setup GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        GPIO.setup(ENA, GPIO.OUT)
        GPIO.setup(ENB, GPIO.OUT)

        # Setup PWM
        self.pwm_a = GPIO.PWM(ENA, PWM_FREQ)
        self.pwm_b = GPIO.PWM(ENB, PWM_FREQ)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

        self.front_distance = float('inf')
        self.left_distance  = float('inf')
        self.right_distance = float('inf')
        self.is_recovering  = False
        self.last_angular   = 0.0

        self.create_subscription(Twist,     '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(LaserScan, '/scan',    self.scan_callback,    10)

        self.get_logger().info('Motor Control Node started with PWM speed control!')

    # ── speed control ─────────────────────────────────────────────────────────

    def set_speed(self, speed):
        """Set speed for both motors (0-100)."""
        self.pwm_a.ChangeDutyCycle(speed)
        self.pwm_b.ChangeDutyCycle(speed)

    # ── LiDAR ─────────────────────────────────────────────────────────────────

    def safe_min(self, indices, ranges):
        vals = [ranges[i] for i in indices
                if 0.05 < ranges[i] < 10.0
                and not math.isinf(ranges[i])
                and not math.isnan(ranges[i])]
        return min(vals) if vals else float('inf')

    def scan_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        front_idx = list(range(0, n//12)) + list(range(11*n//12, n))
        left_idx  = list(range(n//6, n//3))
        right_idx = list(range(2*n//3, 5*n//6))
        self.front_distance = self.safe_min(front_idx, ranges)
        self.left_distance  = self.safe_min(left_idx,  ranges)
        self.right_distance = self.safe_min(right_idx, ranges)

    # ── motor directions ──────────────────────────────────────────────────────

    def forward(self, speed=FULL_SPEED):
        self.set_speed(speed)
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def backward(self, speed=FULL_SPEED):
        self.set_speed(speed)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def turn_left(self, speed=TURN_SPEED):
        self.set_speed(speed)
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def turn_right(self, speed=TURN_SPEED):
        self.set_speed(speed)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def stop(self):
        self.set_speed(0)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)

    # ── obstacle recovery ─────────────────────────────────────────────────────

    def recover_from_obstacle(self):
        self.is_recovering = True
        self.get_logger().warn('RECOVERING from obstacle...')
        self.backward(speed=SLOW_SPEED)
        time.sleep(1.0)
        self.stop()
        time.sleep(0.2)
        if self.right_distance > self.left_distance:
            self.get_logger().info('Turning RIGHT (more space)')
            self.turn_right()
        else:
            self.get_logger().info('Turning LEFT (more space)')
            self.turn_left()
        time.sleep(0.8)
        self.stop()
        self.is_recovering = False

    # ── cmd_vel callback ──────────────────────────────────────────────────────

    def cmd_vel_callback(self, msg):
        if self.is_recovering:
            return

        linear  = msg.linear.x
        angular = msg.angular.z
        self.last_angular = angular

        if linear > 0.1 and self.front_distance < SAFE_DISTANCE:
            self.recover_from_obstacle()
            return

        if linear > 0.1:
            # Slow down when obstacle is close
            if self.front_distance < SLOW_DISTANCE: