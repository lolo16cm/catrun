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
IN3 = 32
IN4 = 33

SAFE_DISTANCE = 0.05
SLOW_DISTANCE = 0.08

class MotorControlNode(Node):

    def __init__(self):
        super().__init__('motor_control')

        # Setup pinmux
        os.system("sudo busybox devmem 0x2430068 w 0x5")  # pin 29 IN1
        os.system("sudo busybox devmem 0x2430070 w 0x5")  # pin 31 IN2
        os.system("sudo busybox devmem 0x2434080 w 0x5")  # pin 32 IN3
        os.system("sudo busybox devmem 0x2434040 w 0x5")  # pin 33 IN4
        time.sleep(0.5)

        # Setup GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        self.stop()

        self.front_distance = float('inf')
        self.left_distance  = float('inf')
        self.right_distance = float('inf')
        self.is_recovering  = False
        self.last_angular   = 0.0

        self.create_subscription(Twist,     '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(LaserScan, '/scan',    self.scan_callback,    10)

        self.get_logger().info('Motor Control Node started!')

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

    def forward(self):
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def backward(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def turn_left(self):
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)

    def turn_right(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    def stop(self):
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)

    def recover_from_obstacle(self):
        self.is_recovering = True
        self.get_logger().warn('RECOVERING from obstacle...')
        self.backward()
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
            self.forward()
            self.get_logger().info(
                f'FORWARD front={self.front_distance:.2f}m',
                throttle_duration_sec=1.0)
        elif linear < -0.1:
            self.backward()
            self.get_logger().info('BACKWARD')
        elif angular > 0.1:
            self.turn_left()
            self.get_logger().info('LEFT')
        elif angular < -0.1:
            self.turn_right()
            self.get_logger().info('RIGHT')
        else:
            self.stop()

    def destroy_node(self):
        self.stop()
        GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
