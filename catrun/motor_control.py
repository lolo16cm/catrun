#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import Jetson.GPIO as GPIO
import os
import time

# Board pin numbers
IN1 = 31
IN2 = 29
IN3 = 32
IN4 = 33

SAFE_DISTANCE = 0.25  # meters - stop if obstacle closer than this
SLOW_DISTANCE = 0.45  # meters - slow down if obstacle closer than this

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control')

        # Setup pinmux
        os.system("sudo busybox devmem 0x2430068 w 0x5")
        os.system("sudo busybox devmem 0x2430070 w 0x5")
        os.system("sudo busybox devmem 0x2434080 w 0x5")
        os.system("sudo busybox devmem 0x2434040 w 0x5")
        time.sleep(0.5)

        # Setup GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        self.stop()

        self.front_distance = float('inf')

        # Subscribe to cmd_vel
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Subscribe to LiDAR scan for obstacle detection
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('Motor Control Node started!')

    def scan_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)
        # Check front ±30 degrees
        front_indices = list(range(0, n//12)) + list(range(11*n//12, n))
        front_ranges = [ranges[i] for i in front_indices 
                       if 0.05 < ranges[i] < 10.0 and ranges[i] != float('inf')]
        if front_ranges:
            self.front_distance = min(front_ranges)
        else:
            self.front_distance = float('inf')

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

    def cmd_vel_callback(self, msg):
        linear  = msg.linear.x
        angular = msg.angular.z

        # Obstacle check — only block forward movement
        if linear > 0.1 and self.front_distance < SAFE_DISTANCE:
            self.stop()
            self.get_logger().warn(
                f'OBSTACLE! front={self.front_distance:.2f}m STOP')
            return

        if linear > 0.1:
            self.forward()
            self.get_logger().info(
                f'FORWARD front={self.front_distance:.2f}m')
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
            self.get_logger().info('STOP')

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