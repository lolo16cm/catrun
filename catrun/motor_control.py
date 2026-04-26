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
IN1 = 32    # was 31
IN2 = 33    # was 29
IN3 = 31    # was 32
IN4 = 29    # was 33

SAFE_DISTANCE = 0.05  # meters - stop if obstacle closer than this
SLOW_DISTANCE = 0.08  # meters - slow down if obstacle closer than this

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
        self.left_distance  = float('inf')  # New
        self.right_distance = float('inf')  # New
        self.is_recovering  = False         # New

        # Last cmd_vel angular to know which direction Nav2 wants to go
        self.last_angular = 0.0             # New

        # Subscribe to cmd_vel
        self.sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Subscribe to LiDAR scan for obstacle detection
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

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
        # Check front ±30 degrees
        # front_indices = list(range(0, n//12)) + list(range(11*n//12, n))
        # front_ranges = [ranges[i] for i in front_indices 
        #                if 0.05 < ranges[i] < 10.0 and ranges[i] != float('inf')]
        # if front_ranges:
        #     self.front_distance = min(front_ranges)
        # else:
        #     self.front_distance = float('inf')
        
        # New
        # Front ±30 degrees
        front_idx = list(range(0, n//12)) + list(range(11*n//12, n))
        # Left 60-120 degrees
        left_idx  = list(range(n//6, n//3))
        # Right 240-300 degrees
        right_idx = list(range(2*n//3, 5*n//6))

        self.front_distance = self.safe_min(front_idx, ranges)
        self.left_distance  = self.safe_min(left_idx, ranges)
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

    # New
    # def turn_90_toward_goal(self):
    #     """Turn 90 degrees toward the side with more space (toward goal direction)"""
    #     self.is_recovering = True
    #     self.get_logger().warn(
    #         f'OBSTACLE! Turning 90° toward goal. '
    #         f'left={self.left_distance:.2f}m right={self.right_distance:.2f}m')

    #     self.stop()
    #     time.sleep(0.2)

    #     # Choose turn direction:
    #     # If Nav2 wants to turn left (positive angular) → turn left
    #     # If Nav2 wants to turn right (negative angular) → turn right
    #     # If no preference → turn toward side with more space
    #     if self.last_angular > 0.1:
    #         turn_dir = 'left'
    #     elif self.last_angular < -0.1:
    #         turn_dir = 'right'
    #     else:
    #         turn_dir = 'left' if self.left_distance > self.right_distance else 'right'

    #     self.get_logger().info(f'Turning {turn_dir} 90°')

    #     # Turn for ~0.9s ≈ 90 degrees (tune this value for your robot)
    #     if turn_dir == 'left':
    #         self.turn_left()
    #     else:
    #         self.turn_right()

    #     time.sleep(0.9)  # Adjust this to get exactly 90°
    #     self.stop()
    #     time.sleep(0.2)

    #     self.is_recovering = False
    #     self.get_logger().info('Recovery complete, resuming navigation')

    def recover_from_obstacle(self):
        """Back up and turn away from obstacle"""
        self.is_recovering = True
        self.get_logger().warn('RECOVERING from obstacle...')

        # Back up for 1 second
        self.backward()
        time.sleep(1.0)
        self.stop()
        time.sleep(0.2)

        # Turn toward the side with more space
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

        # Save Nav2's desired angular for turn direction hint
        self.last_angular = angular

        # Obstacle ahead while moving forward
        if linear > 0.1 and self.front_distance < SAFE_DISTANCE:
            self.turn_90_toward_goal()
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
        # linear  = msg.linear.x
        # angular = msg.angular.z

        # # Obstacle check — only block forward movement
        # if linear > 0.1 and self.front_distance < SAFE_DISTANCE:
        #     self.stop()
        #     self.get_logger().warn(
        #         f'OBSTACLE! front={self.front_distance:.2f}m STOP')
        #     return

        # if linear > 0.1:
        #     self.forward()
        #     self.get_logger().info(
        #         f'FORWARD front={self.front_distance:.2f}m')
        # elif linear < -0.1:
        #     self.backward()
        #     self.get_logger().info('BACKWARD')
        # elif angular > 0.1:
        #     self.turn_left()
        #     self.get_logger().info('LEFT')
        # elif angular < -0.1:
        #     self.turn_right()
        #     self.get_logger().info('RIGHT')
        # else:
        #     self.stop()
        #     self.get_logger().info('STOP')

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
