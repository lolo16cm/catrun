#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import Jetson.GPIO as GPIO
import os
import time

# BOARD pin numbers - Jetson Orin Nano safe GPIO pins
IN1 = 29  # PQ.05
IN2 = 31  # PQ.06
IN3 = 32  # PG.06
IN4 = 33  # PH.00

def setup_pinmux():
    os.system("sudo busybox devmem 0x2430068 w 0x5")  # PIN 29
    os.system("sudo busybox devmem 0x2430070 w 0x5")  # PIN 31
    os.system("sudo busybox devmem 0x2434080 w 0x5")  # PIN 32
    os.system("sudo busybox devmem 0x2434040 w 0x5")  # PIN 33
    time.sleep(0.5)

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control')

        setup_pinmux()

        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        self.stop()

        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)

        self.get_logger().info('Motor Control Node started!')

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

        if linear > 0.1:
            self.forward()
            self.get_logger().info('FORWARD')
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