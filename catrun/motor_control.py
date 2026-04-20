#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import Jetson.GPIO as GPIO
import subprocess

# Board pin numbers
IN1 = 32
IN2 = 33
IN3 = 35
IN4 = 36

def setup_pinmux():
    """Configure pinmux for GPIO output mode"""
    cmds = [
        "busybox devmem 0x2430070 w 0x8",
        "busybox devmem 0x2430068 w 0x8",
        "busybox devmem 0x243D028 w 0x1005",
        "busybox devmem 0x243D018 w 0x5",
    ]
    for cmd in cmds:
        subprocess.run(f"sudo {cmd}", shell=True, check=False)

class MotorControlNode(Node):
    def __init__(self):
        super().__init__('motor_control')

        # Setup pinmux first
        setup_pinmux()

        # Setup GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(IN1, GPIO.OUT)
        GPIO.setup(IN2, GPIO.OUT)
        GPIO.setup(IN3, GPIO.OUT)
        GPIO.setup(IN4, GPIO.OUT)
        self.stop()

        # Subscribe to cmd_vel
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