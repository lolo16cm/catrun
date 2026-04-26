#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MotorTest(Node):
    def __init__(self):
        super().__init__('motor_test')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(1.0)  # wait for publisher to connect
        self.run_test()

    def send(self, linear=0.0, angular=0.0, duration=2.0, label=''):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.get_logger().info(f'Testing: {label}')
        end = time.time() + duration
        while time.time() < end:
            self.pub.publish(msg)
            time.sleep(0.1)
        # Stop
        self.pub.publish(Twist())
        time.sleep(0.5)

    def run_test(self):
        self.get_logger().info('=== Motor Test Starting ===')

        self.send(linear=0.3,  angular=0.0,  duration=2.0, label='FORWARD')
        self.send(linear=-0.3, angular=0.0,  duration=2.0, label='BACKWARD')
        self.send(linear=0.0,  angular=0.5,  duration=2.0, label='TURN LEFT')
        self.send(linear=0.0,  angular=-0.5, duration=2.0, label='TURN RIGHT')

        self.get_logger().info('=== Motor Test Complete ===')

def main(args=None):
    rclpy.init(args=args)
    node = MotorTest()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
