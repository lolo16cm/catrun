#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
import time

# ── Set your robot's starting position here ──────────────
INITIAL_X   =  0.0   # x coordinate on map
INITIAL_Y   =  0.0   # y coordinate on map
INITIAL_YAW =  90.0   # degrees — 0 = facing right, 90 = facing up
# ─────────────────────────────────────────────────────────

class SetPoseNode(Node):
    def __init__(self):
        super().__init__('set_pose_node')
        self.pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.get_logger().info(
            f'Setting initial pose: ({INITIAL_X}, {INITIAL_Y}, {INITIAL_YAW}°)')

    def publish(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = INITIAL_X
        msg.pose.pose.position.y = INITIAL_Y
        yaw = math.radians(INITIAL_YAW)
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.07

        # Publish 10 times to make sure AMCL receives it
        for i in range(10):
            self.pub.publish(msg)
            self.get_logger().info(f'Published pose {i+1}/10')
            time.sleep(0.3)

        self.get_logger().info('✅ Initial pose set!')

def main(args=None):
    rclpy.init(args=args)
    node = SetPoseNode()
    node.publish()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()