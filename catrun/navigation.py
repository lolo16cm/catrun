#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math
import time

# ─── Define your waypoints here ───────────────────────────
WAYPOINTS = {
    'L1': (-0.00548, -0.0158,  0.0),   # Start point
    'L2': (3.19,      1.16,    0.0),   # Middle point
    'L3': (-0.157,   -2.81,    0.0),   # End point
}
# ──────────────────────────────────────────────────────────

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.get_logger().info('Navigation Node started!')

    def set_initial_pose(self, x, y, yaw_deg=0.0):
        self.get_logger().info(f'Setting initial pose: ({x}, {y})')
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        yaw = math.radians(yaw_deg)
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.07
        # Publish multiple times to make sure it's received
        for _ in range(5):
            self.initial_pose_pub.publish(msg)
            time.sleep(0.5)
        self.get_logger().info('Initial pose set!')

    def navigate_to(self, x, y, yaw_deg=0.0, label='goal'):
        self.get_logger().info(f'Navigating to {label}: ({x}, {y})')
        self.client.wait_for_server()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        yaw = math.radians(yaw_deg)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn(f'Goal {label} rejected!')
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info(f'Reached {label}!')
        return True

    def run_mission(self):
        # Set initial pose first
        self.set_initial_pose(-0.00548, -0.0158)
        # Wait for AMCL to localize
        self.get_logger().info('Waiting for Nav2 to be ready...')
        time.sleep(5.0)

        # L1 → L2 → L3 → L1
        self.navigate_to(*WAYPOINTS['L1'], label='L1 start')
        self.navigate_to(*WAYPOINTS['L2'], label='L2')
        self.navigate_to(*WAYPOINTS['L3'], label='L3')
        self.navigate_to(*WAYPOINTS['L1'], label='L1 return')
        self.get_logger().info('Mission complete!')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    node.run_mission()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()