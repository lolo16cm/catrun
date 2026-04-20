#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import math

# ─── Define your waypoints here ───────────────────────────
# Format: (x, y, yaw_degrees)
# Measure these from your map in RViz using "Publish Point"
WAYPOINTS = {
    'L1': (-0.00548,  -0.0158,  0.0),   # Start point
    'L2': (3.19,  1.16,  0.0),   # Middle point  
    'L3': (-0.157, -2.81, 0.0),   # End point
}
# ──────────────────────────────────────────────────────────

class NavigationNode(Node):
    def __init__(self):
        super().__init__('navigation_node')
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().info('Navigation Node started!')

    def navigate_to(self, x, y, yaw_deg=0.0, label='goal'):
        self.get_logger().info(f'Navigating to {label}: ({x}, {y})')

        # Wait for Nav2
        self.client.wait_for_server()

        # Build goal
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y

        # Convert yaw to quaternion
        yaw = math.radians(yaw_deg)
        goal.pose.pose.orientation.z = math.sin(yaw / 2)
        goal.pose.pose.orientation.w = math.cos(yaw / 2)

        # Send goal and wait
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