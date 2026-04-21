#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import math
import time

WAYPOINTS = {
    'L1': (-1.68,  -1.14,  0.0),
    'L2': (-1.0,   -1.0,   0.0),
    'L3': (-0.5,   -0.5,   0.0),
    'L4': (0.0,     0.0,   0.0),
}

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
        for _ in range(5):
            self.initial_pose_pub.publish(msg)
            time.sleep(0.5)
        self.get_logger().info('Initial pose set!')

    def wait_for_nav2(self):
        self.get_logger().info('Waiting for Nav2 to be ready...')
        while not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().info('Nav2 not ready, waiting...')
        self.get_logger().info('Nav2 is ready!')

    def navigate_to(self, x, y, yaw_deg=0.0, label='goal', retries=3):
        for attempt in range(retries):
            self.get_logger().info(
                f'Navigating to {label}: ({x}, {y}) attempt {attempt+1}/{retries}')

            if not self.client.wait_for_server(timeout_sec=10.0):
                self.get_logger().error('Nav2 action server not available!')
                return False

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
                self.get_logger().warn(
                    f'Goal {label} rejected! Retrying in 3s...')
                time.sleep(3.0)
                continue

            self.get_logger().info(f'Goal {label} accepted! Waiting...')
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)

            status = result_future.result().status
            if status == 4:  # SUCCEEDED
                self.get_logger().info(f'✅ Reached {label}!')
                return True
            else:
                self.get_logger().warn(
                    f'Goal {label} failed with status {status}. Retrying...')
                time.sleep(3.0)

        self.get_logger().error(
            f'❌ Failed to reach {label} after {retries} attempts!')
        return False

    def run_mission(self):
        # Wait for Nav2 first
        self.wait_for_nav2()
        time.sleep(3.0)

        # Then set initial pose
        self.set_initial_pose(WAYPOINTS['L1'][0], WAYPOINTS['L1'][1])
        time.sleep(5.0)

        results = []
        results.append(self.navigate_to(*WAYPOINTS['L1'], label='L1 start'))
        results.append(self.navigate_to(*WAYPOINTS['L2'], label='L2'))
        results.append(self.navigate_to(*WAYPOINTS['L3'], label='L3'))
        results.append(self.navigate_to(*WAYPOINTS['L4'], label='L4'))
        results.append(self.navigate_to(*WAYPOINTS['L1'], label='L1 return'))

        if all(results):
            self.get_logger().info('✅ Mission complete! All waypoints reached!')
        else:
            self.get_logger().error('❌ Mission failed! Some waypoints not reached!')


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    node.run_mission()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()