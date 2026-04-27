#!/usr/bin/env python3
"""
seek_cat.py
Navigates the robot toward a target cat.
- Steers toward cat using camera position when visible
- Spins in place when cat is lost
- Visits L1→L2→L3→L4 waypoints if cat still not found
- Loops back to L1 after all waypoints searched
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import String

# ── config ────────────────────────────────────────────────────────────────────
LINEAR_SPEED  = 0.15   # forward speed m/s
ANGULAR_SPEED = 0.4    # turning speed rad/s
CENTER_THRESH = 0.15   # how close to center before moving forward
LOST_TIMEOUT  = 3.0    # seconds before declaring cat lost
SPIN_DURATION = 4.0    # seconds to spin at each waypoint before moving on
# ─────────────────────────────────────────────────────────────────────────────

# !! Update these to match your navigation.py waypoints !!
WAYPOINTS = [
    ('L1',  0.726,  0.831),
    ('L2', -1.0,   -1.0),
    ('L3', -0.5,   -0.5),
    ('L4',  0.0,    0.0),
]


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        self.target_cat     = None
        self.last_seen      = None
        self.cat_cx         = None
        self.active         = False

        # Waypoint search state
        self.searching      = False
        self.spin_start     = None
        self.waypoint_index = 0
        self.nav_sent       = False

        # Nav2 action client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscribers
        self.create_subscription(String,       '/cat_target',   self.target_cb,   10)
        self.create_subscription(String,       '/cat_identity', self.identity_cb, 10)
        self.create_subscription(PointStamped, '/cat_position', self.position_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop at 10 Hz
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Seek Cat node ready. Waiting for /cat_target...')

    # ── callbacks ─────────────────────────────────────────────────────────────

    def target_cb(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self.target_cat = None
            self.active     = False
            self.searching  = False
            self.stop_robot()
            self.get_logger().info('Seek stopped.')
        else:
            self.target_cat     = text
            self.active         = True
            self.searching      = False
            self.waypoint_index = 0
            self.nav_sent       = False
            self.last_seen      = None
            self.cat_cx         = None
            self.get_logger().info(f'Seeking: {self.target_cat}')

    def identity_cb(self, msg: String):
        if self.target_cat and msg.data.strip().lower() == self.target_cat:
            self.last_seen = self.get_clock().now()
            # Cat found — exit search mode
            if self.searching:
                self.get_logger().info(f'Found {self.target_cat}! Switching to seek.')
                self.searching  = False
                self.nav_sent   = False
                self.spin_start = None
                self.stop_robot()

    def position_cb(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.last_seen = self.get_clock().now()

    # ── control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.active or self.target_cat is None:
            return

        # Cat visible recently → steer toward it
        if self.last_seen is not None:
            elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
            if elapsed < LOST_TIMEOUT:
                self.searching = False
                self.steer_toward_cat()
                return

        # Cat lost → search via waypoints
        self.searching = True
        self.waypoint_search()

    # ── steering ──────────────────────────────────────────────────────────────

    def steer_toward_cat(self):
        if self.cat_cx is None:
            return

        twist = Twist()
        error = self.cat_cx - 0.5   # negative=left, positive=right

        if abs(error) > CENTER_THRESH:
            # Turn toward cat + slow creep forward
            twist.angular.z = -ANGULAR_SPEED * (error / 0.5)
            twist.linear.x  = 0.05
        else:
            # Cat centered → move forward
            twist.linear.x  = LINEAR_SPEED
            twist.angular.z = -ANGULAR_SPEED * (error / 0.5) * 0.3

        self.cmd_pub.publish(twist)

    # ── waypoint search ───────────────────────────────────────────────────────

    def waypoint_search(self):
        label, wx, wy = WAYPOINTS[self.waypoint_index]

        # Send Nav2 goal to next waypoint
        if not self.nav_sent:
            self.get_logger().info(
                f'Searching... heading to {label} ({wx}, {wy})')
            self.send_nav_goal(wx, wy)
            self.nav_sent   = True
            self.spin_start = None
            return

        # Arrived — start spinning to look around
        if self.spin_start is None:
            self.spin_start = self.get_clock().now()
            self.get_logger().info(f'Arrived at {label}, spinning to search...')

        elapsed_spin = (self.get_clock().now() - self.spin_start).nanoseconds / 1e9

        if elapsed_spin < SPIN_DURATION:
            # Keep spinning
            twist = Twist()
            twist.angular.z = ANGULAR_SPEED
            self.cmd_pub.publish(twist)
            return

        # Spin done, cat not found → next waypoint
        self.get_logger().info(
            f'{self.target_cat} not found at {label}, trying next waypoint...')
        self.waypoint_index = (self.waypoint_index + 1) % len(WAYPOINTS)
        self.nav_sent       = False
        self.spin_start     = None

    def send_nav_goal(self, x, y):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('Nav2 not ready, skipping waypoint.')
            self.nav_sent = False
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        self.nav_client.send_goal_async(goal)
        self.get_logger().info(f'Nav2 goal sent: ({x}, {y})')

    # ── helpers ───────────────────────────────────────────────────────────────

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = SeekCat()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()