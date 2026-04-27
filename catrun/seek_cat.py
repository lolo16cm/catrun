#!/usr/bin/env python3
"""
seek_cat.py
Navigates the robot toward a target cat (Eevee, Raichu, or Pichu).

Detection pipeline:
  1. LiDAR scan diff → detect moving clusters of cat-size
  2. If moving cluster found → trigger YOLOv8 camera check
  3. If no motion for IDLE_SCAN_INTERVAL → periodic camera sweep
  4. If cat confirmed → steer toward it
  5. If cat lost for LOST_TIMEOUT → spin + visit waypoints
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# ── tunables ──────────────────────────────────────────────────────────────────

LINEAR_SPEED  = 0.15   # m/s forward
ANGULAR_SPEED = 0.4    # rad/s turning
CENTER_THRESH = 0.15   # normalised horizontal error before moving forward
LOST_TIMEOUT  = 5.0    # s — declare cat lost after this
SPIN_DURATION = 4.0    # s — spin at each waypoint before moving on

# LiDAR motion detection
MOTION_DIST_THRESH  = 0.15   # m — point must move this much between scans
CLUSTER_MIN_POINTS  = 3      # minimum lidar points in a cluster
CAT_SIZE_MIN        = 0.10   # m — min cluster width
CAT_SIZE_MAX        = 0.60   # m — max cluster width
CAT_DIST_MAX        = 3.0    # m — only care about clusters within this range

# Fallback periodic camera sweep (when no LiDAR motion)
IDLE_SCAN_INTERVAL  = 8.0    # s — run camera check even if nothing is moving

# Known cats — add or rename as needed
KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

# ─────────────────────────────────────────────────────────────────────────────

WAYPOINTS = [
    ('L1',  0.726,  0.831),
    ('L2', -1.0,   -1.0),
    ('L3', -0.5,   -0.5),
    ('L4',  0.0,    0.0),
]


# ── small helpers ─────────────────────────────────────────────────────────────

def polar_to_xy(r, angle_rad):
    return r * math.cos(angle_rad), r * math.sin(angle_rad)


def cluster_width(points):
    """Rough bounding-box width of a list of (x,y) points."""
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return math.sqrt(dx**2 + dy**2)


def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


# ── node ──────────────────────────────────────────────────────────────────────

class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # ── state ─────────────────────────────────────────────────────────────
        self.target_cat     = None   # 'eevee' | 'raichu' | 'pichu' | None
        self.last_seen      = None   # rclpy.Time when cat was last confirmed
        self.cat_cx         = None   # normalised [0,1] horizontal pos in frame
        self.active         = False

        # Waypoint search state
        self.searching      = False
        self.spin_start     = None
        self.waypoint_index = 0
        self.nav_sent       = False

        # LiDAR
        self.prev_scan      = None   # previous LaserScan ranges for diff
        self.last_camera_check = 0.0  # wall-clock time of last camera trigger

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(String,       '/cat_target',   self.target_cb,   10)
        self.create_subscription(String,       '/cat_identity', self.identity_cb, 10)
        self.create_subscription(PointStamped, '/cat_position', self.position_cb, 10)
        self.create_subscription(LaserScan,    '/scan',         self.scan_cb,     10)

        # ── publishers ────────────────────────────────────────────────────────
        self.cmd_pub            = self.create_publisher(Twist,  '/cmd_vel',              10)
        self.camera_trigger_pub = self.create_publisher(String, '/camera_check_trigger', 10)

        # Control loop 10 Hz
        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f'SeekCat ready. Known cats: {KNOWN_CATS}. '
            'Waiting for /cat_target ...'
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def target_cb(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self.target_cat = None
            self.active     = False
            self.searching  = False
            self.stop_robot()
            self.get_logger().info('Seek stopped.')
            return

        if text not in KNOWN_CATS:
            self.get_logger().warn(
                f'Unknown cat "{text}". Known: {KNOWN_CATS}')
            return

        self.target_cat     = text
        self.active         = True
        self.searching      = False
        self.waypoint_index = 0
        self.nav_sent       = False
        self.last_seen      = None
        self.cat_cx         = None
        self.get_logger().info(f'Seeking: {self.target_cat}')

    def identity_cb(self, msg: String):
        """
        Receives confirmed cat identity from the vision pipeline.
        Expected format: 'eevee' | 'raichu' | 'pichu'
        """
        if self.target_cat and msg.data.strip().lower() == self.target_cat:
            self.last_seen = self.get_clock().now()
            if self.searching:
                self.get_logger().info(
                    f'Found {self.target_cat}! Switching to chase.')
                self.searching  = False
                self.nav_sent   = False
                self.spin_start = None
                self.stop_robot()

    def position_cb(self, msg: PointStamped):
        """Normalised [0,1] x position of cat in camera frame."""
        self.cat_cx    = msg.point.x
        self.last_seen = self.get_clock().now()

    def scan_cb(self, scan: LaserScan):
        """
        LiDAR callback.
        Diffs current scan against previous to find moving clusters.
        If a cat-sized moving cluster is found within range → trigger camera.
        """
        if not self.active:
            self.prev_scan = scan
            return

        if self.prev_scan is None:
            self.prev_scan = scan
            return

        # ── compute moving points ─────────────────────────────────────────────
        moving_pts = []
        n = min(len(scan.ranges), len(self.prev_scan.ranges))

        for i in range(n):
            r_now  = scan.ranges[i]
            r_prev = self.prev_scan.ranges[i]

            # Skip invalid readings
            if not (scan.range_min < r_now  < min(scan.range_max, CAT_DIST_MAX)):
                continue
            if not (scan.range_min < r_prev < min(scan.range_max, CAT_DIST_MAX)):
                continue

            # If range changed significantly → something moved
            if abs(r_now - r_prev) > MOTION_DIST_THRESH:
                angle = scan.angle_min + i * scan.angle_increment
                x, y  = polar_to_xy(r_now, angle)
                moving_pts.append((x, y))

        self.prev_scan = scan

        if not moving_pts:
            return  # nothing moving

        # ── cluster moving points ─────────────────────────────────────────────
        clusters = self._cluster_points(moving_pts, gap=0.3)

        for cluster in clusters:
            if len(cluster) < CLUSTER_MIN_POINTS:
                continue

            width = cluster_width(cluster)
            if CAT_SIZE_MIN <= width <= CAT_SIZE_MAX:
                self.get_logger().info(
                    f'[LiDAR] Cat-sized moving cluster: '
                    f'{len(cluster)} pts, width={width:.2f}m → triggering camera'
                )
                self._trigger_camera()
                return  # one trigger per scan cycle is enough

    # ── camera trigger ────────────────────────────────────────────────────────

    def _trigger_camera(self):
        """
        Publish a trigger so the vision node runs YOLOv8 and publishes
        the identity to /cat_identity.
        Rate-limited to avoid spamming.
        """
        now = time.time()
        if now - self.last_camera_check < 0.5:
            return  # max 2 triggers/sec
        self.last_camera_check = now
        msg = String()
        msg.data = self.target_cat or 'any'
        self.camera_trigger_pub.publish(msg)

    # ── control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.active or self.target_cat is None:
            return

        # ── periodic camera sweep even when nothing is moving ─────────────────
        now = time.time()
        if now - self.last_camera_check > IDLE_SCAN_INTERVAL:
            self.get_logger().debug('[Idle sweep] Triggering periodic camera check')
            self._trigger_camera()

        # ── cat visible recently → chase ──────────────────────────────────────
        if self.last_seen is not None:
            elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
            if elapsed < LOST_TIMEOUT:
                self.searching = False
                self.steer_toward_cat()
                return

        # ── cat lost → search ─────────────────────────────────────────────────
        self.searching = True
        self.waypoint_search()

    # ── steering ──────────────────────────────────────────────────────────────

    def steer_toward_cat(self):
        if self.cat_cx is None:
            return

        twist = Twist()
        error = self.cat_cx - 0.5   # negative=left, positive=right

        if abs(error) > CENTER_THRESH:
            twist.angular.z = -ANGULAR_SPEED * (error / 0.5)
            twist.linear.x  = 0.05
        else:
            twist.linear.x  = LINEAR_SPEED
            twist.angular.z = -ANGULAR_SPEED * (error / 0.5) * 0.3

        self.cmd_pub.publish(twist)

    # ── waypoint search ───────────────────────────────────────────────────────

    def waypoint_search(self):
        label, wx, wy = WAYPOINTS[self.waypoint_index]

        if not self.nav_sent:
            self.get_logger().info(
                f'[Search] {self.target_cat} not found — heading to {label} ({wx}, {wy})')
            self.send_nav_goal(wx, wy)
            self.nav_sent   = True
            self.spin_start = None
            return

        # Arrived — spin and look
        if self.spin_start is None:
            self.spin_start = self.get_clock().now()
            self.get_logger().info(f'[Search] Arrived at {label}, spinning...')

        elapsed_spin = (self.get_clock().now() - self.spin_start).nanoseconds / 1e9

        if elapsed_spin < SPIN_DURATION:
            twist = Twist()
            twist.angular.z = ANGULAR_SPEED
            self.cmd_pub.publish(twist)
            # Also trigger camera while spinning
            self._trigger_camera()
            return

        # Spin done, not found → next waypoint
        self.get_logger().info(
            f'[Search] {self.target_cat} not at {label}, trying next waypoint...')
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
        self.get_logger().info(f'[Nav2] Goal sent: ({x}, {y})')

    # ── clustering ────────────────────────────────────────────────────────────

    def _cluster_points(self, points, gap=0.3):
        """
        Simple single-linkage clustering.
        Returns list of clusters (each cluster = list of (x,y)).
        """
        if not points:
            return []

        clusters = []
        current  = [points[0]]

        for pt in points[1:]:
            # Find minimum distance to any point in current cluster
            min_dist = min(euclidean(pt, cp) for cp in current)
            if min_dist <= gap:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]

        clusters.append(current)
        return clusters

    # ── helpers ───────────────────────────────────────────────────────────────

    def stop_robot(self):
        self.cmd_pub.publish(Twist())


# ── main ──────────────────────────────────────────────────────────────────────

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