#!/usr/bin/env python3
"""
seek_cat.py  –  patrol + follow
================================
State machine:
  IDLE    → waiting for /cat_target
  PATROL  → drive slowly along patrol path, YOLO runs continuously
  FOLLOW  → cat found → follow at 1 foot with back camera
  DONE    → 3 patrol laps with no result → send stop + return home

Key improvements vs spin version:
- No spinning — robot drives patrol path continuously
- YOLO runs every frame during patrol (no spin needed)
- Last known position memory — goes there first if cat lost
- LiDAR moving object detection still triggers camera check
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist, PointStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# ── tunables ──────────────────────────────────────────────────────────────────
# Follow
FOLLOW_DIST_TARGET = 0.30    # 1 foot in metres
FOLLOW_TOLERANCE   = 0.05    # dead zone
FOLLOW_SPEED_MAX   = 0.20    # m/s max backward speed
IMG_WIDTH          = 640     # camera resolution width
CAT_LOST_TIMEOUT   = 4.0     # seconds before declaring cat lost

# LiDAR motion detection
MOTION_DIST_THRESH = 0.15
CLUSTER_MIN_POINTS = 3
CAT_DIST_MAX       = 3.0

# Nav2
MAX_NAV_RETRIES    = 2
NAV_GOAL_TIMEOUT   = 30.0

# Search
MAX_PATROL_LAPS    = 3       # send stop after this many full patrol laps

KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

# Patrol path — robot drives through these continuously
PATROL_PATH = [
    ('L1',  0.898,  0.688),
    ('L2', -1.2,    1.0),
    ('L3',  0.297, -0.5),
    ('Home', 0.0,   0.0),
]

# ── states ────────────────────────────────────────────────────────────────────
IDLE    = 'IDLE'
PATROL  = 'PATROL'
FOLLOW  = 'FOLLOW'
DONE    = 'DONE'

# ── helpers ───────────────────────────────────────────────────────────────────
def polar_to_xy(r, angle_rad):
    return r * math.cos(angle_rad), r * math.sin(angle_rad)


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # ── core state ────────────────────────────────────────────────────────
        self.state      = IDLE
        self.target_cat = None
        self.active     = False

        # Patrol
        self.patrol_index    = 0
        self.patrol_laps     = 0
        self.nav_sent        = False
        self.nav_send_time   = None
        self.nav_goal_handle = None
        self.nav_retries     = 0
        self.nav_arrived     = False

        # Follow
        self.cat_cx          = None
        self.cat_dist        = None
        self.last_seen       = None
        self.last_known_x    = None   # last confirmed cat map position
        self.last_known_y    = None
        self.front_distance  = float('inf')

        # LiDAR
        self._prev_ranges    = None
        self._prev_scan_time = None
        self.moving_objects  = []

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(String,       '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,       '/cat_identity', self._cb_identity, 10)
        self.create_subscription(PointStamped, '/cat_position', self._cb_position, 10)
        self.create_subscription(LaserScan,    '/scan',         self._cb_scan,     10)

        # ── publishers ────────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',      10)
        self.status_pub = self.create_publisher(String, '/seek_status',  10)
        self.target_pub = self.create_publisher(String, '/cat_target',   10)

        self.create_timer(0.1, self._loop)
        self.get_logger().info('SeekCat ready – waiting for /cat_target')

    # ══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            self.get_logger().info('[Target] Seek stopped.')
            return
        if text in KNOWN_CATS:
            self.target_cat  = text
            self.active      = True
            self.patrol_laps = 0
            self.patrol_index = 0
            if self.state == IDLE:
                self._enter_patrol()
            self.get_logger().info(f'[Target] Seeking: {self.target_cat}')

    def _cb_identity(self, msg: String):
        name = msg.data.strip().lower()
        if self.target_cat and name == self.target_cat:
            if self.state == PATROL:
                self.get_logger().info(
                    f'[Identity] {name} confirmed during patrol!')
                self._enter_follow()

    def _cb_position(self, msg: PointStamped):
        """Normalised [0,1] cat position from cat_detector."""
        self.cat_cx   = msg.point.x
        self.cat_dist = msg.point.z if msg.point.z > 0 else None
        self.last_seen = self.get_clock().now()

        if self.state == PATROL:
            self.get_logger().info(
                f'[Position] Cat at cx={self.cat_cx:.2f} — switching to follow!')
            self._enter_follow()

    def _cb_scan(self, msg: LaserScan):
        now    = time.time()
        ranges = list(msg.ranges)
        n      = len(ranges)

        # Track front distance
        front_idx  = list(range(0, n//12)) + list(range(11*n//12, n))
        front_vals = [msg.ranges[i] for i in front_idx
                      if msg.range_min < msg.ranges[i] < msg.range_max
                      and not math.isinf(msg.ranges[i])]
        self.front_distance = min(front_vals) if front_vals else float('inf')

        # Motion detection
        if self._prev_ranges is None or len(self._prev_ranges) != n:
            self._prev_ranges    = ranges
            self._prev_scan_time = now
            return

        dt = now - (self._prev_scan_time or 0)
        if dt < 0.1:
            return

        moving = []
        for i, (r_new, r_old) in enumerate(zip(ranges, self._prev_ranges)):
            if not math.isfinite(r_new) or not math.isfinite(r_old):
                continue
            if r_new > CAT_DIST_MAX:
                continue
            if abs(r_new - r_old) > MOTION_DIST_THRESH:
                angle = msg.angle_min + i * msg.angle_increment
                moving.append((angle, r_new))

        self._prev_ranges    = ranges
        self._prev_scan_time = now

        if self.active:
            clusters = self._cluster(moving)
            if clusters:
                self.moving_objects = clusters

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        if self.state == IDLE or self.state == DONE:
            return
        if self.state == PATROL:
            self._loop_patrol()
        elif self.state == FOLLOW:
            self._loop_follow()

    # ══════════════════════════════════════════════════════════════════════════
    # State: PATROL
    # Robot drives continuously through patrol path
    # YOLO runs every frame via cat_detector — no spinning needed
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_patrol(self):
        self.state       = PATROL
        self.nav_sent    = False
        self.nav_arrived = False
        self.nav_retries = 0
        self.status_pub.publish(String(data=f'patrolling for {self.target_cat}'))
        self.get_logger().info(
            f'[Patrol] Starting patrol lap {self.patrol_laps+1}/{MAX_PATROL_LAPS}')

    def _loop_patrol(self):
        name, x, y = PATROL_PATH[self.patrol_index]

        # Send nav goal if not sent yet
        if not self.nav_sent:
            self.get_logger().info(
                f'[Patrol] → {name} ({x}, {y}) '
                f'[lap {self.patrol_laps+1}/{MAX_PATROL_LAPS}]')
            self._send_nav_goal(x, y, name)
            return

        # Goal timed out
        if time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT:
            self.get_logger().warn(f'[Patrol] Goal to {name} timed out — skipping')
            self._advance_patrol()
            return

        if self.nav_goal_handle is None:
            return  # waiting for handle

        # Check arrival
        status = self.nav_goal_handle.status
        if status == 4:  # SUCCEEDED
            self.get_logger().info(f'[Patrol] Reached {name}')
            self._advance_patrol()
        elif status in (5, 6):
            self.nav_retries += 1
            if self.nav_retries <= MAX_NAV_RETRIES:
                self.get_logger().warn(
                    f'[Patrol] Goal failed, retry {self.nav_retries}/{MAX_NAV_RETRIES}')
                self.nav_sent = False
            else:
                self.get_logger().warn(f'[Patrol] Skipping {name}')
                self._advance_patrol()

    def _advance_patrol(self):
        self.patrol_index += 1
        self.nav_sent      = False
        self.nav_retries   = 0
        self.nav_goal_handle = None

        if self.patrol_index >= len(PATROL_PATH):
            self.patrol_index = 0
            self.patrol_laps += 1
            self.get_logger().info(
                f'[Patrol] Lap {self.patrol_laps}/{MAX_PATROL_LAPS} complete')

            if self.patrol_laps >= MAX_PATROL_LAPS:
                self.get_logger().warn('[Patrol] Max laps — cat not found, stopping')
                self._enter_done()
                return

        self.get_logger().info(
            f'[Patrol] Next: {PATROL_PATH[self.patrol_index][0]}')

    # ══════════════════════════════════════════════════════════════════════════
    # State: FOLLOW
    # Follow cat at 1 foot using proportional control (ball_follower style)
    # Camera faces BACK → move backward to follow
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_follow(self):
        self.state = FOLLOW
        self._cancel_nav_goal()
        self._stop_motors()
        self.status_pub.publish(String(data=f'following {self.target_cat}'))
        self.get_logger().info(
            f'[Follow] Tracking {self.target_cat} at 1 foot')

    def _loop_follow(self):
        now = self.get_clock().now()

        # Check if cat still visible
        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > CAT_LOST_TIMEOUT:
                self.get_logger().info(
                    f'[Follow] Cat lost for {elapsed:.1f}s — resuming patrol')
                self.cat_cx = None

                # Go to last known position first if available
                if self.last_known_x is not None:
                    self.get_logger().info(
                        f'[Follow] Going to last known position '
                        f'({self.last_known_x:.2f}, {self.last_known_y:.2f})')
                    self._enter_patrol()
                    # Insert last known position as next patrol target
                    PATROL_PATH.insert(
                        self.patrol_index,
                        ('LastSeen', self.last_known_x, self.last_known_y))
                else:
                    self._enter_patrol()
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        twist = Twist()

        # ── Angular: proportional steering (ball_follower style) ──────────────
        ball_col        = self.cat_cx * IMG_WIDTH
        error           = ball_col - IMG_WIDTH / 2.0
        twist.angular.z = -error / 300.0

        # ── Linear: maintain 1 foot, move BACKWARD (camera faces back) ────────
        dist = self.cat_dist if self.cat_dist is not None else self.front_distance

        if dist is not None and not math.isinf(dist):
            dist_error = dist - FOLLOW_DIST_TARGET

            if dist_error > FOLLOW_TOLERANCE:
                # Too far → move backward toward cat
                twist.linear.x = -min(FOLLOW_SPEED_MAX, dist_error * 0.4)
            elif dist_error < -FOLLOW_TOLERANCE:
                # Too close → move forward away from cat
                twist.linear.x = max(0.10, abs(dist_error) * 0.4)
            else:
                twist.linear.x = 0.0

            self.get_logger().info(
                f'[Follow] dist={dist:.2f}m err={dist_error:.2f}m '
                f'cx={self.cat_cx:.2f} '
                f'lin={twist.linear.x:.2f} ang={twist.angular.z:.2f}',
                throttle_duration_sec=0.5)
        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

    # ══════════════════════════════════════════════════════════════════════════
    # Nav2 helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _send_nav_goal(self, x, y, name):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('[Nav] Nav2 not ready — waiting...')
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

        self.nav_sent        = True
        self.nav_send_time   = time.time()
        self.nav_goal_handle = None
        self.get_logger().info(f'[Nav] Goal sent: {name} ({x}, {y})')

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('[Nav] Goal rejected!')
            self.nav_sent = False
            return
        self.nav_goal_handle = handle
        self.get_logger().info('[Nav] Goal accepted')

    def _cancel_nav_goal(self):
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.nav_sent = False

    # ══════════════════════════════════════════════════════════════════════════
    # Done + utility
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_done(self):
        self.state = DONE
        self._cancel_nav_goal()
        self._stop_motors()
        self.status_pub.publish(String(data='stop'))
        self.target_pub.publish(String(data='stop'))
        self.get_logger().info('[DONE] Cat not found — sending stop and returning home')
        self._return_home()

    def _return_home(self):
        if not self.nav_client.server_is_ready():
            return
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = 0.0
        goal.pose.pose.position.y    = 0.0
        goal.pose.pose.orientation.w = 1.0
        self.nav_client.send_goal_async(goal)
        self.get_logger().info('[Nav] Returning home (0, 0)')

    def _stop_motors(self):
        self.cmd_pub.publish(Twist())

    def _reset(self):
        self._cancel_nav_goal()
        self._stop_motors()
        self.state         = IDLE
        self.active        = False
        self.target_cat    = None
        self.patrol_index  = 0
        self.patrol_laps   = 0
        self.moving_objects = []
        self.cat_cx        = None
        self.last_seen     = None

    def _cluster(self, points):
        if not points:
            return []
        clusters = []
        used     = [False] * len(points)
        for i, (a0, r0) in enumerate(points):
            if used[i]:
                continue
            group   = [(a0, r0)]
            used[i] = True
            x0, y0  = polar_to_xy(r0, a0)
            for j, (a1, r1) in enumerate(points):
                if used[j]:
                    continue
                x1, y1 = polar_to_xy(r1, a1)
                if math.hypot(x1-x0, y1-y0) < 0.25:
                    group.append((a1, r1))
                    used[j] = True
            if len(group) >= CLUSTER_MIN_POINTS:
                ca = sum(a for a, _ in group) / len(group)
                cr = sum(r for _, r in group) / len(group)
                clusters.append((ca, cr))
        return clusters


def main(args=None):
    rclpy.init(args=args)
    node = SeekCat()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_motors()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()