#!/usr/bin/env python3
"""
seek_cat.py  –  revised
========================
State machine:
  IDLE          → waiting for /cat_target
  SCAN_OBJECTS  → LiDAR found moving objects → turn camera (back-facing) toward
                  each one and wait for YOLO result
  NAVIGATE      → no moving objects (or all checked) → drive to next waypoint
  SPIN_AT_WP    → arrived at waypoint → spin 360° looking for cat
  FOUND         → target confirmed → stop everything and publish "found"
  DONE          → 3 full cycles with no result → publish "stop"

Key fixes vs previous version
-------------------------------
1. FOUND state cancels the active Nav2 goal and stops /cmd_vel immediately.
2. Camera-turn phase (SCAN_OBJECTS) directly publishes angular /cmd_vel to
   rotate the robot so the BACK-mounted camera faces the moving object, then
   waits up to YOLO_WAIT_TIMEOUT seconds for a /cat_identity message.
3. Nav2 retries are capped: each waypoint is retried at most MAX_NAV_RETRIES
   times before moving on, preventing the infinite status=6 loop.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, Twist

# ── tunables ──────────────────────────────────────────────────────────────────
ANGULAR_SPEED        = 0.4     # rad/s for camera-turn and waypoint spin
SPIN_DURATION        = 5.0     # seconds to spin at each waypoint
YOLO_WAIT_TIMEOUT    = 3.5     # seconds to wait for YOLO after turning to object
CAMERA_OFFSET_RAD    = math.pi # camera is at BACK → add 180° to face it at object

# After any motion stops, ignore YOLO results for this long so in-flight
# frames from the previous view are flushed before we accept a detection.
CAMERA_SETTLE_DELAY  = 1.0     # seconds  (tune up if you still see false misses)

# LiDAR motion detection
MOTION_DIST_THRESH   = 0.15    # metres change = "moving"
CLUSTER_MIN_POINTS   = 3
CAT_DIST_MAX         = 3.0     # ignore objects farther than this

# Nav2 reliability
MAX_NAV_RETRIES      = 2       # give up on a waypoint after this many status=6
NAV_GOAL_TIMEOUT     = 25.0    # seconds before declaring a nav goal timed-out

# Search cycles
MAX_SEARCH_CYCLES    = 3

# Follow behavior
FOLLOW_DIST_TARGET = 0.30   # 1 foot in metres
FOLLOW_TOLERANCE   = 0.05   # dead zone
FOLLOW_SPEED_MAX   = 0.20   # m/s max backward speed
IMG_WIDTH          = 640    # camera resolution width

KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

WAYPOINTS = [
    ('L1',  0.898,  0.688),
    ('L2', -1.2,    1.0),
    ('L3',  0.297, -0.5),
]

# ── states ────────────────────────────────────────────────────────────────────
IDLE         = 'IDLE'
SCAN_OBJECTS = 'SCAN_OBJECTS'
NAVIGATE     = 'NAVIGATE'
SPIN_AT_WP   = 'SPIN_AT_WP'
FOUND        = 'FOUND'
DONE         = 'DONE'
FOLLOW       = 'FOLLOW'

# ── helpers ───────────────────────────────────────────────────────────────────
def polar_to_xy(r, angle_rad):
    return r * math.cos(angle_rad), r * math.sin(angle_rad)

def angle_to_object(x, y):
    """Bearing from robot-centre to (x, y) in robot frame."""
    return math.atan2(y, x)


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # ── core state ────────────────────────────────────────────────────────
        self.state          = IDLE
        self.target_cat     = None
        self.active         = False

        # Moving-object inspection
        self.moving_objects     = []   # list of (angle_rad, dist) in robot frame
        self.obj_index          = 0    # which object we are currently checking
        self.turn_start_time    = None
        self.turn_target_angle  = None # radians to rotate (positive = CCW)
        self.turned_enough      = False
        self.camera_settle_start = None  # when motors stopped; flush stale frames
        self.yolo_wait_start    = None
        self.last_identity      = None

        # Waypoint navigation
        self.waypoint_index  = 0
        self.search_cycle    = 0
        self.nav_goal_handle = None
        self.nav_sent        = False
        self.nav_send_time   = None
        self.nav_retries     = 0

        # Spin-at-waypoint
        self.spin_start      = None

        # Previous LiDAR scan for motion detection
        self._prev_ranges    = None
        self._prev_scan_time = None

        # Init follow state variables
        self.cat_cx       = None
        self.cat_dist     = None
        self.last_seen    = None
        self.front_distance = float('inf')

        # ── Nav2 action client ────────────────────────────────────────────────
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # ── subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(String,    '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,    '/cat_identity', self._cb_identity, 10)
        self.create_subscription(LaserScan, '/scan',         self._cb_scan,     10)
        # Cat position from camera (normalised cx)
        from geometry_msgs.msg import PointStamped
        self.create_subscription(
            PointStamped, '/cat_position', self._cb_position, 10)

        # ── publications ──────────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel',     10)
        self.status_pub = self.create_publisher(String, '/seek_status', 10)

        # ── main loop at 10 Hz ────────────────────────────────────────────────
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
            self.target_cat = text
            self.active     = True
            if self.state == IDLE:
                self._enter_scan()
            self.get_logger().info(f'[Target] Seeking: {self.target_cat}')

    def _cb_identity(self, msg: String):
        """YOLO result from cat_detector – used during SCAN_OBJECTS."""
        self.last_identity = msg.data.strip().lower()

    def _cb_position(self, msg):
        """Normalised [0,1] cat position from cat_detector."""
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0 else None
        self.last_seen = self.get_clock().now()
        # If we're searching and cat found → switch to follow
        if self.state in (SCAN_OBJECTS, NAVIGATE, SPIN_AT_WP):
            self.get_logger().info(
                f'[Position] Cat detected at cx={self.cat_cx:.2f} — switching to follow!')
            self._enter_found()

    def _cb_scan(self, msg: LaserScan):
        """Detect moving objects by comparing consecutive scans."""
        now = time.time()
        ranges = list(msg.ranges)
        n      = len(ranges)

        if self._prev_ranges is None or len(self._prev_ranges) != n:
            self._prev_ranges    = ranges
            self._prev_scan_time = now
            return

        dt = now - self._prev_scan_time
        if dt < 0.1:          # wait at least 100 ms between comparisons
            return

        angle_min = msg.angle_min
        angle_inc = msg.angle_increment
        moving    = []

        for i, (r_new, r_old) in enumerate(zip(ranges, self._prev_ranges)):
            if not math.isfinite(r_new) or not math.isfinite(r_old):
                continue
            if r_new > CAT_DIST_MAX:
                continue
            if abs(r_new - r_old) > MOTION_DIST_THRESH:
                angle = angle_min + i * angle_inc
                moving.append((angle, r_new))

        self._prev_ranges    = ranges
        self._prev_scan_time = now

        # Cluster the moving points into distinct objects
        if self.state == IDLE or not self.active:
            return

        clusters = self._cluster(moving)
        if clusters:
            self.moving_objects = clusters   # update continuously

        # Track closest front obstacle for follow distance
        n = len(msg.ranges)
        front_idx = list(range(0, n//12)) + list(range(11*n//12, n))
        front_vals = [msg.ranges[i] for i in front_idx
                    if msg.range_min < msg.ranges[i] < msg.range_max
                    and not math.isinf(msg.ranges[i])]
        self.front_distance = min(front_vals) if front_vals else float('inf')

    # ══════════════════════════════════════════════════════════════════════════
    # Main control loop
    # ══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        if self.state == IDLE or self.state == DONE:
            return
        if self.state == FOUND:
            self._stop_motors()
            return
        # ── NEW: FOLLOW ───────────────────────────────────────────────────────
        if self.state == FOLLOW:
            self._loop_follow()
            return
        if self.state == SCAN_OBJECTS:
            self._loop_scan_objects()
            return
        if self.state == NAVIGATE:
            self._loop_navigate()
            return
        if self.state == SPIN_AT_WP:
            self._loop_spin_at_wp()
            return

    # ══════════════════════════════════════════════════════════════════════════
    # State: SCAN_OBJECTS
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_scan(self):
        if self.moving_objects:
            self.state     = SCAN_OBJECTS
            self.obj_index = 0
            self._start_turn_to_object(self.obj_index)
            self.get_logger().info(
                f'[Scan] {len(self.moving_objects)} moving object(s) detected – inspecting...')
        else:
            self._enter_navigate()

    def _start_turn_to_object(self, idx):
        """Begin rotating so the BACK camera faces objects[idx]."""
        if idx >= len(self.moving_objects):
            # No more objects → go to waypoints
            self._enter_navigate()
            return

        obj_angle, obj_dist = self.moving_objects[idx]
        # Camera is at the back → we need to rotate robot by (obj_angle + π)
        # Normalise to (-π, π]
        turn = (obj_angle + CAMERA_OFFSET_RAD + math.pi) % (2 * math.pi) - math.pi

        self.turn_target_angle   = turn
        self.turn_start_time     = time.time()
        self.turned_enough       = False
        self.camera_settle_start = None
        self.yolo_wait_start     = None
        self.last_identity       = None

        self.get_logger().info(
            f'[Scan] Turning {math.degrees(turn):.1f}° to face object {idx+1} '
            f'(dist={obj_dist:.2f} m) with back camera')

    def _loop_scan_objects(self):
        """
        Phase 1: rotate until we've turned ~turn_target_angle.
        Phase 2: settle – motors stopped, discard stale camera frames for
                 CAMERA_SETTLE_DELAY seconds so the pipeline catches up.
        Phase 3: accept YOLO result for up to YOLO_WAIT_TIMEOUT seconds.
        """
        now = time.time()

        if not self.turned_enough:
            elapsed    = now - self.turn_start_time
            target     = abs(self.turn_target_angle)
            est_turned = elapsed * ANGULAR_SPEED

            direction = 1.0 if self.turn_target_angle >= 0 else -1.0
            if est_turned < target:
                twist = Twist()
                twist.angular.z = direction * ANGULAR_SPEED
                self.cmd_pub.publish(twist)
                return
            else:
                # Motors stop → enter settle phase
                self._stop_motors()
                self.turned_enough       = True
                self.camera_settle_start = now
                self.last_identity       = None  # discard anything queued before stop
                self.get_logger().info(
                    f'[Scan] Turned – settling {CAMERA_SETTLE_DELAY}s for camera...')
                return

        # Phase 2: settle – flush stale identities every tick until delay expires
        if now - self.camera_settle_start < CAMERA_SETTLE_DELAY:
            self.last_identity = None
            return

        # Phase 3: YOLO wait (start timer once, after settle completes)
        if self.yolo_wait_start is None:
            self.yolo_wait_start = now
            self.get_logger().info('[Scan] Camera settled – waiting for YOLO...')

        yolo_elapsed = now - self.yolo_wait_start

        if self.last_identity and self.target_cat in self.last_identity:
            self.get_logger().info(
                f'[Scan] TARGET FOUND: {self.last_identity}')
            self._enter_found()
            return

        if yolo_elapsed >= YOLO_WAIT_TIMEOUT:
            self.get_logger().info(
                f'[Scan] Object {self.obj_index+1}: '
                f'not target (got: {self.last_identity or "none"}) – next object')
            self.obj_index += 1
            if self.obj_index < len(self.moving_objects):
                self._start_turn_to_object(self.obj_index)
            else:
                self.get_logger().info('[Scan] All objects checked – going to waypoints')
                self._enter_navigate()

    # ══════════════════════════════════════════════════════════════════════════
    # State: NAVIGATE
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_navigate(self):
        self.state       = NAVIGATE
        self.nav_sent    = False
        self.nav_retries = 0

    def _loop_navigate(self):
        name, x, y = WAYPOINTS[self.waypoint_index]

        # Check for moving objects first (interrupt navigation)
        if self.moving_objects and not self.nav_sent:
            self.get_logger().info('[Nav] Moving objects detected – pausing nav to inspect')
            self._cancel_nav_goal()
            self._enter_scan()
            return

        # Send the goal if not yet sent
        if not self.nav_sent:
            self.get_logger().info(
                f'[Search] Heading to {name} ({x}, {y}) '
                f'[cycle {self.search_cycle+1}/{MAX_SEARCH_CYCLES}]')
            self._send_nav_goal(x, y, name)
            return

        # Goal timed out?
        if time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT:
            self.get_logger().warn(f'[Nav] Goal to {name} timed out – treating as failed')
            self._handle_nav_failure(name, x, y)
            return

        # Poll goal status
        if self.nav_goal_handle is None:
            return   # still waiting for handle

        status = self.nav_goal_handle.status
        # GoalStatus: 4=SUCCEEDED, 5=CANCELED, 6=ABORTED
        if status == 4:  # SUCCEEDED
            self.get_logger().info(f'[Nav] Arrived at {name}')
            self._enter_spin_at_wp()
        elif status in (5, 6):
            self.get_logger().warn(f'[Nav] Goal failed status={status} for {name}')
            self._handle_nav_failure(name, x, y)

    def _handle_nav_failure(self, name, x, y):
        self.nav_retries += 1
        if self.nav_retries <= MAX_NAV_RETRIES:
            self.get_logger().info(
                f'[Nav] Retrying {name} (attempt {self.nav_retries}/{MAX_NAV_RETRIES})')
            self.nav_sent = False
        else:
            self.get_logger().warn(
                f'[Nav] Giving up on {name} after {MAX_NAV_RETRIES} retries – skipping')
            self._advance_waypoint()

    def _send_nav_goal(self, x, y, name):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('[Nav] Nav2 not ready – waiting...')
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)

        self.nav_sent      = True
        self.nav_send_time = time.time()
        self.nav_goal_handle = None
        self.get_logger().info(f'[Nav] Goal sent: ({x}, {y})')

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('[Nav] Goal rejected!')
            self.nav_sent = False
            return
        self.nav_goal_handle = handle
        self.get_logger().info('[Nav] Goal accepted, navigating...')

    def _cancel_nav_goal(self):
        if self.nav_goal_handle is not None:
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.nav_sent = False

    # ══════════════════════════════════════════════════════════════════════════
    # State: SPIN_AT_WP
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_spin_at_wp(self):
        self.state               = SPIN_AT_WP
        self.spin_start          = time.time()
        self.camera_settle_start = time.time()  # flush frames from navigation
        self.last_identity       = None          # discard anything queued in transit
        name = WAYPOINTS[self.waypoint_index][0]
        self.get_logger().info(
            f'[Spin] Arrived at {name} – settling {CAMERA_SETTLE_DELAY}s then spinning...')

    def _loop_spin_at_wp(self):
        now     = time.time()
        elapsed = now - self.spin_start

        # Settle window: robot is still, flush stale camera frames
        if now - self.camera_settle_start < CAMERA_SETTLE_DELAY:
            self.last_identity = None  # keep discarding until pipeline catches up
            return                     # don't spin yet

        # Check if the target was identified while spinning
        if self.last_identity and self.target_cat in self.last_identity:
            self.get_logger().info('[Spin] TARGET FOUND while spinning!')
            self._enter_found()
            return

        # Spin for SPIN_DURATION (elapsed counts from arrival, settle included)
        if elapsed < SPIN_DURATION + CAMERA_SETTLE_DELAY:
            twist = Twist()
            twist.angular.z = ANGULAR_SPEED
            self.cmd_pub.publish(twist)
        else:
            self._stop_motors()
            self._advance_waypoint()

    def _advance_waypoint(self):
        self.waypoint_index += 1
        if self.waypoint_index >= len(WAYPOINTS):
            self.waypoint_index = 0
            self.search_cycle  += 1
            self.get_logger().info(
                f'[Search] Completed cycle {self.search_cycle}/{MAX_SEARCH_CYCLES}')

        if self.search_cycle >= MAX_SEARCH_CYCLES:
            self.get_logger().warn('[Search] Max cycles reached – sending stop')
            self._enter_done()
        else:
            self._enter_navigate()

    # ══════════════════════════════════════════════════════════════════════════
    # State transitions: FOUND / DONE
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_found(self):
        self.state = FOLLOW
        self._cancel_nav_goal()
        self._stop_motors()
        self.cat_cx       = None
        self.cat_dist     = None
        self.last_seen    = self.get_clock().now()
        self.status_pub.publish(String(data='following'))
        self.get_logger().info(
            f'[FOLLOW] Target "{self.target_cat}" confirmed – following at 1 foot.')

    def _loop_follow(self):
        """
        Follow cat at ~1 foot using ball_follower proportional control.
        Camera faces BACK → move backward to follow.
        """
        now = self.get_clock().now()

        # Check if cat still visible
        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > 5.0:
                self.get_logger().info('[FOLLOW] Cat lost — resuming search.')
                self.state          = SCAN_OBJECTS
                self.moving_objects = []
                self._enter_scan()
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        twist = Twist()

        # ── Angular: proportional steering (same as ball_follower) ────────────
        ball_col        = self.cat_cx * IMG_WIDTH
        error           = ball_col - IMG_WIDTH / 2.0
        twist.angular.z = -error / 300.0

        # ── Linear: maintain 1 foot distance, move BACKWARD ───────────────────
        dist = self.cat_dist if (
            hasattr(self, 'cat_dist') and self.cat_dist is not None
        ) else getattr(self, 'front_distance', None)

        if dist is not None and not math.isinf(dist):
            dist_error = dist - FOLLOW_DIST_TARGET

            if dist_error > FOLLOW_TOLERANCE:
                # Too far → backward toward cat
                twist.linear.x = -min(FOLLOW_SPEED_MAX, dist_error * 0.4)
            elif dist_error < -FOLLOW_TOLERANCE:
                # Too close → forward away from cat
                twist.linear.x = max(0.10, abs(dist_error) * 0.4)
            else:
                twist.linear.x = 0.0

            self.get_logger().info(
                f'[FOLLOW] dist={dist:.2f}m err={dist_error:.2f}m '
                f'cx={self.cat_cx:.2f} lin={twist.linear.x:.2f} '
                f'ang={twist.angular.z:.2f}',
                throttle_duration_sec=0.5)
        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

    def _enter_done(self):
        self.state = DONE
        self._cancel_nav_goal()
        self._stop_motors()
        self.status_pub.publish(String(data='stop'))
        self.get_logger().info('[DONE] No target found after max cycles.')

    # ══════════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════════

    def _stop_motors(self):
        self.cmd_pub.publish(Twist())

    def _reset(self):
        self._cancel_nav_goal()
        self._stop_motors()
        self.state          = IDLE
        self.active         = False
        self.target_cat     = None
        self.waypoint_index = 0
        self.search_cycle   = 0
        self.moving_objects = []
        self.last_identity  = None

    def _cluster(self, points):
        """Simple distance-based clustering → returns list of (centroid_angle, centroid_dist)."""
        if not points:
            return []
        clusters = []
        used     = [False] * len(points)
        for i, (a0, r0) in enumerate(points):
            if used[i]:
                continue
            group = [(a0, r0)]
            used[i] = True
            x0, y0 = polar_to_xy(r0, a0)
            for j, (a1, r1) in enumerate(points):
                if used[j]:
                    continue
                x1, y1 = polar_to_xy(r1, a1)
                if math.hypot(x1 - x0, y1 - y0) < 0.25:
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
        node.stop_robot() if hasattr(node, 'stop_robot') else node._stop_motors()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()