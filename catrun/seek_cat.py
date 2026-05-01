#!/usr/bin/env python3
"""
seek_cat.py
1. Detect moving objects with LiDAR
2. Turn camera (mounted at back) toward each object and check with YOLO
3. If not found after checking all objects → navigate to 3 waypoints and rotate
4. After 3 full search cycles with no results → send stop
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

LINEAR_SPEED    = 0.15
ANGULAR_SPEED   = 0.4
CENTER_THRESH   = 0.12
LOST_TIMEOUT    = 5.0
SPIN_DURATION   = 4.0

# Follow distance ~1 foot
FOLLOW_DIST_TARGET = 0.30
FOLLOW_DIST_MIN    = 0.20
FOLLOW_DIST_MAX    = 0.50
FOLLOW_SPEED_MAX   = 0.20

# LiDAR motion detection
MOTION_DIST_THRESH = 0.15
CLUSTER_MIN_POINTS = 3
CAT_SIZE_MIN       = 0.10
CAT_SIZE_MAX       = 0.60
CAT_DIST_MAX       = 3.0

# Camera is at BACK of robot → add 180° offset when turning toward object
CAMERA_OFFSET_RAD  = math.pi   # 180 degrees

# How long to wait for YOLO result after turning toward object
YOLO_WAIT_TIMEOUT  = 3.0   # seconds

# How long to rotate at each waypoint
WAYPOINT_SPIN_DURATION = 4.0

# Max full search cycles before sending stop
MAX_SEARCH_CYCLES  = 3

KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

WAYPOINTS = [
    ('L1',  0.898,  0.688),
    ('L2', -1.2,    1.0),
    ('L3',  0.297, -0.5),
]

# ── constants ─────────────────────────────────────────────────────────────────
CAMERA_HFOV_DEG    = 63.0                          # IMX477 horizontal FOV
CAMERA_HFOV_RAD    = math.radians(CAMERA_HFOV_DEG)
SPIN_STOPS         = math.ceil(360.0 / CAMERA_HFOV_DEG)  # = 6 stops
SPIN_ANGLE_RAD     = math.radians(360.0 / SPIN_STOPS)    # angle per stop
SPIN_PAUSE_SEC     = 2.0                           # pause at each stop
SPIN_TURN_SEC      = SPIN_ANGLE_RAD / ANGULAR_SPEED      # time to turn one stop


# ── helpers ───────────────────────────────────────────────────────────────────

def polar_to_xy(r, angle_rad):
    return r * math.cos(angle_rad), r * math.sin(angle_rad)

def cluster_width(points):
    if not points:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)

def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def angle_to_object(x, y):
    """Angle from robot to object in robot frame."""
    return math.atan2(y, x)

# ── node ──────────────────────────────────────────────────────────────────────

class SeekCat(Node):

    # ── states ────────────────────────────────────────────────────────────────
    STATE_IDLE          = 'idle'
    STATE_CHECK_OBJECTS = 'check_objects'   # turning toward moving objects
    STATE_TURNING       = 'turning'         # currently turning toward an object
    STATE_WAITING_YOLO  = 'waiting_yolo'    # waiting for YOLO result
    STATE_WAYPOINT_NAV  = 'waypoint_nav'    # navigating to waypoint
    STATE_WAYPOINT_SPIN = 'waypoint_spin'   # spinning at waypoint
    STATE_FOLLOWING     = 'following'       # following found cat

    def __init__(self):
        super().__init__('seek_cat')

        # ── core state ────────────────────────────────────────────────────────
        self.target_cat    = None
        self.active        = False
        self.state         = self.STATE_IDLE
        self.following     = False

        # Moving objects detected by LiDAR — list of (angle_rad, dist)
        self.moving_objects     = []
        self.object_check_index = 0   # which object we're currently checking

        # Turning state
        self.turn_target_angle  = None   # desired robot yaw offset
        self.turn_start_time    = None
        self.turn_duration      = 1.5    # seconds per 90° turn (tune this)

        # YOLO wait state
        self.yolo_wait_start    = None
        self.yolo_result        = None   # set by identity_cb

        # Waypoint search state
        self.waypoint_index     = 0
        self.nav_sent           = False
        self.spin_start         = None
        self.search_cycles      = 0      # counts full L1→L2→L3 cycles

        # Follow state
        self.last_seen          = None
        self.cat_cx             = None
        self.cat_dist           = None

        # LiDAR
        self.prev_scan          = None
        self.front_distance     = float('inf')
        self.last_scan_time     = 0.0

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
        self.target_pub         = self.create_publisher(String, '/cat_target',           10)
        self.status_pub         = self.create_publisher(String, '/seek_status',          10)

        self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f'SeekCat ready. Known: {KNOWN_CATS}. '
            'Waiting for /cat_target ...'
        )

    # ── callbacks ─────────────────────────────────────────────────────────────

    def target_cb(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._stop_all()
            return
        if text not in KNOWN_CATS:
            self.get_logger().warn(f'Unknown cat "{text}"')
            return

        self.target_cat      = text
        self.active          = True
        self.following       = False
        self.search_cycles   = 0
        self.waypoint_index  = 0
        self.nav_sent        = False
        self.last_seen       = None
        self.cat_cx          = None
        self.cat_dist        = None
        self.moving_objects  = []
        self.object_check_index = 0
        self.state           = self.STATE_CHECK_OBJECTS
        self.get_logger().info(f'Seeking: {self.target_cat}')
        self._publish_status(f'seeking {self.target_cat}')

    def identity_cb(self, msg: String):
        """Called when YOLO identifies a cat."""
        name = msg.data.strip().lower()
        self.yolo_result = name

        if self.target_cat and name == self.target_cat:
            self.last_seen = self.get_clock().now()
            self.get_logger().info(
                f'✅ Found {self.target_cat}! Switching to follow.')
            self.state     = self.STATE_FOLLOWING
            self.following = True
            self.nav_sent  = False
            self.stop_robot()
            self._publish_status(f'following {self.target_cat}')

    def position_cb(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.last_seen = self.get_clock().now()
        if not self.following and self.active:
            self.following = True
            self.state     = self.STATE_FOLLOWING
            self._publish_status(f'following {self.target_cat}')

    def scan_cb(self, scan: LaserScan):
        # Update front distance
        ranges = scan.ranges
        n = len(ranges)
        front_idx = list(range(0, n//12)) + list(range(11*n//12, n))
        front_vals = [ranges[i] for i in front_idx
                      if scan.range_min < ranges[i] < scan.range_max
                      and not math.isinf(ranges[i])]
        self.front_distance = min(front_vals) if front_vals else float('inf')

        if not self.active:
            self.prev_scan = scan
            return

        if self.prev_scan is None:
            self.prev_scan = scan
            return

        # Detect moving objects
        moving_pts = []
        n = min(len(scan.ranges), len(self.prev_scan.ranges))
        for i in range(n):
            r_now  = scan.ranges[i]
            r_prev = self.prev_scan.ranges[i]
            if not (scan.range_min < r_now  < min(scan.range_max, CAT_DIST_MAX)):
                continue
            if not (scan.range_min < r_prev < min(scan.range_max, CAT_DIST_MAX)):
                continue
            if abs(r_now - r_prev) > MOTION_DIST_THRESH:
                angle = scan.angle_min + i * scan.angle_increment
                x, y  = polar_to_xy(r_now, angle)
                moving_pts.append((x, y, angle, r_now))

        self.prev_scan = scan

        if not moving_pts:
            return

        # Cluster moving points
        pts_xy = [(p[0], p[1]) for p in moving_pts]
        clusters = self._cluster_points(pts_xy, gap=0.3)

        new_objects = []
        for cluster in clusters:
            if len(cluster) < CLUSTER_MIN_POINTS:
                continue
            width = cluster_width(cluster)
            if CAT_SIZE_MIN <= width <= CAT_SIZE_MAX:
                # Cluster center
                cx = sum(p[0] for p in cluster) / len(cluster)
                cy = sum(p[1] for p in cluster) / len(cluster)
                dist  = math.sqrt(cx**2 + cy**2)
                angle = math.atan2(cy, cx)
                new_objects.append((angle, dist))

        if new_objects and self.state == self.STATE_IDLE:
            # New moving objects detected — start checking them
            self.moving_objects     = new_objects
            self.object_check_index = 0
            self.state              = self.STATE_CHECK_OBJECTS
            self.get_logger().info(
                f'[LiDAR] {len(new_objects)} moving object(s) detected!')

        elif new_objects and self.state == self.STATE_CHECK_OBJECTS:
            # Update object list while checking
            self.moving_objects = new_objects

    # ── control loop ──────────────────────────────────────────────────────────

    def control_loop(self):
        if not self.active or self.target_cat is None:
            return

        # ── FOLLOWING ─────────────────────────────────────────────────────────
        if self.state == self.STATE_FOLLOWING:
            if self.last_seen is not None:
                elapsed = (self.get_clock().now() - self.last_seen).nanoseconds / 1e9
                if elapsed < LOST_TIMEOUT:
                    self.follow_cat()
                    return
                else:
                    self.get_logger().info(f'{self.target_cat} lost! Searching...')
                    self.following = False
                    self.state     = self.STATE_CHECK_OBJECTS
                    self.moving_objects     = []
                    self.object_check_index = 0
                    self._publish_status(f'lost {self.target_cat}')
            return

        # ── CHECK MOVING OBJECTS ───────────────────────────────────────────────
        if self.state == self.STATE_CHECK_OBJECTS:
            if self.object_check_index < len(self.moving_objects):
                # Turn toward next object
                angle, dist = self.moving_objects[self.object_check_index]
                # Camera is at back → turn robot so back faces the object
                turn_angle = angle + CAMERA_OFFSET_RAD
                self.get_logger().info(
                    f'[Check] Object {self.object_check_index+1}/'
                    f'{len(self.moving_objects)} at angle={math.degrees(angle):.1f}° '
                    f'dist={dist:.2f}m — turning camera toward it')
                self._start_turn(turn_angle)
                self.state = self.STATE_TURNING
            else:
                # No more objects to check → go to waypoints
                self.get_logger().info(
                    '[Check] All objects checked, none matched. '
                    'Navigating to waypoints...')
                self.state          = self.STATE_WAYPOINT_NAV
                self.nav_sent       = False

        # ── TURNING ───────────────────────────────────────────────────────────
        elif self.state == self.STATE_TURNING:
            self._execute_turn()

        # ── WAITING FOR YOLO ──────────────────────────────────────────────────
        elif self.state == self.STATE_WAITING_YOLO:
            self._wait_for_yolo()

        # ── WAYPOINT NAVIGATION ───────────────────────────────────────────────
        elif self.state == self.STATE_WAYPOINT_NAV:
            self._navigate_waypoint()

        # ── WAYPOINT SPINNING ─────────────────────────────────────────────────
        elif self.state == self.STATE_WAYPOINT_SPIN:
            self._spin_at_waypoint()

    # ── turning toward object ─────────────────────────────────────────────────

    def _start_turn(self, angle_rad):
        """Start turning robot by angle_rad."""
        self.turn_target_angle = angle_rad
        self.turn_start_time   = time.time()
        # Estimate turn duration based on angle
        self.turn_duration = abs(angle_rad) / ANGULAR_SPEED
        self.turn_duration = max(0.5, min(self.turn_duration, 4.0))
        self.yolo_result   = None

    def _execute_turn(self):
        """Execute the turn then trigger YOLO."""
        elapsed = time.time() - self.turn_start_time

        if elapsed < self.turn_duration:
            twist = Twist()
            # Turn direction based on angle sign
            if self.turn_target_angle > 0:
                twist.angular.z = ANGULAR_SPEED
            else:
                twist.angular.z = -ANGULAR_SPEED
            self.cmd_pub.publish(twist)
        else:
            # Turn complete — stop and trigger YOLO
            self.stop_robot()
            self.get_logger().info('[Turn] Complete — triggering YOLO check')
            self._trigger_camera()
            self.yolo_wait_start = time.time()
            self.yolo_result     = None
            self.state           = self.STATE_WAITING_YOLO

    def _wait_for_yolo(self):
        elapsed = time.time() - self.yolo_wait_start

        # Keep triggering camera during the wait
        if elapsed % 0.5 < 0.1:
            self._trigger_camera()

        if self.yolo_result is not None:
            if self.yolo_result == self.target_cat:
                return  # identity_cb already switched to FOLLOWING
            else:
                self.get_logger().info(
                    f'[YOLO] Got "{self.yolo_result}", '
                    f'not "{self.target_cat}" — next object')
                self.object_check_index += 1
                self.yolo_result = None
                self.state = self.STATE_CHECK_OBJECTS

        elif elapsed > SPIN_PAUSE_SEC:  # use same 2s pause
            self.get_logger().info(
                f'[YOLO] No result after {SPIN_PAUSE_SEC}s — next object')
            self.object_check_index += 1
            self.state = self.STATE_CHECK_OBJECTS

    # ── waypoint navigation ───────────────────────────────────────────────────

    def _navigate_waypoint(self):
        if self.waypoint_index >= len(WAYPOINTS):
            # Completed all waypoints — count as one cycle
            self.search_cycles  += 1
            self.waypoint_index  = 0
            self.get_logger().info(
                f'[Search] Cycle {self.search_cycles}/{MAX_SEARCH_CYCLES} complete.')

            if self.search_cycles >= MAX_SEARCH_CYCLES:
                self.get_logger().warn(
                    f'[Search] {MAX_SEARCH_CYCLES} cycles with no results — stopping.')
                self._send_stop()
                return

        label, wx, wy = WAYPOINTS[self.waypoint_index]

        if not self.nav_sent:
            self.get_logger().info(
                f'[Search] Heading to {label} ({wx}, {wy}) '
                f'[cycle {self.search_cycles+1}/{MAX_SEARCH_CYCLES}]')
            self._send_nav_goal(wx, wy)
            self.nav_sent   = True
            self.spin_start = None
            return

        # Nav sent — transition to spinning
        if self.spin_start is None:
            self.spin_start = self.get_clock().now()
            self.get_logger().info(f'[Search] Arrived at {label}, spinning...')
            self.state = self.STATE_WAYPOINT_SPIN

    def _spin_at_waypoint(self):
        """
        Rotate 360° in SPIN_STOPS steps.
        At each stop: pause SPIN_PAUSE_SEC and trigger YOLO.
        """
        now = time.time()

        # Initialise sub-state on first entry
        if not hasattr(self, '_spin_stop_idx'):
            self._spin_stop_idx   = 0
            self._spin_phase      = 'turning'   # 'turning' | 'pausing'
            self._spin_phase_start = now

        label = WAYPOINTS[self.waypoint_index][0]

        # ── turning phase ─────────────────────────────────────────────────────
        if self._spin_phase == 'turning':
            elapsed = now - self._spin_phase_start
            if elapsed < SPIN_TURN_SEC:
                twist = Twist()
                twist.angular.z = ANGULAR_SPEED
                self.cmd_pub.publish(twist)
            else:
                # Finished turning one stop — pause
                self.stop_robot()
                self._spin_phase       = 'pausing'
                self._spin_phase_start = now
                self.yolo_result       = None
                self._trigger_camera()
                self.get_logger().info(
                    f'[Spin] Stop {self._spin_stop_idx+1}/{SPIN_STOPS} '
                    f'at {label} — pausing {SPIN_PAUSE_SEC}s for YOLO')

        # ── pausing phase ─────────────────────────────────────────────────────
        elif self._spin_phase == 'pausing':
            elapsed = now - self._spin_phase_start

            # Check if YOLO found the target during pause
            if self.yolo_result == self.target_cat:
                self.get_logger().info(
                    f'[Spin] Found {self.target_cat} at stop '
                    f'{self._spin_stop_idx+1}!')
                self._reset_spin_state()
                # identity_cb already switches to FOLLOWING
                return

            if elapsed >= SPIN_PAUSE_SEC:
                self._spin_stop_idx += 1

                if self._spin_stop_idx >= SPIN_STOPS:
                    # Full 360° done — not found here
                    self.get_logger().info(
                        f'[Spin] Full 360° at {label} — {self.target_cat} not found')
                    self._reset_spin_state()
                    self.stop_robot()
                    self.waypoint_index += 1
                    self.nav_sent        = False
                    self.spin_start      = None
                    self.state           = self.STATE_WAYPOINT_NAV
                else:
                    # Next stop
                    self._spin_phase       = 'turning'
                    self._spin_phase_start = now

    def _reset_spin_state(self):
        if hasattr(self, '_spin_stop_idx'):
            del self._spin_stop_idx
        if hasattr(self, '_spin_phase'):
            del self._spin_phase
        if hasattr(self, '_spin_phase_start'):
            del self._spin_phase_start

    # ── follow behavior ───────────────────────────────────────────────────────

    def follow_cat(self):
        if self.cat_cx is None:
            return

        twist = Twist()
        error = self.cat_cx - 0.5

        # Angular steering — same direction
        twist.angular.z = -ANGULAR_SPEED * (error / 0.5)

        # Linear — camera faces BACK so move BACKWARD to follow
        dist = self.cat_dist if self.cat_dist is not None else self.front_distance

        if dist < FOLLOW_DIST_MIN:
            # Too close — move FORWARD (away from cat)
            twist.linear.x = 0.10
            self.get_logger().info(
                f'Too close ({dist:.2f}m) — moving forward',
                throttle_duration_sec=1.0)

        elif dist > FOLLOW_DIST_MAX:
            # Too far — move BACKWARD (toward cat)
            speed = min(FOLLOW_SPEED_MAX,
                        FOLLOW_SPEED_MAX * (dist - FOLLOW_DIST_TARGET) / 0.3)
            twist.linear.x = -speed if abs(error) < CENTER_THRESH else 0.0
            self.get_logger().info(
                f'Following backward ({dist:.2f}m)',
                throttle_duration_sec=1.0)

        elif FOLLOW_DIST_MIN <= dist <= FOLLOW_DIST_TARGET:
            # In sweet spot — slow backward approach
            twist.linear.x = -0.05 if abs(error) < CENTER_THRESH else 0.0

        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

    # ── nav goal ──────────────────────────────────────────────────────────────

    def _send_nav_goal(self, x, y):
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

    # ── camera trigger ────────────────────────────────────────────────────────

    def _trigger_camera(self):
        msg = String()
        msg.data = self.target_cat or 'any'
        self.camera_trigger_pub.publish(msg)

    # ── stop ──────────────────────────────────────────────────────────────────

    def _send_stop(self):
        """Send stop command to /cat_target."""
        self.get_logger().info('[Seek] Sending STOP — target not found.')
        self.target_pub.publish(String(data='stop'))
        self._stop_all()

    def _stop_all(self):
        self.target_cat = None
        self.active     = False
        self.following  = False
        self.state      = self.STATE_IDLE
        self.stop_robot()
        self.get_logger().info('Seek stopped.')
        self._publish_status('stopped')

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self, status: str):
        self.status_pub.publish(String(data=status))

    # ── clustering ────────────────────────────────────────────────────────────

    def _cluster_points(self, points, gap=0.3):
        if not points:
            return []
        clusters = []
        current  = [points[0]]
        for pt in points[1:]:
            if min(euclidean(pt, cp) for cp in current) <= gap:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]
        clusters.append(current)
        return clusters


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