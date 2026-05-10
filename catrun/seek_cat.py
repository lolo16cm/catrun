#!/usr/bin/env python3
"""
seek_cat.py - Fixed
==================
Includes:
  - Rear-camera follow geometry (chase backward, retreat forward)
  - Continuous-spin object check (better detection performance than
    discrete-step in testing)
  - Trusts cat_detector's multi-frame confirmation - identity callback
    fires only on confirmed detections
  - Robust shutdown
"""

import math
import sys
import signal
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# tunables
ANGULAR_SPEED       = 0.3
LINEAR_SPEED        = 0.15

# Camera
CAMERA_HFOV_DEG     = 63.0
CAMERA_HFOV_RAD     = math.radians(CAMERA_HFOV_DEG)
CAMERA_OFFSET_RAD   = math.pi
YOLO_PAUSE_SEC      = 2.0

# Follow - keep about 1m from the cat (assignment spec)
FOLLOW_DIST_TARGET  = 1.0
FOLLOW_TOLERANCE    = 0.10
FOLLOW_SPEED_MAX    = 0.20
FOLLOW_KP_LIN       = 0.4
FOLLOW_KP_ANG       = 1.5
LOST_TIMEOUT        = 5.0

# LiDAR motion detection
MOTION_DIST_THRESH   = 0.25
CLUSTER_MIN_POINTS   = 8
CAT_SIZE_MIN         = 0.15
CAT_SIZE_MAX         = 0.50
CAT_DIST_MAX         = 2.5
OBJ_MIN_DIST         = 0.60
OBJ_MAX_CHECK        = 3
OBJECT_PERSIST_COUNT = 5

# Obstacle avoidance
SAFE_DISTANCE        = 0.25
AVOID_DISTANCE       = 0.35
AVOID_MIN_DURATION   = 3.0

# Search
MAX_SEARCH_CYCLES   = 3
NAV_GOAL_TIMEOUT    = 45.0

KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

WAYPOINTS = [
    ('L1',   -0.91,   0.33),
    ('L2',    2.87,   1.057),
    ('L3',    1.396,  1.127),
    ('L4',    1.1,   -1.037),
    ('L5',   -0.178, -1.31),
    ('home',  0.0,    0.0),
]

# states
IDLE        = 'IDLE'
NAVIGATE    = 'NAVIGATE'
CHECK_OBJ   = 'CHECK_OBJ'
YOLO_WAIT   = 'YOLO_WAIT'
FOLLOW      = 'FOLLOW'
AVOID       = 'AVOID'
DONE        = 'DONE'


def polar_to_xy(r, a):
    return r * math.cos(a), r * math.sin(a)


def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # Core
        self.state       = IDLE
        self.target_cat  = None
        self.active      = False

        # Navigation
        self.waypoint_index   = 0
        self.search_cycles    = 0
        self.nav_goal_handle  = None
        self.nav_sent         = False
        self.nav_send_time    = None
        self.nav_arrived      = False
        self.nav_retries      = 0
        self._last_goal_time  = 0.0

        # Object checking
        self.pending_objects    = []
        self.object_candidates  = {}
        self.current_obj_index  = 0
        self.turn_start         = None
        self.turn_sec           = 0.0
        self.turn_direction     = 1
        self.yolo_start         = None
        self.pre_check_state    = NAVIGATE
        self.pre_check_wp_index = 0

        # Follow
        self.cat_cx    = None
        self.cat_dist  = None
        self.last_seen = None

        # LiDAR
        self.prev_scan        = None
        self.front_distance   = float('inf')
        self.front_left_dist  = float('inf')
        self.front_right_dist = float('inf')
        self.left_distance    = float('inf')
        self.right_distance   = float('inf')
        self.back_distance    = float('inf')
        self.open_directions  = []

        # Avoid
        self.avoid_turn_dir  = 1
        self.avoid_start     = None
        self.pre_avoid_state = NAVIGATE

        # Camera trigger rate limit
        self.last_trigger_time = 0.0

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subs
        self.create_subscription(String,       '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,       '/cat_identity', self._cb_identity, 10)
        self.create_subscription(PointStamped, '/cat_position', self._cb_position, 10)
        self.create_subscription(LaserScan,    '/scan',         self._cb_scan,     10)

        # Pubs
        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',              10)
        self.cam_pub    = self.create_publisher(String, '/camera_check_trigger', 10)
        self.target_pub = self.create_publisher(String, '/cat_target',           10)
        self.status_pub = self.create_publisher(String, '/seek_status',          10)

        self.create_timer(0.1, self._loop)
        self.get_logger().info('SeekCat ready - rear-cam, continuous spin')

    # Callbacks
    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        if text not in KNOWN_CATS:
            return
        self.target_cat        = text
        self.active            = True
        self.search_cycles     = 0
        self.waypoint_index    = 0
        self.nav_sent          = False
        self.nav_arrived       = False
        self.pending_objects   = []
        self.object_candidates = {}
        self.state             = NAVIGATE
        self.get_logger().info(f'Seeking: {self.target_cat}')
        self._publish_status(f'seeking {self.target_cat}')

    def _cb_identity(self, msg: String):
        # cat_detector publishes /cat_identity only when 3 of last 5 frames
        # agree (multi-frame confirmation). So we trust it directly.
        name = msg.data.strip().lower()
        if not self.target_cat or name != self.target_cat:
            return
        if self.state in (NAVIGATE, CHECK_OBJ, YOLO_WAIT, AVOID):
            self.get_logger().info(
                f'YOLO confirmed {self.target_cat}! Switching to follow.')
            self._enter_follow()

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0 else None
        self.last_seen = self.get_clock().now()
        if self.state in (NAVIGATE, CHECK_OBJ, YOLO_WAIT, AVOID):
            self.get_logger().info(
                f'Cat position cx={self.cat_cx:.2f} - following!')
            self._enter_follow()

    def _cb_scan(self, msg: LaserScan):
        ranges = msg.ranges
        n      = len(ranges)

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if msg.range_min < ranges[i] < msg.range_max
                    and math.isfinite(ranges[i])]
            return min(vals) if vals else float('inf')

        self.front_distance   = safe_min(
            list(range(0, n//12)) + list(range(11*n//12, n)))
        self.front_left_dist  = safe_min(list(range(n//12, n//4)))
        self.front_right_dist = safe_min(list(range(3*n//4, 11*n//12)))
        self.left_distance    = safe_min(list(range(n//4,    5*n//12)))
        self.right_distance   = safe_min(list(range(7*n//12, 3*n//4)))
        self.back_distance    = safe_min(list(range(5*n//12, 7*n//12)))

        # Open directions
        open_dirs = []
        num_sectors = int(math.ceil(2 * math.pi / CAMERA_HFOV_RAD))
        for s in range(num_sectors):
            i_start = max(0, int(s * CAMERA_HFOV_RAD / msg.angle_increment))
            i_end   = min(n-1, int((s+1) * CAMERA_HFOV_RAD / msg.angle_increment))
            sector_vals = [ranges[i] for i in range(i_start, i_end+1)
                           if msg.range_min < ranges[i] < msg.range_max
                           and math.isfinite(ranges[i])]
            if sector_vals and min(sector_vals) > SAFE_DISTANCE:
                center = msg.angle_min + (s + 0.5) * CAMERA_HFOV_RAD
                open_dirs.append((center, min(sector_vals)))
        self.open_directions = open_dirs

        # Motion detection with persistence filter
        if not self.active or self.prev_scan is None:
            self.prev_scan = msg
            return

        if self.state == NAVIGATE and self.nav_sent and not self.nav_arrived:
            self.prev_scan = msg
            return

        moving_pts = []
        prev_ranges = self.prev_scan.ranges
        for i in range(min(n, len(prev_ranges))):
            r_now  = ranges[i]
            r_prev = prev_ranges[i]
            if not (msg.range_min < r_now < min(msg.range_max, CAT_DIST_MAX)):
                continue
            if not (msg.range_min < r_prev < min(msg.range_max, CAT_DIST_MAX)):
                continue
            if abs(r_now - r_prev) > MOTION_DIST_THRESH:
                angle = msg.angle_min + i * msg.angle_increment
                x, y  = polar_to_xy(r_now, angle)
                moving_pts.append((x, y, angle, r_now))

        self.prev_scan = msg

        if not moving_pts:
            self.object_candidates = {
                k: v-1 for k, v in self.object_candidates.items() if v > 1}
            return

        pts_xy   = [(p[0], p[1]) for p in moving_pts]
        clusters = self._cluster(pts_xy)

        new_candidates = {}
        for cluster in clusters:
            if len(cluster) < CLUSTER_MIN_POINTS:
                continue
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            w  = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
            if not (CAT_SIZE_MIN <= w <= CAT_SIZE_MAX):
                continue
            cx    = sum(xs)/len(xs)
            cy    = sum(ys)/len(ys)
            dist  = math.sqrt(cx**2 + cy**2)
            angle = math.atan2(cy, cx)
            if dist < OBJ_MIN_DIST:
                continue

            angle_key = round(math.degrees(angle) / 20) * 20
            prev_count = self.object_candidates.get(angle_key, 0)
            new_candidates[angle_key] = prev_count + 1

            if new_candidates[angle_key] >= OBJECT_PERSIST_COUNT:
                already_pending = any(
                    abs(math.degrees(a) - angle_key) < 25
                    for a, d in self.pending_objects)
                if not already_pending:
                    self.pending_objects.append((angle, dist))
                    self.get_logger().info(
                        f'[LiDAR] Confirmed moving object at '
                        f'{math.degrees(angle):.0f} deg dist={dist:.2f}m '
                        f'- interrupting now!')
                    self._interrupt_for_object()

        self.object_candidates = new_candidates

    # Main loop
    def _loop(self):
        if not self.active or self.target_cat is None:
            return

        if self.state not in (FOLLOW, AVOID, IDLE, DONE):
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'Obstacle {self.front_distance:.2f}m!')
                self._enter_avoid()
                return

        if self.state in (NAVIGATE, CHECK_OBJ, YOLO_WAIT):
            self._trigger_camera()

        if self.state == NAVIGATE:
            self._loop_navigate()
        elif self.state == CHECK_OBJ:
            self._loop_check_obj()
        elif self.state == YOLO_WAIT:
            self._loop_yolo_wait()
        elif self.state == FOLLOW:
            self._loop_follow()
        elif self.state == AVOID:
            self._loop_avoid()

    # NAVIGATE
    def _loop_navigate(self):
        label, wx, wy = WAYPOINTS[self.waypoint_index]

        if not self.nav_sent:
            if time.time() - self._last_goal_time < 1.0:
                return
            self._last_goal_time = time.time()
            self.get_logger().info(
                f'[Nav] -> {label} ({wx:.2f}, {wy:.2f}) '
                f'[cycle {self.search_cycles+1}/{MAX_SEARCH_CYCLES}]')
            self._send_nav_goal(wx, wy)
            return

        if self.nav_send_time is not None and \
                time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT:
            self.get_logger().warn(f'[Nav] Timeout to {label}')
            self._handle_nav_failure()
            return

        if not self.nav_arrived:
            self.get_logger().info(
                f'[Nav] Travelling to {label}...',
                throttle_duration_sec=5.0)
            return

        self.get_logger().info(
            f'[Nav] Arrived at {label} - moving to next waypoint')
        self._advance_waypoint()

    def _send_nav_goal(self, x, y):
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn('[Nav] Nav2 not ready - waiting...',
                                   throttle_duration_sec=3.0)
            self.nav_sent = False
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        self.nav_arrived     = False
        self.nav_sent        = True
        self.nav_send_time   = time.time()
        self.nav_goal_handle = None

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._nav_accepted_cb)
        self.get_logger().info(f'[Nav] Goal sent ({x:.2f}, {y:.2f})')

    def _nav_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('[Nav] Goal rejected!')
            self.nav_sent = False
            return
        self.nav_goal_handle = handle
        self.get_logger().info('[Nav] Goal accepted - navigating...')
        handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('[Nav] Physically arrived!')
            self.nav_arrived = True
        else:
            self.get_logger().warn(f'[Nav] Failed status={status}')
            self._handle_nav_failure()

    def _handle_nav_failure(self):
        self.nav_retries += 1
        if self.nav_retries <= 2:
            self.nav_sent = False
        else:
            self.get_logger().warn('[Nav] Max retries - skipping')
            self._advance_waypoint()

    def _advance_waypoint(self):
        self.waypoint_index += 1
        self.nav_sent        = False
        self.nav_arrived     = False
        self.nav_retries     = 0
        self.nav_goal_handle = None

        if self.waypoint_index >= len(WAYPOINTS):
            self.waypoint_index = 0
            self.search_cycles += 1
            self.get_logger().info(
                f'[Search] Cycle {self.search_cycles}/{MAX_SEARCH_CYCLES}')
            if self.search_cycles >= MAX_SEARCH_CYCLES:
                self._enter_done()
                return

        self.state = NAVIGATE

    def _cancel_nav(self):
        if self.nav_goal_handle is not None:
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.nav_goal_handle = None
        self.nav_sent    = False
        self.nav_arrived = False

    # CHECK_OBJ - continuous spin (better in testing than chunked)
    def _interrupt_for_object(self):
        if not self.pending_objects:
            return

        self.pending_objects = sorted(
            self.pending_objects, key=lambda o: o[1])[:OBJ_MAX_CHECK]

        self.get_logger().info(
            f'[Interrupt] Checking '
            f'{len(self.pending_objects)} confirmed object(s)!')

        self.pre_check_state    = self.state
        self.pre_check_wp_index = self.waypoint_index
        self._cancel_nav()
        self._stop_motors()
        self.state             = CHECK_OBJ
        self.current_obj_index = 0
        self._start_obj_turn(0)

    def _start_obj_turn(self, idx):
        if idx >= len(self.pending_objects):
            self.get_logger().info(
                '[Check] All objects checked - resuming navigation')
            self.pending_objects   = []
            self.object_candidates = {}
            self.waypoint_index    = self.pre_check_wp_index
            self.nav_sent          = False
            self.nav_arrived       = False
            self.state             = NAVIGATE
            return

        angle, dist  = self.pending_objects[idx]
        cam_angle    = normalize_angle(angle + CAMERA_OFFSET_RAD)
        turn_sec     = CAMERA_HFOV_RAD / ANGULAR_SPEED

        self.current_obj_index = idx
        self.turn_start        = time.time()
        self.turn_sec          = turn_sec
        self.turn_direction    = 1 if cam_angle >= 0 else -1

        self.get_logger().info(
            f'[Check] Object {idx+1}/{len(self.pending_objects)} '
            f'at {math.degrees(angle):.0f} deg dist={dist:.2f}m - '
            f'turning camera toward it')

    def _loop_check_obj(self):
        now     = time.time()
        elapsed = now - self.turn_start

        if elapsed < self.turn_sec:
            if self.front_distance < SAFE_DISTANCE:
                self._stop_motors()
                return
            twist = Twist()
            twist.angular.z = self.turn_direction * ANGULAR_SPEED
            self.cmd_pub.publish(twist)
        else:
            self._stop_motors()
            self._trigger_camera()
            self.yolo_start = now
            self.state      = YOLO_WAIT
            self.get_logger().info(
                f'[Check] Turned - waiting {YOLO_PAUSE_SEC}s for YOLO')

    def _loop_yolo_wait(self):
        elapsed = time.time() - self.yolo_start

        if elapsed % 0.5 < 0.11:
            self._trigger_camera()

        if elapsed >= YOLO_PAUSE_SEC:
            self.get_logger().info(
                f'[Check] Object {self.current_obj_index+1} - '
                f'not target - next')
            self.current_obj_index += 1
            self.state = CHECK_OBJ
            self._start_obj_turn(self.current_obj_index)

    # ──────────────────────────────────────────────────────────────────
    # FOLLOW - REAR-CAMERA GEOMETRY
    # The camera faces BACKWARD. So:
    #   - To CHASE (close gap):  drive BACKWARD (linear.x < 0)
    #   - To RETREAT (open gap): drive FORWARD  (linear.x > 0)
    # cat_cx > 0.5 means cat is on camera-right = robot's LEFT, turn LEFT.
    # ──────────────────────────────────────────────────────────────────

    def _enter_follow(self):
        self.state = FOLLOW
        self._cancel_nav()
        self._stop_motors()
        self.pending_objects   = []
        self.object_candidates = {}
        self.status_pub.publish(String(data=f'following {self.target_cat}'))
        self.get_logger().info(f'[Follow] Tracking {self.target_cat} '
                               f'(target dist {FOLLOW_DIST_TARGET}m)')

    def _loop_follow(self):
        now = self.get_clock().now()

        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > LOST_TIMEOUT:
                self.get_logger().info('[Follow] Cat lost - resuming search')
                self.cat_cx      = None
                self.cat_dist    = None
                self.nav_sent    = False
                self.nav_arrived = False
                self.state       = NAVIGATE
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        twist = Twist()

        # Angular: positive cx error -> cat on robot's LEFT -> turn LEFT
        cx_error = self.cat_cx - 0.5
        twist.angular.z = FOLLOW_KP_ANG * cx_error
        if twist.angular.z >  ANGULAR_SPEED: twist.angular.z =  ANGULAR_SPEED
        if twist.angular.z < -ANGULAR_SPEED: twist.angular.z = -ANGULAR_SPEED

        # Linear: only use cat_dist (camera depth)
        if self.cat_dist is None or math.isinf(self.cat_dist) \
                or math.isnan(self.cat_dist):
            twist.linear.x = 0.0
            self.cmd_pub.publish(twist)
            self.get_logger().info(
                f'[Follow] cx={self.cat_cx:.2f} no_depth ang={twist.angular.z:+.2f}',
                throttle_duration_sec=1.0)
            return

        dist       = self.cat_dist
        dist_error = dist - FOLLOW_DIST_TARGET

        if dist_error > FOLLOW_TOLERANCE:
            # Too far - chase by driving BACKWARD (rear-camera)
            if self.back_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'[Follow] Too far ({dist:.2f}m) but rear blocked '
                    f'({self.back_distance:.2f}m) - turning only',
                    throttle_duration_sec=1.0)
                twist.linear.x = 0.0
            else:
                speed = min(FOLLOW_SPEED_MAX, FOLLOW_KP_LIN * dist_error)
                twist.linear.x = -speed   # NEGATIVE = backward = toward cat

        elif dist_error < -FOLLOW_TOLERANCE:
            # Too close - retreat by driving FORWARD
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'[Follow] Too close ({dist:.2f}m) but front blocked '
                    f'({self.front_distance:.2f}m) - turning only',
                    throttle_duration_sec=1.0)
                twist.linear.x = 0.0
            else:
                speed = min(FOLLOW_SPEED_MAX, FOLLOW_KP_LIN * abs(dist_error))
                twist.linear.x = +speed   # POSITIVE = forward = away from cat

        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f'[Follow] cx={self.cat_cx:.2f} dist={dist:.2f}m '
            f'err={dist_error:+.2f}m lin={twist.linear.x:+.2f} '
            f'ang={twist.angular.z:+.2f} '
            f'(front={self.front_distance:.2f} back={self.back_distance:.2f})',
            throttle_duration_sec=0.5)

    # AVOID
    def _enter_avoid(self):
        if self.nav_sent and not self.nav_arrived:
            self.get_logger().warn(
                f'[Avoid] Obstacle {self.front_distance:.2f}m '
                f'but Nav2 is active - letting Nav2 handle it')
            return

        self.pre_avoid_state = self.state
        self.state           = AVOID
        self.avoid_start     = time.time()
        self._stop_motors()

        if self.open_directions:
            best = self.open_directions[0][0]
            self.avoid_turn_dir = 1 if best > 0 else -1
        elif self.front_right_dist > self.front_left_dist:
            self.avoid_turn_dir = -1
        else:
            self.avoid_turn_dir = 1

        self.get_logger().info(
            f'[Avoid] front={self.front_distance:.2f}m '
            f'-> {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"}')

    def _loop_avoid(self):
        elapsed = time.time() - self.avoid_start

        if elapsed < AVOID_MIN_DURATION:
            twist = Twist()
            if elapsed < 1.5:
                twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            else:
                twist.linear.x  = -0.08
                twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED * 0.5
            self.cmd_pub.publish(twist)
            return

        if self.front_distance > AVOID_DISTANCE:
            self.get_logger().info('[Avoid] Clear - resuming')
            self._stop_motors()
            self.nav_sent    = False
            self.nav_arrived = False
            self.state       = NAVIGATE
            return

        if elapsed < AVOID_MIN_DURATION * 2:
            twist = Twist()
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            self.cmd_pub.publish(twist)
        else:
            self.avoid_turn_dir = -self.avoid_turn_dir
            self.avoid_start    = time.time()
            self.get_logger().warn('[Avoid] Still stuck - flipping direction')

    # DONE
    def _enter_done(self):
        self.state = DONE
        self._cancel_nav()
        self._stop_motors()
        self.target_pub.publish(String(data='stop'))
        self._publish_status('stop')
        self.get_logger().info('[DONE] Not found - returning home')
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

    # Utilities
    def _trigger_camera(self):
        now = time.time()
        if now - self.last_trigger_time < 0.4:
            return
        self.last_trigger_time = now
        msg = String()
        msg.data = self.target_cat or 'any'
        self.cam_pub.publish(msg)

    def _stop_motors(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self, s):
        self.status_pub.publish(String(data=s))

    def _reset(self):
        self._cancel_nav()
        self._stop_motors()
        self.state             = IDLE
        self.active            = False
        self.target_cat        = None
        self.waypoint_index    = 0
        self.search_cycles     = 0
        self.pending_objects   = []
        self.object_candidates = {}
        self.cat_cx            = None
        self.cat_dist          = None
        self.last_seen         = None
        self._publish_status('stopped')

    def _cluster(self, points, gap=0.3):
        if not points:
            return []
        clusters = []
        current  = [points[0]]
        for pt in points[1:]:
            if min(math.hypot(pt[0]-c[0], pt[1]-c[1]) for c in current) <= gap:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]
        clusters.append(current)
        return clusters


def _safe_cleanup(node):
    """Cancel any in-flight Nav2 goal and stop motors."""
    if node is not None:
        try:
            node._cancel_nav()
        except Exception:
            pass
        try:
            node._stop_motors()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None

    def _sigterm_handler(signum, frame):
        _safe_cleanup(node)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        node = SeekCat()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[seek_cat] error: {e}", file=sys.stderr)
    finally:
        _safe_cleanup(node)
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("[seek_cat] clean shutdown complete")


if __name__ == '__main__':
    main()
