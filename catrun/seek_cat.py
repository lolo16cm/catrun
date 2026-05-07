#!/usr/bin/env python3
"""
seek_cat.py - Fixed version
============================
Fixes:
1. Nav2 arrival detection using result callback not status polling
2. Camera triggered continuously during search not just at spin stops  
3. Moving objects interrupt navigation and get checked immediately
4. Replaced spinning with LiDAR-guided directional sweep
5. Obstacle avoidance built into all movement
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
ANGULAR_SPEED       = 0.3    # rad/s — slow for reliable camera detection
LINEAR_SPEED        = 0.15   # m/s

# Camera
CAMERA_HFOV_DEG     = 63.0
CAMERA_HFOV_RAD     = math.radians(CAMERA_HFOV_DEG)
CAMERA_OFFSET_RAD   = math.pi   # camera faces back
YOLO_PAUSE_SEC      = 2.5        # pause at each direction for YOLO
NUM_SWEEP_STOPS     = math.ceil(360.0 / CAMERA_HFOV_DEG)  # 6 stops

# Follow
FOLLOW_DIST_TARGET  = 0.30
FOLLOW_DIST_MIN     = 0.20
FOLLOW_DIST_MAX     = 0.50
FOLLOW_SPEED_MAX    = 0.20
CENTER_THRESH       = 0.12
LOST_TIMEOUT        = 5.0

# Obstacle avoidance
SAFE_DISTANCE       = 0.35   # m — stop immediately
AVOID_DISTANCE      = 0.50   # m — start slowing

# LiDAR motion detection
MOTION_DIST_THRESH  = 0.15
CLUSTER_MIN_POINTS  = 3
CAT_SIZE_MIN        = 0.10
CAT_SIZE_MAX        = 0.60
CAT_DIST_MAX        = 3.0

# Search
MAX_SEARCH_CYCLES   = 3
NAV_GOAL_TIMEOUT    = 30.0   # s

KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

WAYPOINTS = [
    ('L1', -0.91,   0.33),
    ('L2',  2.87,   1.057),
    ('L3',  1.396,  1.127),
    ('L4',  1.1,   -1.037),
    ('L5', -0.178, -1.31),
]

# ── states ────────────────────────────────────────────────────────────────────
IDLE     = 'IDLE'
NAVIGATE = 'NAVIGATE'
SWEEP    = 'SWEEP'
FOLLOW   = 'FOLLOW'
AVOID    = 'AVOID'
DONE     = 'DONE'


def polar_to_xy(r, a):
    return r * math.cos(a), r * math.sin(a)

def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # Core
        self.state          = IDLE
        self.target_cat     = None
        self.active         = False

        # Navigation
        self.waypoint_index     = 0
        self.search_cycles      = 0
        self.nav_goal_handle    = None
        self.nav_sent           = False
        self.nav_send_time      = None
        self.nav_arrived        = False
        self.nav_retries        = 0
        self.MAX_NAV_RETRIES    = 2

        # Sweep at waypoint
        # Instead of spinning blindly, sweep based on LiDAR open corridors
        self.sweep_stop         = 0
        self.sweep_phase        = 'turning'   # 'turning' | 'pausing'
        self.sweep_phase_start  = None
        self.sweep_directions   = []   # angles to check (from LiDAR open space)
        self.sweep_turn_start   = None
        self.sweep_turned_sec   = 0.0

        # Moving object check
        self.moving_objects         = []
        self.checking_object        = False
        self.object_check_index     = 0
        self.object_turn_start      = None
        self.object_turn_sec        = 0.0
        self.object_turned          = False
        self.object_yolo_start      = None
        self.last_yolo_result       = None

        # Follow
        self.cat_cx             = None
        self.cat_dist           = None
        self.last_seen          = None

        # LiDAR
        self.prev_scan          = None
        self.front_distance     = float('inf')
        self.front_left_dist    = float('inf')
        self.front_right_dist   = float('inf')
        self.left_distance      = float('inf')
        self.right_distance     = float('inf')
        self.open_directions    = []   # angles with clear space

        # Avoid
        self.avoid_turn_dir     = 1
        self.avoid_start        = None
        self.pre_avoid_state    = NAVIGATE

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
        self.get_logger().info('SeekCat ready')

    # ══════════════════════════════════════════════════════════════════════════
    # Callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        if text not in KNOWN_CATS:
            return
        self.target_cat     = text
        self.active         = True
        self.search_cycles  = 0
        self.waypoint_index = 0
        self.nav_sent       = False
        self.nav_arrived    = False
        self.moving_objects = []
        self.last_yolo_result = None
        self.state          = NAVIGATE
        self.get_logger().info(f'Seeking: {self.target_cat}')
        self._publish_status(f'seeking {self.target_cat}')

    def _cb_identity(self, msg: String):
        name = msg.data.strip().lower()
        self.last_yolo_result = name
        if self.target_cat and name == self.target_cat:
            if self.state in (NAVIGATE, SWEEP, AVOID):
                self.get_logger().info(f'✅ FOUND {self.target_cat}!')
                self._enter_follow()

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0 else None
        self.last_seen = self.get_clock().now()
        if self.state in (NAVIGATE, SWEEP, AVOID):
            self.get_logger().info(
                f'Cat position received cx={self.cat_cx:.2f} — following!')
            self._enter_follow()

    def _cb_scan(self, msg: LaserScan):
        ranges = msg.ranges
        n      = len(ranges)

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if msg.range_min < ranges[i] < msg.range_max
                    and math.isfinite(ranges[i])]
            return min(vals) if vals else float('inf')

        # Sector distances
        self.front_distance   = safe_min(list(range(0, n//12)) + list(range(11*n//12, n)))
        self.front_left_dist  = safe_min(list(range(n//12, n//4)))
        self.front_right_dist = safe_min(list(range(3*n//4, 11*n//12)))
        self.left_distance    = safe_min(list(range(n//4, 5*n//12)))
        self.right_distance   = safe_min(list(range(7*n//12, 3*n//4)))

        # Find open directions — divide 360° into FOV sectors
        open_dirs = []
        num_sectors = int(math.ceil(2 * math.pi / CAMERA_HFOV_RAD))
        for s in range(num_sectors):
            sector_center = msg.angle_min + (s + 0.5) * CAMERA_HFOV_RAD
            i_start = max(0,   int(s * CAMERA_HFOV_RAD / msg.angle_increment))
            i_end   = min(n-1, int((s+1) * CAMERA_HFOV_RAD / msg.angle_increment))
            sector_vals = [ranges[i] for i in range(i_start, i_end+1)
                           if msg.range_min < ranges[i] < msg.range_max
                           and math.isfinite(ranges[i])]
            if sector_vals and min(sector_vals) > SAFE_DISTANCE:
                open_dirs.append((sector_center, min(sector_vals)))

        self.open_directions = open_dirs

        # Motion detection
        if not self.active or self.prev_scan is None:
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
            return

        # Cluster
        pts_xy   = [(p[0], p[1]) for p in moving_pts]
        clusters = self._cluster(pts_xy)
        new_objs = []
        for cluster in clusters:
            if len(cluster) < CLUSTER_MIN_POINTS:
                continue
            xs = [p[0] for p in cluster]
            ys = [p[1] for p in cluster]
            w  = math.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)
            if CAT_SIZE_MIN <= w <= CAT_SIZE_MAX:
                cx    = sum(xs) / len(xs)
                cy    = sum(ys) / len(ys)
                dist  = math.sqrt(cx**2 + cy**2)
                angle = math.atan2(cy, cx)
                # Only add if in open/safe direction
                if dist > SAFE_DISTANCE:
                    new_objs.append((angle, dist))

        if new_objs:
            self.moving_objects = new_objs
            self.get_logger().info(
                f'[LiDAR] {len(new_objs)} moving object(s) — '
                f'will check after current action')

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop
    # ══════════════════════════════════════════════════════════════════════════

    def _loop(self):
        if not self.active or self.target_cat is None:
            return

        # ── Obstacle check — always active except during follow ───────────────
        if self.state not in (FOLLOW, AVOID, IDLE, DONE):
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'⚠️ Obstacle {self.front_distance:.2f}m — avoiding!')
                self._enter_avoid()
                return

        if self.state == NAVIGATE:
            self._loop_navigate()
        elif self.state == SWEEP:
            self._loop_sweep()
        elif self.state == FOLLOW:
            self._loop_follow()
        elif self.state == AVOID:
            self._loop_avoid()

    # ══════════════════════════════════════════════════════════════════════════
    # NAVIGATE — drive to waypoint using Nav2
    # ══════════════════════════════════════════════════════════════════════════

    def _loop_navigate(self):
        label, wx, wy = WAYPOINTS[self.waypoint_index]

        # Check moving objects — pause nav and check them
        if self.moving_objects and not self.nav_arrived:
            self.get_logger().info(
                f'[Nav] Moving object detected — checking before continuing')
            self._cancel_nav()
            self._enter_sweep(check_objects_first=True)
            return

        if not self.nav_sent:
            self.get_logger().info(
                f'[Nav] → {label} ({wx:.2f}, {wy:.2f}) '
                f'[cycle {self.search_cycles+1}/{MAX_SEARCH_CYCLES}]')
            self._send_nav_goal(wx, wy)
            return

        # Timeout check
        if time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT:
            self.get_logger().warn(f'[Nav] Timeout to {label}')
            self._handle_nav_failure()
            return

        # Waiting for arrival callback
        if not self.nav_arrived:
            self.get_logger().info(
                f'[Nav] Travelling to {label}...',
                throttle_duration_sec=3.0)
            return

        # Arrived — do LiDAR sweep at this waypoint
        self.get_logger().info(f'[Nav] ✅ Arrived at {label} — starting sweep')
        self._enter_sweep()

    def _send_nav_goal(self, x, y):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('[Nav] Nav2 not ready')
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        self.nav_arrived   = False
        self.nav_sent      = True
        self.nav_send_time = time.time()
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
        self.get_logger().info('[Nav] Goal accepted — navigating...')
        handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        """Called ONLY when robot physically arrives or fails."""
        status = future.result().status
        if status == 4:  # SUCCEEDED
            self.get_logger().info('[Nav] ✅ Physically arrived!')
            self.nav_arrived = True
        else:
            self.get_logger().warn(f'[Nav] Failed status={status}')
            self._handle_nav_failure()

    def _handle_nav_failure(self):
        self.nav_retries += 1
        if self.nav_retries <= self.MAX_NAV_RETRIES:
            self.get_logger().info(
                f'[Nav] Retry {self.nav_retries}/{self.MAX_NAV_RETRIES}')
            self.nav_sent = False
        else:
            self.get_logger().warn('[Nav] Max retries — skipping waypoint')
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
            self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
        self.nav_sent    = False
        self.nav_arrived = False

    # ══════════════════════════════════════════════════════════════════════════
    # SWEEP — LiDAR-guided directional search
    # Checks open corridors detected by LiDAR instead of blind spinning
    # Continuously triggers camera at each stop
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_sweep(self, check_objects_first=False):
        self.state      = SWEEP
        self.sweep_stop = 0
        self._stop_motors()

        # Build sweep directions:
        # 1. Moving objects first (highest priority)
        # 2. Then open corridors from LiDAR
        # 3. Fallback: evenly spaced directions
        directions = []

        if check_objects_first and self.moving_objects:
            # Add moving object directions first
            for angle, dist in self.moving_objects:
                # Camera at back → add π
                cam_angle = normalize_angle(angle + CAMERA_OFFSET_RAD)
                directions.append(('object', cam_angle, dist))
            self.moving_objects = []
            self.get_logger().info(
                f'[Sweep] Checking {len(directions)} moving objects first')

        # Add open corridor directions
        for angle, dist in self.open_directions:
            cam_angle = normalize_angle(angle + CAMERA_OFFSET_RAD)
            directions.append(('open', cam_angle, dist))

        # Fallback if no open corridors
        if not directions:
            num = NUM_SWEEP_STOPS
            for i in range(num):
                cam_angle = normalize_angle(
                    i * (2 * math.pi / num) + CAMERA_OFFSET_RAD)
                directions.append(('fallback', cam_angle, 0))
            self.get_logger().info(
                f'[Sweep] No open corridors — fallback {num}-stop sweep')
        else:
            self.get_logger().info(
                f'[Sweep] {len(directions)} directions to check '
                f'({sum(1 for d in directions if d[0]=="object")} objects, '
                f'{sum(1 for d in directions if d[0]=="open")} corridors)')

        self.sweep_directions  = directions
        self.sweep_phase       = 'turning'
        self.sweep_phase_start = time.time()
        self.last_yolo_result  = None
        self._trigger_camera()  # trigger immediately on entry

    def _loop_sweep(self):
        now = time.time()

        if self.sweep_stop >= len(self.sweep_directions):
            # All directions checked
            self.get_logger().info(
                f'[Sweep] All {len(self.sweep_directions)} directions checked '
                f'— {self.target_cat} not found')
            self._advance_waypoint()
            return

        kind, target_angle, dist = self.sweep_directions[self.sweep_stop]
        elapsed = now - self.sweep_phase_start

        # ── Turning phase ─────────────────────────────────────────────────────
        if self.sweep_phase == 'turning':
            # Time to turn one FOV stop
            turn_sec = CAMERA_HFOV_RAD / ANGULAR_SPEED

            if elapsed < turn_sec:
                # Check obstacle before turning
                if self.front_distance < SAFE_DISTANCE:
                    self._stop_motors()
                    return
                twist = Twist()
                # Turn direction: positive angle = left, negative = right
                twist.angular.z = ANGULAR_SPEED if target_angle >= 0 else -ANGULAR_SPEED
                self.cmd_pub.publish(twist)
            else:
                # Turn done — stop and pause for YOLO
                self._stop_motors()
                self.sweep_phase       = 'pausing'
                self.sweep_phase_start = now
                self.last_yolo_result  = None  # clear stale results
                self._trigger_camera()         # trigger YOLO immediately
                self.get_logger().info(
                    f'[Sweep] Stop {self.sweep_stop+1}/'
                    f'{len(self.sweep_directions)} '
                    f'({kind}) — pausing {YOLO_PAUSE_SEC}s for YOLO')

        # ── Pausing phase — wait for YOLO ─────────────────────────────────────
        elif self.sweep_phase == 'pausing':
            # Keep triggering camera every 0.5s during pause
            if elapsed % 0.5 < 0.11:
                self._trigger_camera()

            # Check if YOLO found the target
            if self.last_yolo_result == self.target_cat:
                self.get_logger().info(
                    f'[Sweep] ✅ {self.target_cat} confirmed at stop '
                    f'{self.sweep_stop+1}!')
                self._enter_follow()
                return

            # Pause done — move to next direction
            if elapsed >= YOLO_PAUSE_SEC:
                self.sweep_stop       += 1
                self.sweep_phase       = 'turning'
                self.sweep_phase_start = now
                self.last_yolo_result  = None

    # ══════════════════════════════════════════════════════════════════════════
    # FOLLOW — follow cat at 1 foot, camera faces back
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_follow(self):
        self.state = FOLLOW
        self._cancel_nav()
        self._stop_motors()
        self.status_pub.publish(String(data=f'following {self.target_cat}'))
        self.get_logger().info(f'[Follow] Tracking {self.target_cat}')

    def _loop_follow(self):
        now = self.get_clock().now()

        # Cat lost?
        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > LOST_TIMEOUT:
                self.get_logger().info('[Follow] Cat lost — resuming search')
                self.cat_cx    = None
                self.nav_sent  = False
                self.nav_arrived = False
                self.state     = NAVIGATE
                self._publish_status(f'lost {self.target_cat}')
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        # Obstacle while following — stop
        if self.front_distance < SAFE_DISTANCE:
            self._stop_motors()
            self.get_logger().warn(
                f'[Follow] Obstacle {self.front_distance:.2f}m — pausing',
                throttle_duration_sec=1.0)
            return

        twist = Twist()

        # Angular: proportional from ball_follower
        error           = (self.cat_cx * 640) - 320
        twist.angular.z = -error / 300.0

        # Linear: maintain 1 foot, camera faces BACK → move backward
        dist = self.cat_dist if self.cat_dist is not None else self.front_distance

        if dist is not None and not math.isinf(dist):
            dist_error = dist - FOLLOW_DIST_TARGET
            if dist_error > 0.05:
                twist.linear.x = -min(FOLLOW_SPEED_MAX, dist_error * 0.4)
            elif dist_error < -0.05:
                twist.linear.x = max(0.10, abs(dist_error) * 0.4)
            else:
                twist.linear.x = 0.0
        else:
            twist.linear.x = 0.0

        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f'[Follow] dist={dist:.2f}m lin={twist.linear.x:.2f} '
            f'ang={twist.angular.z:.2f}',
            throttle_duration_sec=0.5)

    # ══════════════════════════════════════════════════════════════════════════
    # AVOID — turn toward most open direction
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_avoid(self):
        self.pre_avoid_state = self.state
        self.state           = AVOID
        self.avoid_start     = time.time()
        self._cancel_nav()
        self._stop_motors()

        # Turn toward most open side
        if self.open_directions:
            best = self.open_directions[0][0]
            self.avoid_turn_dir = 1 if best > 0 else -1
        elif self.front_right_dist > self.front_left_dist:
            self.avoid_turn_dir = -1  # right
        else:
            self.avoid_turn_dir = 1   # left

        self.get_logger().info(
            f'[Avoid] front={self.front_distance:.2f}m '
            f'turning {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"}')

    def _loop_avoid(self):
        elapsed = time.time() - self.avoid_start

        if self.front_distance > AVOID_DISTANCE:
            self.get_logger().info('[Avoid] ✅ Clear — resuming')
            self._stop_motors()
            self.nav_sent  = False
            self.nav_arrived = False
            self.state     = NAVIGATE
            return

        twist = Twist()

        if elapsed < 2.0:
            # Turn toward open space
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
        elif elapsed < 3.0:
            # Back up slightly
            twist.linear.x  = -0.08
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED * 0.5
        else:
            # Flip direction if still stuck
            self.avoid_turn_dir = -self.avoid_turn_dir
            self.avoid_start    = time.time()
            self.get_logger().warn('[Avoid] Still stuck — flipping direction')

        self.cmd_pub.publish(twist)

    # ══════════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════════

    def _enter_done(self):
        self.state = DONE
        self._cancel_nav()
        self._stop_motors()
        self.target_pub.publish(String(data='stop'))
        self._publish_status('stop')
        self.get_logger().info('[DONE] Not found — returning home')
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

    # ══════════════════════════════════════════════════════════════════════════
    # Utilities
    # ══════════════════════════════════════════════════════════════════════════

    def _trigger_camera(self):
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
        self.state         = IDLE
        self.active        = False
        self.target_cat    = None
        self.waypoint_index = 0
        self.search_cycles  = 0
        self.moving_objects = []
        self.cat_cx         = None
        self.last_seen      = None
        self._publish_status('stopped')

    def _cluster(self, points, gap=0.3):
        if not points:
            return []
        clusters = []
        current  = [points[0]]
        for pt in points[1:]:
            dists = [math.hypot(pt[0]-c[0], pt[1]-c[1]) for c in current]
            if min(dists) <= gap:
                current.append(pt)
            else:
                clusters.append(current)
                current = [pt]
        clusters.append(current)
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