#!/usr/bin/env python3
"""
seek_cat.py - WATCH MODE (stop-and-scan free-wander)
====================================================
Find the named cat by alternating "drive to a random spot" and "stop &
scan for 5 seconds." Detection happens during the stops (clean frames,
no motion blur). When found, follow ball-follower style: keep ~1 m
distance, turn to keep cat centered. Always avoid obstacles.

State machine
-------------
  IDLE        : no target, motors off
  WANDER      : Nav2 driving to a random reachable costmap point. YOLO
                runs the whole time but we don't expect reliable hits
                while moving - this is just travel time.
  WANDER_MAN  : Manual-drive fallback when Nav2 keeps rejecting goals
                (typically AMCL covariance going bad). Drives forward,
                turns when LiDAR sees an obstacle.
  SCAN_PAUSE  : Stopped, holding still for SCAN_DURATION_SEC. This is
                where detection actually happens (clean YOLO frames).
                Entered from:
                  - WANDER when Nav2 goal arrives
                  - WANDER/WANDER_MAN when LiDAR sees cat-sized motion
                If YOLO publishes /cat_position during the pause -> FOLLOW.
                If pause expires -> back to WANDER.
  FOLLOW      : Visual servoing, ball-follower style.
                  error_x   = cat_cx - 0.5
                  angular.z = -kp_ang * error_x
                  dist_err  = cat_dist - 1.0
                  linear.x  > 0 forward, < 0 backward
                Angular gain ramps 0 -> max over FOLLOW_GAIN_RAMP_SEC
                for smooth start.
  AVOID       : Sidestep when LiDAR sees something inside SAFE_DISTANCE.
                After clearance, resume previous state.
  DONE        : Give up - return home, motors off.

Why stop-and-scan
-----------------
Camera at ~15 Hz, YOLO at ~4 Hz. While the robot is rotating during
Nav2 navigation, frames are motion-blurred and the cat sweeps through
FOV in 1-2 frames - not enough for cat_detector's internal 3-of-5
voting. Stopping for 5 seconds gives clean frames and ~20 YOLO ticks,
plenty to confirm a real detection.

Why no CONFIRM state in seek_cat
--------------------------------
cat_detector.py already does multi-frame voting BEFORE publishing
/cat_position. By the time we receive a position, it's verified. So
we go straight to FOLLOW on first sighting. The FOLLOW gain ramp gives
smooth start; we don't need an extra stop-and-verify on top.
"""

import math
import sys
import signal
import time
import random

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# ─── Speeds ──────────────────────────────────────────────────────────────
ANGULAR_SPEED        = 0.3
WANDER_MAN_LIN_SPEED = 0.10
WANDER_MAN_ANG_SPEED = 0.4

# ─── Follow (ball-follower style) ────────────────────────────────────────
FOLLOW_DIST_TARGET     = 1.0
FOLLOW_TOLERANCE       = 0.15
FOLLOW_SPEED_MAX       = 0.12
FOLLOW_KP_LIN          = 0.4
FOLLOW_KP_ANG_MAX      = 1.5
FOLLOW_GAIN_RAMP_SEC   = 0.8     # ramp angular gain 0->max (smooth start)
FOLLOW_FRONT_HARD_STOP = 0.50
FOLLOW_NO_DEPTH_SPEED  = 0.05
LOST_TIMEOUT_RAMPING   = 4.0
LOST_TIMEOUT_MOVING    = 1.5

# ─── Scan pause ──────────────────────────────────────────────────────────
# Stop and watch this long at each random point and at every LiDAR
# motion event. Detection happens here - YOLO needs ~1 second of clean
# frames to accumulate cat_detector's internal 3-of-5 voting.
SCAN_DURATION_SEC      = 5.0

# ─── LiDAR motion priority ───────────────────────────────────────────────
# When LiDAR sees a persistent cat-sized cluster moving in front of us,
# stop wandering and scan immediately. We do NOT spin to face it - the
# camera FOV (~63 deg) already covers the front sector LiDAR is watching.
MOTION_DIST_THRESH     = 0.20     # m: scan-to-scan delta to call "moving"
MOTION_RANGE_MAX       = 3.0      # m: only consider motion within this range
MOTION_CLUSTER_MIN     = 5        # min points in a moving cluster
MOTION_SIZE_MIN        = 0.10     # m: cluster width min
MOTION_SIZE_MAX        = 0.60     # m: cluster width max
MOTION_PERSIST_COUNT   = 3        # need this many scans of agreement
MOTION_FRONT_HALF_RAD  = math.radians(35.0)   # only watch the front ~70 deg

# ─── Obstacle avoidance ──────────────────────────────────────────────────
SAFE_DISTANCE          = 0.25
AVOID_DISTANCE         = 0.35
AVOID_MIN_DURATION     = 2.0
LIDAR_MIN_RANGE        = 0.20    # filter chassis self-occlusion

# ─── Wander ──────────────────────────────────────────────────────────────
WANDER_MIN_DIST        = 1.0
WANDER_MAX_TRIES       = 30
WANDER_FREE_CELL_MAX   = 30
NAV_GOAL_TIMEOUT       = 30.0
MAX_GOAL_FAILURES      = 10
NAV2_FALLBACK_FAILURES = 3
WANDER_MAN_DURATION    = 12.0

SEARCH_TIME_BUDGET_SEC = 300.0   # 5 min total search


KNOWN_CATS = {'eevee', 'raichu', 'pichu'}


# ─── States ──────────────────────────────────────────────────────────────
IDLE       = 'IDLE'
WANDER     = 'WANDER'
WANDER_MAN = 'WANDER_MAN'
SCAN_PAUSE = 'SCAN_PAUSE'
FOLLOW     = 'FOLLOW'
AVOID      = 'AVOID'
DONE       = 'DONE'


def yaw_to_quat_z_w(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # ─── State ────────────────────────────────────────────────────
        self.state             = IDLE
        self.target_cat        = None
        self.active            = False
        self.search_start_time = None

        # Nav2
        self.nav_goal_handle = None
        self.nav_sent        = False
        self.nav_send_time   = None
        self.nav_arrived     = False
        self.nav_failures    = 0
        self.man_drive_start = None

        # Cat sighting
        self.cat_cx       = None
        self.cat_dist     = None
        self.last_seen    = None
        self.follow_start = None

        # Scan pause
        self.scan_pause_start = None

        # LiDAR
        self.front_distance   = float('inf')
        self.front_left_dist  = float('inf')
        self.front_right_dist = float('inf')
        self.left_distance    = float('inf')
        self.right_distance   = float('inf')
        self.back_distance    = float('inf')

        # LiDAR motion tracking
        self.prev_scan      = None
        self.motion_persist = 0

        # Robot pose
        self.robot_x   = 0.0
        self.robot_y   = 0.0
        self.have_pose = False

        # Costmap
        self.costmap_msg = None

        # Avoid
        self.avoid_turn_dir  = 1
        self.avoid_start     = None
        self.pre_avoid_state = WANDER

        self.last_trigger_time = 0.0

        # ─── ROS interface ────────────────────────────────────────────
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(String,        '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,        '/cat_identity', self._cb_identity, 10)
        self.create_subscription(PointStamped,  '/cat_position', self._cb_position, 10)
        self.create_subscription(LaserScan,     '/scan',         self._cb_scan,     10)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self._cb_costmap, 1)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self._cb_pose, 10)

        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',              10)
        self.cam_pub    = self.create_publisher(String, '/camera_check_trigger', 10)
        self.target_pub = self.create_publisher(String, '/cat_target',           10)
        self.status_pub = self.create_publisher(String, '/seek_status',          10)

        self.create_timer(0.1, self._loop)
        self.get_logger().info(
            'SeekCat ready - stop-and-scan wander '
            f'(scan={SCAN_DURATION_SEC}s per stop)')

    # ─── Callbacks ────────────────────────────────────────────────────
    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        if text not in KNOWN_CATS:
            return
        self.target_cat        = text
        self.active            = True
        self.search_start_time = time.time()
        self.nav_failures      = 0
        self.nav_sent          = False
        self.nav_arrived       = False
        self.cat_cx            = None
        self.cat_dist          = None
        self.last_seen         = None
        self.state             = WANDER
        self.get_logger().info(f'Seeking: {self.target_cat}')
        self._publish_status(f'seeking {self.target_cat}')

    def _cb_identity(self, msg: String):
        name = msg.data.strip().lower()
        if not self.target_cat or name != self.target_cat:
            return
        if self.state in (WANDER, WANDER_MAN, SCAN_PAUSE):
            self.get_logger().info(
                f'YOLO sees {self.target_cat} (state={self.state})')

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0.05 else None
        self.last_seen = self.get_clock().now()

        # cat_detector already votes 3-of-5 before publishing /cat_position
        # so we trust the first hit and go straight to FOLLOW. The angular
        # gain ramp in FOLLOW gives smooth start.
        if self.state in (WANDER, WANDER_MAN, SCAN_PAUSE):
            self.get_logger().info(
                f'[/cat_position] cx={self.cat_cx:.2f} '
                f'dist={"n/a" if self.cat_dist is None else f"{self.cat_dist:.2f}m"} '
                f'(state={self.state}) -> FOLLOW')
            self._enter_follow()

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.have_pose = True

    def _cb_costmap(self, msg: OccupancyGrid):
        self.costmap_msg = msg

    def _cb_scan(self, msg: LaserScan):
        ranges  = msg.ranges
        n       = len(ranges)
        eff_min = max(msg.range_min, LIDAR_MIN_RANGE)

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if eff_min < ranges[i] < msg.range_max
                    and math.isfinite(ranges[i])]
            return min(vals) if vals else float('inf')

        self.front_distance   = safe_min(
            list(range(0, n//12)) + list(range(11*n//12, n)))
        self.front_left_dist  = safe_min(list(range(n//12, n//4)))
        self.front_right_dist = safe_min(list(range(3*n//4, 11*n//12)))
        self.left_distance    = safe_min(list(range(n//4,    5*n//12)))
        self.right_distance   = safe_min(list(range(7*n//12, 3*n//4)))
        self.back_distance    = safe_min(list(range(5*n//12, 7*n//12)))

        # Motion detection (only when actively wandering)
        if (self.active
                and self.state in (WANDER, WANDER_MAN)
                and self.prev_scan is not None):
            self._check_front_motion(msg, ranges, eff_min)

        self.prev_scan = msg

    def _check_front_motion(self, msg, ranges, eff_min):
        prev_ranges = self.prev_scan.ranges
        n    = len(ranges)
        ainc = msg.angle_increment
        amin = msg.angle_min
        rmax = min(msg.range_max, MOTION_RANGE_MAX)

        moving_pts = []
        for i in range(min(n, len(prev_ranges))):
            r_now  = ranges[i]
            r_prev = prev_ranges[i]
            if not (eff_min < r_now < rmax):
                continue
            if not (eff_min < r_prev < rmax):
                continue
            angle = amin + i * ainc
            a = (angle + math.pi) % (2*math.pi) - math.pi
            if abs(a) > MOTION_FRONT_HALF_RAD:
                continue
            if abs(r_now - r_prev) > MOTION_DIST_THRESH:
                x = r_now * math.cos(a)
                y = r_now * math.sin(a)
                moving_pts.append((x, y))

        if len(moving_pts) < MOTION_CLUSTER_MIN:
            self.motion_persist = max(0, self.motion_persist - 1)
            return

        xs = [p[0] for p in moving_pts]
        ys = [p[1] for p in moving_pts]
        width = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if not (MOTION_SIZE_MIN <= width <= MOTION_SIZE_MAX):
            self.motion_persist = max(0, self.motion_persist - 1)
            return

        self.motion_persist += 1
        if self.motion_persist >= MOTION_PERSIST_COUNT:
            self.get_logger().info(
                f'[LiDAR] Motion in front (width={width:.2f}m, '
                f'{len(moving_pts)} pts) -> SCAN_PAUSE')
            self.motion_persist = 0
            self._enter_scan_pause(reason='lidar_motion')

    # ─── Main loop ────────────────────────────────────────────────────
    def _loop(self):
        if not self.active or self.target_cat is None:
            return

        # Time budget (only during search, not follow)
        if (self.search_start_time is not None
                and self.state not in (FOLLOW, DONE)
                and time.time() - self.search_start_time > SEARCH_TIME_BUDGET_SEC):
            self.get_logger().warn('[Done] Search time budget exceeded')
            self._enter_done()
            return

        # Hard obstacle override
        if self.state not in (AVOID, IDLE, DONE):
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'[{self.state}] Front blocked '
                    f'({self.front_distance:.2f}m) -> AVOID')
                self._enter_avoid()
                return

        # Trigger camera detector in any state where YOLO is useful
        if self.state in (WANDER, WANDER_MAN, SCAN_PAUSE):
            self._trigger_camera()

        if self.state == WANDER:
            self._loop_wander()
        elif self.state == WANDER_MAN:
            self._loop_wander_manual()
        elif self.state == SCAN_PAUSE:
            self._loop_scan_pause()
        elif self.state == FOLLOW:
            self._loop_follow()
        elif self.state == AVOID:
            self._loop_avoid()

    # ─── WANDER (Nav2 to random costmap point) ────────────────────────
    def _loop_wander(self):
        if not self.nav_client.server_is_ready() or self.costmap_msg is None:
            self.get_logger().info(
                '[Wander] Waiting for Nav2 / costmap...',
                throttle_duration_sec=3.0)
            return

        if not self.nav_sent:
            goal = self._pick_random_costmap_goal()
            if goal is None:
                self.get_logger().warn(
                    '[Wander] No reachable random goal found; will retry',
                    throttle_duration_sec=2.0)
                return
            gx, gy = goal
            self._send_nav_goal(gx, gy)
            return

        if (self.nav_send_time is not None
                and time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT):
            self.get_logger().warn('[Wander] Goal timeout - retry')
            self._handle_nav_failure()
            return

        if self.nav_arrived:
            self.get_logger().info(
                '[Wander] Arrived -> SCAN_PAUSE (stop & scan)')
            self.nav_sent    = False
            self.nav_arrived = False
            self._enter_scan_pause(reason='arrived_at_goal')

    def _pick_random_costmap_goal(self):
        cm = self.costmap_msg
        if cm is None:
            return None
        info = cm.info
        w, h = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        data = cm.data

        for _ in range(WANDER_MAX_TRIES):
            ix = random.randrange(w)
            iy = random.randrange(h)
            cost = data[iy * w + ix]
            if cost < 0 or cost > WANDER_FREE_CELL_MAX:
                continue
            gx = ox + (ix + 0.5) * res
            gy = oy + (iy + 0.5) * res
            if self.have_pose:
                d = math.hypot(gx - self.robot_x, gy - self.robot_y)
                if d < WANDER_MIN_DIST:
                    continue
            return (gx, gy)
        return None

    def _send_nav_goal(self, x, y):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(x)
        goal.pose.pose.position.y    = float(y)
        if self.have_pose:
            yaw = math.atan2(y - self.robot_y, x - self.robot_x)
        else:
            yaw = 0.0
        qz, qw = yaw_to_quat_z_w(yaw)
        goal.pose.pose.orientation.z = qz
        goal.pose.pose.orientation.w = qw

        self.nav_arrived     = False
        self.nav_sent        = True
        self.nav_send_time   = time.time()
        self.nav_goal_handle = None

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._nav_accepted_cb)
        self.get_logger().info(
            f'[Wander] Goal sent ({x:.2f}, {y:.2f}) yaw={math.degrees(yaw):.0f}')

    def _nav_accepted_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('[Wander] Goal rejected')
            self._handle_nav_failure()
            return
        self.nav_goal_handle = handle
        handle.get_result_async().add_done_callback(self._nav_result_cb)

    def _nav_result_cb(self, future):
        if self.state != WANDER:
            return
        status = future.result().status
        if status == 4:
            self.nav_arrived = True
            self.nav_failures = 0
        else:
            self.get_logger().warn(f'[Wander] Goal status={status}')
            self._handle_nav_failure()

    def _handle_nav_failure(self):
        if self.state != WANDER:
            return
        self.nav_sent    = False
        self.nav_arrived = False
        self.nav_failures += 1

        if self.nav_failures >= MAX_GOAL_FAILURES:
            self.get_logger().error(
                f'[Wander] {MAX_GOAL_FAILURES} failures in a row - giving up')
            self._enter_done()
            return

        if self.nav_failures >= NAV2_FALLBACK_FAILURES:
            self.get_logger().warn(
                f'[Wander] {self.nav_failures} Nav2 failures - '
                f'manual drive for {WANDER_MAN_DURATION}s')
            self._enter_wander_manual()

    def _cancel_nav(self):
        if self.nav_goal_handle is not None:
            try:
                self.nav_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.nav_goal_handle = None
        self.nav_sent    = False
        self.nav_arrived = False

    # ─── WANDER_MAN (manual drive fallback) ───────────────────────────
    def _enter_wander_manual(self):
        self.state = WANDER_MAN
        self.man_drive_start = time.time()
        self._cancel_nav()
        self._stop_motors()

    def _loop_wander_manual(self):
        elapsed = time.time() - self.man_drive_start
        if elapsed > WANDER_MAN_DURATION:
            self.get_logger().info(
                '[ManDrive] Done -> SCAN_PAUSE before retrying Nav2')
            self.nav_failures = 0
            self._enter_scan_pause(reason='man_drive_done')
            return

        twist = Twist()
        if self.front_distance < FOLLOW_FRONT_HARD_STOP:
            if self.front_left_dist > self.front_right_dist:
                twist.angular.z = +WANDER_MAN_ANG_SPEED
            else:
                twist.angular.z = -WANDER_MAN_ANG_SPEED
            twist.linear.x = 0.0
        else:
            twist.linear.x = WANDER_MAN_LIN_SPEED
            if self.front_left_dist > self.front_right_dist + 0.3:
                twist.angular.z = +0.1
            elif self.front_right_dist > self.front_left_dist + 0.3:
                twist.angular.z = -0.1
        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f'[ManDrive] t={elapsed:.1f}s lin={twist.linear.x:+.2f} '
            f'ang={twist.angular.z:+.2f} front={self.front_distance:.2f}',
            throttle_duration_sec=1.0)

    # ─── SCAN_PAUSE ───────────────────────────────────────────────────
    # Stop and watch for SCAN_DURATION_SEC. Detection happens here.
    # If YOLO publishes /cat_position during the pause, _cb_position
    # transitions us straight to FOLLOW. Otherwise we resume WANDER.
    def _enter_scan_pause(self, reason='arrived'):
        self.get_logger().info(
            f'[Scan] Holding {SCAN_DURATION_SEC}s ({reason}) - YOLO scanning')
        self._cancel_nav()
        self._stop_motors()
        self.state            = SCAN_PAUSE
        self.scan_pause_start = time.time()
        self.motion_persist   = 0   # reset so we don't immediately re-trigger
        self.status_pub.publish(String(data=f'scanning {self.target_cat}'))

    def _loop_scan_pause(self):
        elapsed = time.time() - self.scan_pause_start
        self._stop_motors()
        self._trigger_camera()

        if elapsed % 1.0 < 0.11:
            self.get_logger().info(
                f'[Scan] t={elapsed:.1f}/{SCAN_DURATION_SEC}s, watching...',
                throttle_duration_sec=1.0)

        if elapsed >= SCAN_DURATION_SEC:
            self.get_logger().info(
                f'[Scan] No cat in {SCAN_DURATION_SEC}s - back to WANDER')
            self.state = WANDER

    # ─── FOLLOW (ball-follower style) ─────────────────────────────────
    def _enter_follow(self):
        self.state = FOLLOW
        self._cancel_nav()
        self._stop_motors()
        self.follow_start = time.time()
        self.status_pub.publish(String(data=f'following {self.target_cat}'))
        self.get_logger().info(
            f'[Follow] Tracking {self.target_cat} '
            f'(target dist {FOLLOW_DIST_TARGET}m)')

    def _current_kp_ang(self):
        if self.follow_start is None:
            return FOLLOW_KP_ANG_MAX
        t = time.time() - self.follow_start
        if t >= FOLLOW_GAIN_RAMP_SEC:
            return FOLLOW_KP_ANG_MAX
        return FOLLOW_KP_ANG_MAX * (t / FOLLOW_GAIN_RAMP_SEC)

    def _follow_lost_timeout(self):
        if (self.follow_start is not None
                and (time.time() - self.follow_start) < FOLLOW_GAIN_RAMP_SEC):
            return LOST_TIMEOUT_RAMPING
        return LOST_TIMEOUT_MOVING

    def _loop_follow(self):
        now = self.get_clock().now()

        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > self._follow_lost_timeout():
                self.get_logger().info(
                    f'[Follow] Cat lost ({elapsed:.1f}s) - back to WANDER')
                self.cat_cx       = None
                self.cat_dist     = None
                self.follow_start = None
                self.nav_sent     = False
                self.nav_arrived  = False
                self.state        = WANDER
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        twist = Twist()

        # Angular: center the cat (ball-follower style + gain ramp)
        cx_error        = self.cat_cx - 0.5
        kp_ang          = self._current_kp_ang()
        twist.angular.z = -kp_ang * cx_error
        if twist.angular.z >  ANGULAR_SPEED: twist.angular.z =  ANGULAR_SPEED
        if twist.angular.z < -ANGULAR_SPEED: twist.angular.z = -ANGULAR_SPEED

        # Linear: keep target distance
        if self.cat_dist is None:
            if self.front_distance < FOLLOW_FRONT_HARD_STOP:
                twist.linear.x = 0.0
            else:
                twist.linear.x = +FOLLOW_NO_DEPTH_SPEED
        else:
            dist       = self.cat_dist
            dist_error = dist - FOLLOW_DIST_TARGET

            if dist_error > FOLLOW_TOLERANCE:
                if self.front_distance < FOLLOW_FRONT_HARD_STOP:
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = +min(FOLLOW_SPEED_MAX,
                                          FOLLOW_KP_LIN * dist_error)
            elif dist_error < -FOLLOW_TOLERANCE:
                if self.back_distance < SAFE_DISTANCE:
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = -min(FOLLOW_SPEED_MAX,
                                          FOLLOW_KP_LIN * abs(dist_error))
            else:
                twist.linear.x = 0.0

        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f'[Follow] cx={self.cat_cx:.2f} '
            f'dist={"n/a" if self.cat_dist is None else f"{self.cat_dist:.2f}m"} '
            f'kp_ang={kp_ang:.2f} '
            f'lin={twist.linear.x:+.2f} ang={twist.angular.z:+.2f} '
            f'(front={self.front_distance:.2f} back={self.back_distance:.2f})',
            throttle_duration_sec=0.5)

    # ─── AVOID ────────────────────────────────────────────────────────
    def _enter_avoid(self):
        # Let Nav2 handle obstacles itself during an active goal
        if (self.state == WANDER
                and self.nav_sent and not self.nav_arrived):
            self.get_logger().warn(
                f'[Avoid] Front {self.front_distance:.2f}m '
                f'but Nav2 is active - letting Nav2 handle it')
            return

        self.pre_avoid_state = self.state
        self.state           = AVOID
        self.avoid_start     = time.time()
        self._cancel_nav()
        self._stop_motors()

        if self.front_right_dist > self.front_left_dist:
            self.avoid_turn_dir = -1
        else:
            self.avoid_turn_dir = 1

        self.get_logger().info(
            f'[Avoid] front={self.front_distance:.2f}m '
            f'-> {"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"} '
            f'(was {self.pre_avoid_state})')

    def _loop_avoid(self):
        elapsed = time.time() - self.avoid_start

        if elapsed < AVOID_MIN_DURATION:
            twist = Twist()
            if elapsed < 1.0:
                twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            else:
                twist.linear.x  = -0.08
                twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED * 0.5
            self.cmd_pub.publish(twist)
            return

        if self.front_distance > AVOID_DISTANCE:
            self._stop_motors()
            now = self.get_clock().now()
            cat_fresh = (self.last_seen is not None and
                         (now - self.last_seen).nanoseconds / 1e9 < LOST_TIMEOUT_MOVING)
            if self.pre_avoid_state == FOLLOW and cat_fresh:
                self.get_logger().info('[Avoid] Clear -> resume FOLLOW')
                self.state = FOLLOW
            elif self.pre_avoid_state == WANDER_MAN:
                self.get_logger().info('[Avoid] Clear -> resume manual drive')
                self.man_drive_start = time.time()
                self.state = WANDER_MAN
            elif self.pre_avoid_state == SCAN_PAUSE:
                self.get_logger().info('[Avoid] Clear -> resume SCAN_PAUSE')
                self.scan_pause_start = time.time()   # restart timer
                self.state = SCAN_PAUSE
            else:
                self.get_logger().info('[Avoid] Clear -> resume WANDER')
                self.nav_sent    = False
                self.nav_arrived = False
                self.state       = WANDER
            return

        if elapsed < AVOID_MIN_DURATION * 2.5:
            twist = Twist()
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            self.cmd_pub.publish(twist)
        else:
            self.avoid_turn_dir = -self.avoid_turn_dir
            self.avoid_start    = time.time()
            self.get_logger().warn('[Avoid] Still stuck - flipping direction')

    # ─── DONE ─────────────────────────────────────────────────────────
    def _enter_done(self):
        self.state = DONE
        self._cancel_nav()
        self._stop_motors()
        self.target_pub.publish(String(data='stop'))
        self._publish_status('stop')
        self.get_logger().info('[DONE] Search ended - returning home')
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

    # ─── Helpers ──────────────────────────────────────────────────────
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
        self.cat_cx            = None
        self.cat_dist          = None
        self.last_seen         = None
        self.follow_start      = None
        self.scan_pause_start  = None
        self.nav_failures      = 0
        self.search_start_time = None
        self.motion_persist    = 0
        self._publish_status('stopped')


def _safe_cleanup(node):
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