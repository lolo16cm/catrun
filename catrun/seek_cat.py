#!/usr/bin/env python3
"""
seek_cat.py - WATCH MODE (free-wander edition)
==============================================
Find the named cat by wandering the room (Nav2 + costmap-based random
goals), then follow it ball-follower style: keep ~1 m distance, turn
to keep cat centered. Always avoid obstacles.

State machine
-------------
  IDLE       : no target, motors off
  WANDER     : Nav2 driving to a random reachable point in the costmap.
               YOLO runs on the camera the whole time.
  WANDER_MAN : Manual-drive fallback. Used when Nav2 has been rejecting
               goals repeatedly (typically AMCL covariance going bad).
               Drives forward, turns when LiDAR sees an obstacle.
  CONFIRM    : Cat just detected - stop and watch for a second to make
               sure it wasn't a fluke YOLO frame.
  FOLLOW     : Visual servoing, ball-follower style.
                 error_x   = cat_cx - 0.5          (center the cat)
                 angular.z = -kp_ang * error_x     (turn toward cat)
                 dist_err  = cat_dist - TARGET     (1.0 m)
                 linear.x  > 0 to approach, < 0 to back away
               Angular gain ramps over the first 0.8s so we don't
               snap-turn at first sighting (which causes motion blur
               and loses the target).
  AVOID      : Sidestep when LiDAR sees something inside SAFE_DISTANCE.
               After clearance, resume previous state.
  DONE       : Give up - return home, motors off.

What's gone vs. previous version
--------------------------------
  * No fixed waypoint list - the robot wanders.
  * No CHECK_OBJ "spin to face every moving LiDAR cluster" - that was
    the source of the constant spinning. YOLO runs from the camera
    regardless of what LiDAR sees.
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

# ─── Follow (ball-follower style: keep target distance) ──────────────────
FOLLOW_DIST_TARGET     = 1.0    # m - keep this far from cat
FOLLOW_TOLERANCE       = 0.15   # m - deadband around target
FOLLOW_SPEED_MAX       = 0.12   # m/s - cap on approach/retreat speed
FOLLOW_KP_LIN          = 0.4
FOLLOW_KP_ANG_MAX      = 1.5
FOLLOW_GAIN_RAMP_SEC   = 0.8    # ramp angular gain 0->max (smooth start)
FOLLOW_FRONT_HARD_STOP = 0.50   # never drive forward closer than this
FOLLOW_NO_DEPTH_SPEED  = 0.05   # m/s crawl when no distance estimate
LOST_TIMEOUT_RAMPING   = 4.0    # tolerate longer drop-out during smooth start
LOST_TIMEOUT_MOVING    = 1.5    # snappier drop-out once moving fast

# ─── Confirm hold ────────────────────────────────────────────────────────
CONFIRM_HOLD_SEC       = 1.0    # stop & watch for this long after first sight
CONFIRM_REQUIRE_HITS   = 2      # need this many position updates to commit

# ─── Obstacle avoidance ──────────────────────────────────────────────────
SAFE_DISTANCE          = 0.25
AVOID_DISTANCE         = 0.35
AVOID_MIN_DURATION     = 2.0
LIDAR_MIN_RANGE        = 0.20   # filter chassis self-occlusion

# ─── Wander ──────────────────────────────────────────────────────────────
WANDER_MIN_DIST        = 1.0    # m - new goal at least this far away
WANDER_MAX_TRIES       = 30
WANDER_FREE_CELL_MAX   = 30     # costmap cells <= this are "free"
NAV_GOAL_TIMEOUT       = 30.0
MAX_GOAL_FAILURES      = 10     # in a row -> DONE
NAV2_FALLBACK_FAILURES = 3      # in a row -> drop to manual drive
WANDER_MAN_DURATION    = 12.0   # seconds of manual drive before retrying Nav2

SEARCH_TIME_BUDGET_SEC = 300.0  # 5 min total search budget


KNOWN_CATS = {'eevee', 'raichu', 'pichu'}


# ─── States ──────────────────────────────────────────────────────────────
IDLE       = 'IDLE'
WANDER     = 'WANDER'
WANDER_MAN = 'WANDER_MAN'
CONFIRM    = 'CONFIRM'
FOLLOW     = 'FOLLOW'
AVOID      = 'AVOID'
DONE       = 'DONE'


def yaw_to_quat_z_w(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class SeekCat(Node):

    def __init__(self):
        super().__init__('seek_cat')

        # ─── State ────────────────────────────────────────────────────
        self.state       = IDLE
        self.target_cat  = None
        self.active      = False
        self.search_start_time = None

        self.nav_goal_handle  = None
        self.nav_sent         = False
        self.nav_send_time    = None
        self.nav_arrived      = False
        self.nav_failures     = 0
        self.man_drive_start  = None

        self.cat_cx        = None
        self.cat_dist      = None
        self.last_seen     = None
        self.confirm_start = None
        self.confirm_hits  = 0
        self.follow_start  = None

        self.robot_x  = 0.0
        self.robot_y  = 0.0
        self.have_pose = False

        self.costmap_msg = None

        self.front_distance   = float('inf')
        self.front_left_dist  = float('inf')
        self.front_right_dist = float('inf')
        self.left_distance    = float('inf')
        self.right_distance   = float('inf')
        self.back_distance    = float('inf')

        self.avoid_turn_dir  = 1
        self.avoid_start     = None
        self.pre_avoid_state = WANDER

        self.last_trigger_time = 0.0

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
            'SeekCat ready - free-wander mode (WANDER -> CONFIRM -> FOLLOW)')

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
        if self.state in (WANDER, WANDER_MAN):
            self.get_logger().info(
                f'YOLO sees {self.target_cat} - going to CONFIRM')

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0.05 else None
        self.last_seen = self.get_clock().now()

        if self.state in (WANDER, WANDER_MAN):
            self._enter_confirm()
        elif self.state == CONFIRM:
            self.confirm_hits += 1

    def _cb_pose(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.have_pose = True

    def _cb_costmap(self, msg: OccupancyGrid):
        self.costmap_msg = msg

    def _cb_scan(self, msg: LaserScan):
        ranges = msg.ranges
        n      = len(ranges)
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

    # ─── Main loop ────────────────────────────────────────────────────
    def _loop(self):
        if not self.active or self.target_cat is None:
            return

        # Time budget (search only, not follow)
        if self.search_start_time is not None and \
                self.state not in (FOLLOW, DONE) and \
                time.time() - self.search_start_time > SEARCH_TIME_BUDGET_SEC:
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

        if self.state in (WANDER, WANDER_MAN, CONFIRM):
            self._trigger_camera()

        if self.state == WANDER:
            self._loop_wander()
        elif self.state == WANDER_MAN:
            self._loop_wander_manual()
        elif self.state == CONFIRM:
            self._loop_confirm()
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

        if self.nav_send_time is not None and \
                time.time() - self.nav_send_time > NAV_GOAL_TIMEOUT:
            self.get_logger().warn('[Wander] Goal timeout - retry')
            self._handle_nav_failure()
            return

        if self.nav_arrived:
            self.get_logger().info('[Wander] Arrived - picking next spot')
            self.nav_sent    = False
            self.nav_arrived = False

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
        # Face the direction we're going so the camera looks forward
        # along the path of travel
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
            f'[Wander] Goal sent ({x:.2f}, {y:.2f}) '
            f'yaw={math.degrees(yaw):.0f}')

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
        if status == 4:  # SUCCEEDED
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
                f'falling back to manual drive for {WANDER_MAN_DURATION}s')
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

    # ─── WANDER_MAN (manual-drive fallback) ───────────────────────────
    def _enter_wander_manual(self):
        self.state = WANDER_MAN
        self.man_drive_start = time.time()
        self._cancel_nav()
        self._stop_motors()

    def _loop_wander_manual(self):
        elapsed = time.time() - self.man_drive_start
        if elapsed > WANDER_MAN_DURATION:
            self.get_logger().info('[ManDrive] Done - retrying Nav2')
            self.nav_failures = 0
            self.state = WANDER
            self._stop_motors()
            return

        twist = Twist()
        if self.front_distance < FOLLOW_FRONT_HARD_STOP:
            # Turn toward more open side
            if self.front_left_dist > self.front_right_dist:
                twist.angular.z = +WANDER_MAN_ANG_SPEED
            else:
                twist.angular.z = -WANDER_MAN_ANG_SPEED
            twist.linear.x = 0.0
        else:
            twist.linear.x = WANDER_MAN_LIN_SPEED
            # Gentle drift toward more open side
            if self.front_left_dist > self.front_right_dist + 0.3:
                twist.angular.z = +0.1
            elif self.front_right_dist > self.front_left_dist + 0.3:
                twist.angular.z = -0.1
        self.cmd_pub.publish(twist)
        self.get_logger().info(
            f'[ManDrive] t={elapsed:.1f}s lin={twist.linear.x:+.2f} '
            f'ang={twist.angular.z:+.2f} front={self.front_distance:.2f}',
            throttle_duration_sec=1.0)

    # ─── CONFIRM ──────────────────────────────────────────────────────
    def _enter_confirm(self):
        self.get_logger().info(
            f'[Confirm] Cat sighted - holding {CONFIRM_HOLD_SEC}s')
        self._cancel_nav()
        self._stop_motors()
        self.state         = CONFIRM
        self.confirm_start = time.time()
        self.confirm_hits  = 1
        self.status_pub.publish(String(data=f'confirming {self.target_cat}'))

    def _loop_confirm(self):
        elapsed = time.time() - self.confirm_start
        self._stop_motors()
        self._trigger_camera()

        if elapsed < CONFIRM_HOLD_SEC:
            return

        if self.confirm_hits >= CONFIRM_REQUIRE_HITS:
            self.get_logger().info(
                f'[Confirm] Verified ({self.confirm_hits} hits) -> FOLLOW')
            self._enter_follow()
        else:
            self.get_logger().info(
                f'[Confirm] Only {self.confirm_hits} hits - fluke, WANDER')
            self.cat_cx    = None
            self.cat_dist  = None
            self.last_seen = None
            self.state     = WANDER

    # ─── FOLLOW (ball-follower style) ─────────────────────────────────
    # Mapping from the assignment:
    #   error_x   = ball_col - img_width/2   (negative if on left)
    #   angular.z = -error_x / 300           (turn TOWARD the ball)
    #   dist_err  = front_dist - 1.0         (1m target)
    #     dist_err > 0  ->  forward
    #     dist_err < 0  ->  backward
    #
    # Ours: cat_cx is normalized [0..1], so error = cat_cx - 0.5.
    #   angular.z = -kp_ang * (cat_cx - 0.5)
    # Gain ramps from 0 -> FOLLOW_KP_ANG_MAX during the first
    # FOLLOW_GAIN_RAMP_SEC so the robot starts smoothly (avoiding the
    # snap-turn that smears the camera and loses the target).

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
        if self.follow_start is not None and \
                (time.time() - self.follow_start) < FOLLOW_GAIN_RAMP_SEC:
            return LOST_TIMEOUT_RAMPING
        return LOST_TIMEOUT_MOVING

    def _loop_follow(self):
        now = self.get_clock().now()

        # Lost-cat check
        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > self._follow_lost_timeout():
                self.get_logger().info(
                    f'[Follow] Cat lost ({elapsed:.1f}s) - WANDER')
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

        # Angular: center the cat (ball-follower style, with gain ramp)
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
                # Too far - approach
                if self.front_distance < FOLLOW_FRONT_HARD_STOP:
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = +min(FOLLOW_SPEED_MAX,
                                          FOLLOW_KP_LIN * dist_error)
            elif dist_error < -FOLLOW_TOLERANCE:
                # Too close - back away
                if self.back_distance < SAFE_DISTANCE:
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = -min(FOLLOW_SPEED_MAX,
                                          FOLLOW_KP_LIN * abs(dist_error))
            else:
                # Sweet spot
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
        # Let Nav2 handle obstacles itself while a goal is executing
        if self.state == WANDER and self.nav_sent and not self.nav_arrived:
            self.get_logger().warn(
                f'[Avoid] Front {self.front_distance:.2f}m but Nav2 '
                f'is active - letting Nav2 handle it')
            return

        self.pre_avoid_state = self.state
        self.state           = AVOID
        self.avoid_start     = time.time()
        self._cancel_nav()
        self._stop_motors()

        # Turn toward the more open side
        if self.front_right_dist > self.front_left_dist:
            self.avoid_turn_dir = -1   # turn right
        else:
            self.avoid_turn_dir = 1    # turn left

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
            # Return to whatever was running
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
        self.nav_failures      = 0
        self.search_start_time = None
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
