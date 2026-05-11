#!/usr/bin/env python3
"""
flee_behavior.py - PLAY MODE (smart-direction flee + stop-and-confirm)
======================================================================
Cat-style behavior using LiDAR + dual cameras. Single state machine,
no Nav2.

  - Sit still and watch (ambush). If a cat appears, STOP and confirm
    for ~1.5s. If still seen, pick the BEST escape direction and
    sprint that way.
  - After fleeing, stop briefly and check. If cat still nearby, flee
    again (different direction). If clear, go back to ambush.
  - After several boring ambushes with no cat, wander a bit to find
    one (slow LiDAR-based forward + turn-at-walls).

State machine
-------------
  IDLE         : motors off, no target
  WAIT_AMBUSH  : stationary, scanning camera, 20s timer
  CONFIRM_FLEE : stopped, watching ~1.5s to verify cat is real
  FLEE_TURN    : rotating in place to chosen escape angle
  FLEE_RUN     : sprinting forward; ends on distance/obstacle/cat
  SCAN_AFTER   : stopped, 2s look-around with camera
  SEEK_CAT     : LiDAR-based wander while scanning camera (no Nav2)
  AVOID        : sidestep obstacle, resume previous state
  DONE         : 'stop' received, motors off

Camera awareness
----------------
cat_detector publishes /cat_position with frame_id encoding which
camera saw the cat ('camera_front' or 'camera_rear'). We use that
to compute the cat's angle correctly in the robot frame:
  - front cam: cat is in front, angle from cat_cx as before
  - rear  cam: cat is BEHIND, angle around pi
The smart-direction scoring then picks a direction AWAY from where
the cat actually is.

Smart escape-direction scoring (no fixed pattern)
-------------------------------------------------
On each FLEE_TURN entry we evaluate 8 sectors covering 360 degrees:
  score(sector) =   1.0 * angular_dist_from_cat   (further from cat = better)
                  + 0.5 * sector_clearance_norm   (more open space = better)
                  - 1.0 * recently_used_penalty   (avoid same way twice)
The highest-scoring sector becomes the flee direction.
"""

import math
import sys
import signal
import time
import collections

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


# ─── Camera FOV (assumes both cameras have same FOV) ─────────────────────
CAMERA_HFOV_DEG = 63.0
CAMERA_HFOV_RAD = math.radians(CAMERA_HFOV_DEG)

# ─── Speeds ──────────────────────────────────────────────────────────────
ANGULAR_SPEED        = 0.4    # rad/s for turning
FLEE_LINEAR_SPEED    = 0.18   # m/s flee sprint speed (faster than seek)
SEEK_LINEAR_SPEED    = 0.10   # m/s seek-mode wander speed
SEEK_TURN_SPEED      = 0.4    # rad/s when wall-avoiding during seek

# ─── Ambush ──────────────────────────────────────────────────────────────
AMBUSH_DURATION_SEC  = 20.0
AMBUSH_CYCLES_BEFORE_SEEK = 1

# ─── Confirm (stop & verify before fleeing) ──────────────────────────────
# When YOLO publishes /cat_position the robot stops and watches for a
# moment to confirm it really is a cat (not a fluke YOLO frame). Need
# CONFIRM_FLEE_HITS additional position updates during the hold to
# commit to the flee; otherwise treat it as a false alarm.
CONFIRM_FLEE_HOLD_SEC  = 1.5
CONFIRM_FLEE_HITS      = 2

# ─── Flee ────────────────────────────────────────────────────────────────
FLEE_TARGET_DIST_M   = 2.5
FLEE_MAX_DURATION_S  = 8.0
FLEE_TURN_TOLERANCE  = math.radians(15.0)
RECENTLY_USED_SECTORS = 2

# ─── Scan after flee ─────────────────────────────────────────────────────
SCAN_AFTER_SEC       = 2.0
SCAN_AFTER_CAT_TIMEOUT = 1.5

# ─── Seek ────────────────────────────────────────────────────────────────
SEEK_DURATION_SEC    = 30.0

# ─── Obstacle avoidance ──────────────────────────────────────────────────
SAFE_DISTANCE        = 0.30
AVOID_DISTANCE       = 0.40
AVOID_MIN_DURATION   = 1.5
LIDAR_MIN_RANGE      = 0.20

# ─── Sectors ─────────────────────────────────────────────────────────────
NUM_SECTORS          = 8


# ─── States ──────────────────────────────────────────────────────────────
IDLE         = 'IDLE'
WAIT_AMBUSH  = 'WAIT_AMBUSH'
CONFIRM_FLEE = 'CONFIRM_FLEE'
FLEE_TURN    = 'FLEE_TURN'
FLEE_RUN     = 'FLEE_RUN'
SCAN_AFTER   = 'SCAN_AFTER'
SEEK_CAT     = 'SEEK_CAT'
AVOID        = 'AVOID'
DONE         = 'DONE'


def normalize_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def angular_distance(a1, a2):
    """Smallest unsigned distance between two angles, [0, pi]."""
    return abs(normalize_angle(a1 - a2))


class FleeBehavior(Node):

    def __init__(self):
        super().__init__('flee_behavior')

        # ─── State ────────────────────────────────────────────────────
        self.state            = IDLE
        self.active           = False
        self.ambush_start     = None
        self.ambush_cycles    = 0
        self.flee_start       = None
        self.flee_target_yaw  = None
        self.flee_dist_traveled = 0.0
        self.scan_after_start = None
        self.seek_start       = None

        # Cat sighting
        self.cat_cx     = None
        self.cat_dist   = None
        self.cat_camera = 'front'   # 'front' or 'rear' - which camera saw it
        self.last_seen  = None

        # Confirmation state
        self.confirm_start  = None
        self.confirm_hits   = 0
        self.confirm_camera = 'front'

        # Heading (estimated from cmd_vel)
        self.estimated_yaw = 0.0
        self.last_yaw_update_t = None

        # LiDAR
        self.sector_clearance = [float('inf')] * NUM_SECTORS
        self.front_distance   = float('inf')
        self.front_left_dist  = float('inf')
        self.front_right_dist = float('inf')
        self.back_distance    = float('inf')

        # Recent flee sectors
        self.recent_sectors = collections.deque(maxlen=RECENTLY_USED_SECTORS)

        # Avoid
        self.avoid_turn_dir  = 1
        self.avoid_start     = None
        self.pre_avoid_state = WAIT_AMBUSH

        # Seek
        self.seek_turn_until = None

        self.last_trigger_time = 0.0

        # ─── ROS interface ────────────────────────────────────────────
        self.create_subscription(String,       '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,       '/cat_identity', self._cb_identity, 10)
        self.create_subscription(PointStamped, '/cat_position', self._cb_position, 10)
        self.create_subscription(LaserScan,    '/scan',         self._cb_scan,     10)

        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',              10)
        self.cam_pub    = self.create_publisher(String, '/camera_check_trigger', 10)
        self.target_pub = self.create_publisher(String, '/cat_target',           10)
        self.status_pub = self.create_publisher(String, '/seek_status',          10)

        self.create_timer(0.1, self._loop)
        self.get_logger().info(
            'FleeBehavior ready - dual-cam smart-direction flee w/ confirm')

    # ─── Callbacks ────────────────────────────────────────────────────
    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        self.active           = True
        self.ambush_cycles    = 0
        self.cat_cx           = None
        self.cat_dist         = None
        self.last_seen        = None
        self.estimated_yaw    = 0.0
        self.recent_sectors.clear()
        self._enter_wait_ambush()
        self.get_logger().info(f'Play mode active (target={text})')

    def _cb_identity(self, msg: String):
        # We rely on /cat_position arriving; identity is just for logging.
        pass

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0.05 else None
        self.last_seen = self.get_clock().now()

        # Parse which camera saw the cat from frame_id
        # cat_detector publishes 'camera_front' or 'camera_rear'
        fid = msg.header.frame_id or ''
        if fid.endswith('rear'):
            self.cat_camera = 'rear'
        else:
            self.cat_camera = 'front'

        # In idle states (ambush/seek/scan), STOP and confirm rather
        # than snap-react. This prevents fleeing on a single fluke
        # YOLO frame.
        if self.state in (WAIT_AMBUSH, SCAN_AFTER, SEEK_CAT):
            self.get_logger().info(
                f'[/cat_position] cx={self.cat_cx:.2f} '
                f'dist={"n/a" if self.cat_dist is None else f"{self.cat_dist:.2f}m"} '
                f'cam={self.cat_camera} (state={self.state}) -> CONFIRM_FLEE')
            self._enter_confirm_flee(self.cat_camera)
        elif self.state == CONFIRM_FLEE:
            # Count this as an additional hit during the hold
            self.confirm_hits += 1
            if self.cat_camera != self.confirm_camera:
                self.get_logger().info(
                    f'[Confirm] Now also seen by {self.cat_camera} cam')
        elif self.state == FLEE_RUN:
            # Cat reappears mid-flee - cut short and re-decide.
            # (Mirroring back to CONFIRM_FLEE would create flapping;
            # we go straight to FLEE_TURN with new direction.)
            self.get_logger().info(
                f'[/cat_position] cat reappeared during FLEE_RUN '
                f'(cam={self.cat_camera}) - re-deciding direction')
            self._enter_flee_turn()

    def _cb_scan(self, msg: LaserScan):
        ranges  = msg.ranges
        n       = len(ranges)
        eff_min = max(msg.range_min, LIDAR_MIN_RANGE)
        rmax    = msg.range_max
        amin    = msg.angle_min
        ainc    = msg.angle_increment

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if eff_min < ranges[i] < rmax
                    and math.isfinite(ranges[i])]
            return min(vals) if vals else float('inf')

        self.front_distance   = safe_min(
            list(range(0, n//12)) + list(range(11*n//12, n)))
        self.front_left_dist  = safe_min(list(range(n//12, n//4)))
        self.front_right_dist = safe_min(list(range(3*n//4, 11*n//12)))
        self.back_distance    = safe_min(list(range(5*n//12, 7*n//12)))

        # Sector clearances (8 sectors of 45 deg)
        sector_vals = [[] for _ in range(NUM_SECTORS)]
        sector_size = 2 * math.pi / NUM_SECTORS
        for i in range(n):
            r = ranges[i]
            if not (eff_min < r < rmax) or not math.isfinite(r):
                continue
            angle = normalize_angle(amin + i * ainc)
            sec = int(((angle + math.pi) % (2*math.pi)) / sector_size) % NUM_SECTORS
            sector_vals[sec].append(r)

        for s in range(NUM_SECTORS):
            self.sector_clearance[s] = (min(sector_vals[s])
                                        if sector_vals[s] else 0.0)

    # ─── Heading estimation (no encoders) ─────────────────────────────
    def _update_yaw_from_cmd(self, ang_z):
        now = time.time()
        if self.last_yaw_update_t is not None:
            dt = now - self.last_yaw_update_t
            self.estimated_yaw += ang_z * dt
            self.estimated_yaw = normalize_angle(self.estimated_yaw)
        self.last_yaw_update_t = now

    # ─── Cat angle in robot frame ─────────────────────────────────────
    def _cat_world_angle(self):
        """Estimate cat's angle in robot frame from cat_cx + which camera
        saw it.

        Front cam looks at angle 0 (forward).  cat_cx > 0.5 -> cat on
        right of robot -> negative angle.

        Rear cam looks at angle pi (backward).  Because the rear cam is
        mirrored vs the robot body (rear cam's "left" is the robot's
        "right"), the offset sign flips when transformed to robot frame.
        """
        if self.cat_cx is None:
            return 0.0
        offset_from_center = (0.5 - self.cat_cx) * CAMERA_HFOV_RAD
        if self.cat_camera == 'rear':
            return normalize_angle(math.pi - offset_from_center)
        return offset_from_center

    def _pick_flee_direction(self):
        """Score 8 sectors, return the angle (rad in robot frame) of
        the best one to flee toward."""
        cat_angle = self._cat_world_angle()
        sector_size = 2 * math.pi / NUM_SECTORS

        max_clearance = max(self.sector_clearance) or 1.0

        scores = []
        for s in range(NUM_SECTORS):
            sec_angle = -math.pi + (s + 0.5) * sector_size

            ang_dist = angular_distance(sec_angle, cat_angle)
            clearance = self.sector_clearance[s]
            clearance_norm = clearance / max_clearance

            score = (1.0 * ang_dist
                     + 0.5 * clearance_norm * math.pi)

            if clearance < SAFE_DISTANCE * 1.5:
                score -= 5.0

            if s in self.recent_sectors:
                score -= 1.0

            scores.append((score, s, sec_angle, clearance, ang_dist))

        scores.sort(key=lambda x: -x[0])
        best_score, best_sec, best_angle, best_clear, best_angdist = scores[0]
        self.recent_sectors.append(best_sec)

        self.get_logger().info(
            f'[Flee] Picked sector {best_sec} '
            f'(angle={math.degrees(best_angle):+.0f}deg, '
            f'clear={best_clear:.2f}m, '
            f'angdist_from_cat={math.degrees(best_angdist):.0f}deg, '
            f'cat_at={math.degrees(cat_angle):+.0f}deg cam={self.cat_camera}, '
            f'score={best_score:.2f})')
        return best_angle

    # ─── Main loop ────────────────────────────────────────────────────
    def _loop(self):
        if not self.active:
            return

        # Hard obstacle override (only when moving)
        if self.state in (FLEE_RUN, SEEK_CAT):
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'[{self.state}] Front blocked '
                    f'({self.front_distance:.2f}m) -> AVOID')
                self._enter_avoid()
                return

        if self.state in (WAIT_AMBUSH, SCAN_AFTER, SEEK_CAT, CONFIRM_FLEE):
            self._trigger_camera()

        if self.state == WAIT_AMBUSH:
            self._loop_wait_ambush()
        elif self.state == CONFIRM_FLEE:
            self._loop_confirm_flee()
        elif self.state == FLEE_TURN:
            self._loop_flee_turn()
        elif self.state == FLEE_RUN:
            self._loop_flee_run()
        elif self.state == SCAN_AFTER:
            self._loop_scan_after()
        elif self.state == SEEK_CAT:
            self._loop_seek_cat()
        elif self.state == AVOID:
            self._loop_avoid()

    # ─── WAIT_AMBUSH ──────────────────────────────────────────────────
    def _enter_wait_ambush(self):
        self.state = WAIT_AMBUSH
        self.ambush_start = time.time()
        self._stop_motors()
        self._publish_status('ambushing')
        self.get_logger().info(
            f'[Ambush] Sitting still {AMBUSH_DURATION_SEC}s, watching for cats')

    def _loop_wait_ambush(self):
        elapsed = time.time() - self.ambush_start
        self._stop_motors()

        if int(elapsed) % 5 == 0 and elapsed - int(elapsed) < 0.11:
            self.get_logger().info(
                f'[Ambush] {elapsed:.0f}/{AMBUSH_DURATION_SEC:.0f}s '
                f'(cycle {self.ambush_cycles+1})')

        if elapsed >= AMBUSH_DURATION_SEC:
            self.ambush_cycles += 1
            if self.ambush_cycles >= AMBUSH_CYCLES_BEFORE_SEEK:
                self.get_logger().info(
                    f'[Ambush] {self.ambush_cycles} ambushes done -> SEEK_CAT')
                self._enter_seek_cat()
            else:
                self._enter_wait_ambush()

    # ─── CONFIRM_FLEE ─────────────────────────────────────────────────
    def _enter_confirm_flee(self, initial_camera='front'):
        self.state         = CONFIRM_FLEE
        self.confirm_start = time.time()
        self.confirm_hits  = 0    # don't count the triggering sighting;
                                  # need NEW sightings during the hold
        self.confirm_camera = initial_camera
        self._stop_motors()
        self._publish_status('confirming')
        self.get_logger().info(
            f'[Confirm] Cat sighted by {initial_camera} cam - '
            f'holding {CONFIRM_FLEE_HOLD_SEC}s to verify')

    def _loop_confirm_flee(self):
        elapsed = time.time() - self.confirm_start
        self._stop_motors()
        self._trigger_camera()

        if elapsed < CONFIRM_FLEE_HOLD_SEC:
            return

        if self.confirm_hits >= CONFIRM_FLEE_HITS:
            self.get_logger().info(
                f'[Confirm] Verified ({self.confirm_hits} extra hits) -> FLEE')
            self._enter_flee_turn()
        else:
            self.get_logger().info(
                f'[Confirm] Only {self.confirm_hits} extra hits - fluke, '
                f'back to WAIT_AMBUSH')
            self.cat_cx    = None
            self.cat_dist  = None
            self.last_seen = None
            self._enter_wait_ambush()

    # ─── FLEE_TURN ────────────────────────────────────────────────────
    def _enter_flee_turn(self):
        target_angle = self._pick_flee_direction()
        self.flee_target_yaw = normalize_angle(
            self.estimated_yaw + target_angle)
        self.state = FLEE_TURN
        self.flee_start = time.time()
        self._stop_motors()
        self._publish_status('fleeing')
        self.get_logger().info(
            f'[Flee] Turning to relative angle '
            f'{math.degrees(target_angle):+.0f}deg '
            f'(target_yaw={math.degrees(self.flee_target_yaw):+.0f}deg)')

    def _loop_flee_turn(self):
        yaw_err = normalize_angle(self.flee_target_yaw - self.estimated_yaw)

        if abs(yaw_err) < FLEE_TURN_TOLERANCE:
            self.get_logger().info(
                f'[Flee] Turn complete (err={math.degrees(yaw_err):+.0f}deg) '
                f'-> FLEE_RUN')
            self._enter_flee_run()
            return

        if time.time() - self.flee_start > 4.0:
            self.get_logger().warn(
                f'[Flee] Turn timed out (err still '
                f'{math.degrees(yaw_err):+.0f}deg) - running anyway')
            self._enter_flee_run()
            return

        twist = Twist()
        twist.angular.z = ANGULAR_SPEED if yaw_err > 0 else -ANGULAR_SPEED
        self.cmd_pub.publish(twist)
        self._update_yaw_from_cmd(twist.angular.z)

    # ─── FLEE_RUN ─────────────────────────────────────────────────────
    def _enter_flee_run(self):
        self.state = FLEE_RUN
        self.flee_start = time.time()
        self.flee_dist_traveled = 0.0
        self._publish_status('fleeing')

    def _loop_flee_run(self):
        elapsed = time.time() - self.flee_start

        self.flee_dist_traveled += FLEE_LINEAR_SPEED * 0.1

        if self.flee_dist_traveled >= FLEE_TARGET_DIST_M:
            self.get_logger().info(
                f'[Flee] Reached {FLEE_TARGET_DIST_M}m -> SCAN_AFTER')
            self._enter_scan_after()
            return
        if elapsed > FLEE_MAX_DURATION_S:
            self.get_logger().info(
                f'[Flee] Max time {FLEE_MAX_DURATION_S}s -> SCAN_AFTER')
            self._enter_scan_after()
            return

        twist = Twist()
        twist.linear.x = FLEE_LINEAR_SPEED
        if self.front_left_dist > self.front_right_dist + 0.3:
            twist.angular.z = +0.08
        elif self.front_right_dist > self.front_left_dist + 0.3:
            twist.angular.z = -0.08
        self.cmd_pub.publish(twist)
        self._update_yaw_from_cmd(twist.angular.z)

        self.get_logger().info(
            f'[Flee-run] t={elapsed:.1f}s d~{self.flee_dist_traveled:.2f}m '
            f'front={self.front_distance:.2f}',
            throttle_duration_sec=1.0)

    # ─── SCAN_AFTER ───────────────────────────────────────────────────
    def _enter_scan_after(self):
        self.state = SCAN_AFTER
        self.scan_after_start = time.time()
        self._stop_motors()
        self._publish_status('checking')
        self.get_logger().info(
            f'[Scan] Stopped, looking around for {SCAN_AFTER_SEC}s')

    def _loop_scan_after(self):
        elapsed = time.time() - self.scan_after_start
        self._stop_motors()
        self._trigger_camera()

        if elapsed >= SCAN_AFTER_SEC:
            now = self.get_clock().now()
            if self.last_seen is not None:
                age = (now - self.last_seen).nanoseconds / 1e9
                if age < SCAN_AFTER_CAT_TIMEOUT:
                    self.get_logger().info(
                        f'[Scan] Cat sighting still fresh ({age:.1f}s) '
                        f'-> CONFIRM_FLEE')
                    self._enter_confirm_flee(self.cat_camera)
                    return
            self.get_logger().info('[Scan] All clear -> WAIT_AMBUSH')
            self._enter_wait_ambush()

    # ─── SEEK_CAT ─────────────────────────────────────────────────────
    def _enter_seek_cat(self):
        self.state = SEEK_CAT
        self.seek_start = time.time()
        self.seek_turn_until = None
        self._publish_status('seeking')
        self.get_logger().info(
            f'[Seek] Wandering to find a cat (max {SEEK_DURATION_SEC}s)')

    def _loop_seek_cat(self):
        elapsed = time.time() - self.seek_start
        if elapsed > SEEK_DURATION_SEC:
            self.get_logger().info(
                f'[Seek] Timed out after {SEEK_DURATION_SEC}s -> WAIT_AMBUSH')
            self.ambush_cycles = 0
            self._enter_wait_ambush()
            return

        twist = Twist()
        now = time.time()
        if self.seek_turn_until is not None and now < self.seek_turn_until:
            twist.angular.z = self._seek_turn_direction * SEEK_TURN_SPEED
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        if self.front_distance < SAFE_DISTANCE * 1.4:
            self._seek_turn_direction = (+1 if self.front_left_dist
                                              > self.front_right_dist else -1)
            self.seek_turn_until = now + 1.0
            twist.angular.z = self._seek_turn_direction * SEEK_TURN_SPEED
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        self.seek_turn_until = None
        twist.linear.x = SEEK_LINEAR_SPEED
        if self.front_left_dist > self.front_right_dist + 0.3:
            twist.angular.z = +0.1
        elif self.front_right_dist > self.front_left_dist + 0.3:
            twist.angular.z = -0.1
        self.cmd_pub.publish(twist)
        self._update_yaw_from_cmd(twist.angular.z)

        self.get_logger().info(
            f'[Seek] t={elapsed:.0f}s lin={twist.linear.x:.2f} '
            f'ang={twist.angular.z:+.2f} front={self.front_distance:.2f}',
            throttle_duration_sec=2.0)

    # ─── AVOID ────────────────────────────────────────────────────────
    def _enter_avoid(self):
        self.pre_avoid_state = self.state
        self.state           = AVOID
        self.avoid_start     = time.time()
        self._stop_motors()

        if self.front_right_dist > self.front_left_dist:
            self.avoid_turn_dir = -1
        else:
            self.avoid_turn_dir = 1

        self.get_logger().info(
            f'[Avoid] front={self.front_distance:.2f}m -> '
            f'{"LEFT" if self.avoid_turn_dir > 0 else "RIGHT"} '
            f'(was {self.pre_avoid_state})')

    def _loop_avoid(self):
        elapsed = time.time() - self.avoid_start

        if elapsed < AVOID_MIN_DURATION:
            twist = Twist()
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            if elapsed > 0.8:
                twist.linear.x = -0.05
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        if self.front_distance > AVOID_DISTANCE:
            self._stop_motors()
            self.get_logger().info(
                f'[Avoid] Clear -> resume {self.pre_avoid_state}')
            if self.pre_avoid_state == FLEE_RUN:
                self.flee_start = time.time()
                self.state = FLEE_RUN
            elif self.pre_avoid_state == SEEK_CAT:
                self.state = SEEK_CAT
                self.seek_turn_until = None
            elif self.pre_avoid_state == WAIT_AMBUSH:
                self._enter_wait_ambush()
            elif self.pre_avoid_state == CONFIRM_FLEE:
                self._enter_confirm_flee(self.cat_camera)
            else:
                self.state = self.pre_avoid_state
            return

        if elapsed < AVOID_MIN_DURATION * 2.5:
            twist = Twist()
            twist.angular.z = self.avoid_turn_dir * ANGULAR_SPEED
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
        else:
            self.avoid_turn_dir = -self.avoid_turn_dir
            self.avoid_start = time.time()
            self.get_logger().warn('[Avoid] Still stuck - flipping direction')

    # ─── Helpers ──────────────────────────────────────────────────────
    def _trigger_camera(self):
        now = time.time()
        if now - self.last_trigger_time < 0.4:
            return
        self.last_trigger_time = now
        msg = String()
        msg.data = 'any'
        self.cam_pub.publish(msg)

    def _stop_motors(self):
        self.cmd_pub.publish(Twist())

    def _publish_status(self, s):
        self.status_pub.publish(String(data=s))

    def _reset(self):
        self._stop_motors()
        self.state             = IDLE
        self.active            = False
        self.cat_cx            = None
        self.cat_dist          = None
        self.cat_camera        = 'front'
        self.last_seen         = None
        self.flee_start        = None
        self.flee_target_yaw   = None
        self.scan_after_start  = None
        self.seek_start        = None
        self.ambush_start      = None
        self.ambush_cycles     = 0
        self.confirm_start     = None
        self.confirm_hits      = 0
        self.recent_sectors.clear()
        self._publish_status('stopped')


def _safe_cleanup(node):
    if node is not None:
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
        node = FleeBehavior()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[flee_behavior] error: {e}", file=sys.stderr)
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
        print("[flee_behavior] clean shutdown complete")


if __name__ == '__main__':
    main()