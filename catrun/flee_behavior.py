#!/usr/bin/env python3
"""
flee_behavior.py - PLAY MODE (smart-direction flee)
===================================================
Cat-style behavior using a single front CSI camera and the LiDAR.
No Nav2, no map - purely reactive with smart decisions.

  - Sit still and watch (ambush). If a cat appears, pick the BEST
    escape direction by scoring all sectors around the robot, then
    sprint that way for ~2.5m.
  - After fleeing, stop briefly and check. If cat still nearby, flee
    again (different direction). If clear, go back to ambush.
  - After several boring ambushes with no cat, wander a bit to find
    one (slow LiDAR-based forward + turn-at-walls).

State machine
-------------
  IDLE         : motors off, no target
  WAIT_AMBUSH  : stationary, scanning camera, 20s timer
  FLEE_TURN    : rotating in place to chosen escape angle
  FLEE_RUN     : sprinting forward; ends on distance/obstacle/cat
  SCAN_AFTER   : stopped, 2s look-around with camera
  SEEK_CAT     : LiDAR-based wander while scanning camera (no Nav2)
  AVOID        : sidestep obstacle, resume previous state
  DONE         : 'stop' received, motors off

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


# ─── Camera FOV (front CSI) ──────────────────────────────────────────────
CAMERA_HFOV_DEG = 63.0
CAMERA_HFOV_RAD = math.radians(CAMERA_HFOV_DEG)

# ─── Speeds ──────────────────────────────────────────────────────────────
ANGULAR_SPEED        = 0.4    # rad/s for turning
FLEE_LINEAR_SPEED    = 0.18   # m/s flee sprint speed (faster than seek)
SEEK_LINEAR_SPEED    = 0.10   # m/s seek-mode wander speed
SEEK_TURN_SPEED      = 0.4    # rad/s when wall-avoiding during seek

# ─── Ambush ──────────────────────────────────────────────────────────────
AMBUSH_DURATION_SEC  = 20.0   # how long to sit still and watch
AMBUSH_CYCLES_BEFORE_SEEK = 1 # do this many ambushes before going to seek

# ─── Flee ────────────────────────────────────────────────────────────────
FLEE_TARGET_DIST_M   = 2.5    # sprint until we've fled this far
FLEE_MAX_DURATION_S  = 8.0    # safety cap on flee time
FLEE_TURN_TOLERANCE  = math.radians(15.0)  # close enough to target heading
RECENTLY_USED_SECTORS = 2     # remember this many recent flee directions

# ─── Scan after flee ─────────────────────────────────────────────────────
SCAN_AFTER_SEC       = 2.0
SCAN_AFTER_CAT_TIMEOUT = 1.5  # cat sighting older than this = "no cat"

# ─── Seek (cat-hunting wander) ───────────────────────────────────────────
SEEK_DURATION_SEC    = 30.0   # max time spent seeking before back to ambush

# ─── Confirm (stop & verify before fleeing) ──────────────────────────────
# When YOLO publishes /cat_position, robot stops and watches for a moment
# to make sure it really is a cat (not a fluke). Need at least
# CONFIRM_FLEE_HITS additional position updates during the hold to commit
# to the flee. If only the original sighting and nothing else arrives,
# we treat it as a false alarm and resume what we were doing.
CONFIRM_FLEE_HOLD_SEC  = 1.5   # how long to stop and watch
CONFIRM_FLEE_HITS      = 2     # need this many MORE sightings during hold

# ─── Obstacle avoidance ──────────────────────────────────────────────────
SAFE_DISTANCE        = 0.30   # block forward motion when this close
AVOID_DISTANCE       = 0.40   # need this clear before resuming
AVOID_MIN_DURATION   = 1.5
LIDAR_MIN_RANGE      = 0.20   # filter chassis self-occlusion

# ─── Sector layout ───────────────────────────────────────────────────────
NUM_SECTORS          = 8      # 45 deg each


# ─── States ──────────────────────────────────────────────────────────────
IDLE          = 'IDLE'
WAIT_AMBUSH   = 'WAIT_AMBUSH'
CONFIRM_FLEE  = 'CONFIRM_FLEE'   # stopped, watching to verify cat is real
FLEE_TURN     = 'FLEE_TURN'
FLEE_RUN      = 'FLEE_RUN'
SCAN_AFTER    = 'SCAN_AFTER'
SEEK_CAT      = 'SEEK_CAT'
AVOID         = 'AVOID'
DONE          = 'DONE'


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
        self.flee_origin_x    = 0.0   # not strictly needed without odom; used optionally
        self.flee_dist_traveled = 0.0
        self.scan_after_start = None
        self.seek_start       = None

        # Cat sighting (front camera)
        self.cat_cx     = None
        self.cat_dist   = None
        self.cat_camera = 'front'   # 'front' or 'rear' - which camera saw it
        self.last_seen  = None   # rclpy Time

        # Confirmation hold state
        self.confirm_start = None
        self.confirm_hits  = 0
        self.confirm_camera = 'front'  # which camera triggered the confirmation

        # Robot heading (estimated from yaw integration of /cmd_vel since
        # we have no encoders). For relative heading targets only.
        self.estimated_yaw = 0.0
        self.last_yaw_update_t = None

        # LiDAR sectors (filled per scan)
        self.sector_clearance = [float('inf')] * NUM_SECTORS
        self.front_distance   = float('inf')
        self.front_left_dist  = float('inf')
        self.front_right_dist = float('inf')
        self.back_distance    = float('inf')

        # Recent flee sector history
        self.recent_sectors = collections.deque(maxlen=RECENTLY_USED_SECTORS)

        # Avoid
        self.avoid_turn_dir  = 1
        self.avoid_start     = None
        self.pre_avoid_state = WAIT_AMBUSH

        # SEEK turn tracking
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
            'FleeBehavior ready - smart-direction flee (no Nav2)')

    # ─── Callbacks ────────────────────────────────────────────────────
    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        # Play mode: any non-stop string means "be in play mode"
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
        # Unused - we react to /cat_position which already implies a cat
        pass

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0.05 else None
        self.last_seen = self.get_clock().now()

        # Parse which camera saw the cat from frame_id
        # ('camera_front' or 'camera_rear')
        fid = msg.header.frame_id or ''
        if fid.endswith('rear'):
            self.cat_camera = 'rear'
        else:
            self.cat_camera = 'front'

        # If we're idle (ambush/scan/seek), STOP and confirm rather than
        # snap-react. Sneaky false positives have caused the robot to
        # flee for nothing - the confirm hold filters those out.
        if self.state in (WAIT_AMBUSH, SCAN_AFTER, SEEK_CAT):
            self.get_logger().info(
                f'[/cat_position] cx={self.cat_cx:.2f} '
                f'dist={"n/a" if self.cat_dist is None else f"{self.cat_dist:.2f}m"} '
                f'cam={self.cat_camera} (state={self.state}) -> CONFIRM_FLEE')
            self._enter_confirm_flee(initial_camera=self.cat_camera)
        elif self.state == CONFIRM_FLEE:
            # Count this as a hit during the confirm hold
            self.confirm_hits += 1
            # If the camera that just saw the cat differs from the one
            # that triggered the confirm, track that too - means the
            # cat may have moved past us.
            if self.cat_camera != self.confirm_camera:
                self.get_logger().info(
                    f'[Confirm] Now seen by {self.cat_camera} camera too')
        elif self.state == FLEE_RUN:
            # Cat reappears mid-flee - cut short and re-decide
            self.get_logger().info(
                f'[/cat_position] cat reappeared during FLEE_RUN '
                f'(cam={self.cat_camera}) - re-deciding')
            self._enter_confirm_flee(initial_camera=self.cat_camera)

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

        # Cardinal distances for obstacle checks
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
            # Map to sector index 0..7. Sector 0 is centered on 0 (forward).
            sec = int(((angle + math.pi) % (2*math.pi)) / sector_size) % NUM_SECTORS
            sector_vals[sec].append(r)

        for s in range(NUM_SECTORS):
            self.sector_clearance[s] = (min(sector_vals[s])
                                        if sector_vals[s] else 0.0)

    # ─── Heading estimation (no encoders) ─────────────────────────────
    def _update_yaw_from_cmd(self, ang_z):
        """Integrate angular velocity over time to estimate heading.
        Crude but adequate for relative turns (we don't need a global
        heading - just to know we've turned about 90 degrees)."""
        now = time.time()
        if self.last_yaw_update_t is not None:
            dt = now - self.last_yaw_update_t
            self.estimated_yaw += ang_z * dt
            self.estimated_yaw = normalize_angle(self.estimated_yaw)
        self.last_yaw_update_t = now

    # ─── Sector scoring ───────────────────────────────────────────────
    def _cat_world_angle(self):
        """Estimate cat's angle in robot frame from cat_cx + camera id.

        The front camera looks down +X (angle 0).
        The rear  camera looks down -X (angle pi).

        cat_cx is normalized [0, 1]; 0.5 = center of the camera image.
          cat_cx > 0.5 -> cat to the right of camera -> negative angle in cam frame
          cat_cx < 0.5 -> cat to the left  of camera -> positive angle in cam frame

        Adjusted for rear camera, the left/right are mirrored vs. the
        robot's frame (the rear cam looks backward, so its "right" is
        the robot's "left"). We handle that here.
        """
        if self.cat_cx is None:
            return 0.0

        # Angle within the camera FOV
        offset_from_center = (0.5 - self.cat_cx) * CAMERA_HFOV_RAD

        if self.cat_camera == 'rear':
            # Rear camera points at +pi (behind). And because the rear
            # camera is mirrored relative to the robot body (left side
            # of rear cam image = robot's right side), the offset flips
            # sign when we transform into the robot frame.
            return normalize_angle(math.pi - offset_from_center)
        else:
            # Front camera: offset is directly the angle in robot frame
            return offset_from_center

    def _pick_flee_direction(self):
        """Score every sector, return the angle (rad in robot frame)
        of the best one to flee toward."""
        cat_angle = self._cat_world_angle()
        sector_size = 2 * math.pi / NUM_SECTORS

        # Most-clear sector is the reference for clearance normalization
        max_clearance = max(self.sector_clearance) or 1.0

        scores = []
        for s in range(NUM_SECTORS):
            # Sector center angle in robot frame (forward = 0)
            sec_angle = -math.pi + (s + 0.5) * sector_size

            ang_dist = angular_distance(sec_angle, cat_angle)  # [0, pi]
            clearance = self.sector_clearance[s]
            clearance_norm = clearance / max_clearance         # [0, 1]

            score = (1.0 * ang_dist
                     + 0.5 * clearance_norm * math.pi)         # scale to ~ pi

            # Heavy penalty for unsafe directions
            if clearance < SAFE_DISTANCE * 1.5:
                score -= 5.0

            # Avoid running same way twice in a row
            if s in self.recent_sectors:
                score -= 1.0

            scores.append((score, s, sec_angle, clearance, ang_dist))

        # Sort by score desc and pick the best
        scores.sort(key=lambda x: -x[0])
        best_score, best_sec, best_angle, best_clear, best_angdist = scores[0]
        self.recent_sectors.append(best_sec)

        self.get_logger().info(
            f'[Flee] Picked sector {best_sec} '
            f'(angle={math.degrees(best_angle):+.0f}deg, '
            f'clear={best_clear:.2f}m, '
            f'angdist_from_cat={math.degrees(best_angdist):.0f}deg, '
            f'score={best_score:.2f})')
        return best_angle

    # ─── Main loop ────────────────────────────────────────────────────
    def _loop(self):
        if not self.active:
            return

        # Hard obstacle override (does NOT apply during turning - we may
        # need to turn even if there's something in front of us)
        if self.state in (FLEE_RUN, SEEK_CAT):
            if self.front_distance < SAFE_DISTANCE:
                self.get_logger().warn(
                    f'[{self.state}] Front blocked '
                    f'({self.front_distance:.2f}m) -> AVOID')
                self._enter_avoid()
                return

        # Trigger camera detector while we want to see cats
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
                    f'[Ambush] {self.ambush_cycles} ambushes done '
                    f'-> SEEK_CAT')
                self._enter_seek_cat()
            else:
                self._enter_wait_ambush()   # restart the ambush timer

    # ─── CONFIRM_FLEE ─────────────────────────────────────────────────
    # When the detector first publishes /cat_position, we don't react
    # instantly. We stop and hold for CONFIRM_FLEE_HOLD_SEC, counting
    # additional position updates. If we get enough hits in the hold
    # window, the cat is real and we proceed to FLEE_TURN. Otherwise
    # it was a fluke and we go back to where we were.
    def _enter_confirm_flee(self, initial_camera='front'):
        self.state         = CONFIRM_FLEE
        self.confirm_start = time.time()
        self.confirm_hits  = 0          # the initial sighting doesn't count;
                                        # we want fresh ones during the hold
        self.confirm_camera = initial_camera
        self._stop_motors()
        self._publish_status('confirming')
        self.get_logger().info(
            f'[Confirm] Cat sighted by {initial_camera} cam - holding '
            f'{CONFIRM_FLEE_HOLD_SEC}s to verify')

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
                f'resuming WAIT_AMBUSH')
            self.cat_cx     = None
            self.cat_dist   = None
            self.last_seen  = None
            self._enter_wait_ambush()

    # ─── FLEE_TURN ────────────────────────────────────────────────────
    def _enter_flee_turn(self):
        # Pick best direction, set as target
        target_angle = self._pick_flee_direction()
        # Convert to absolute target heading
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
        # Compute yaw error
        yaw_err = normalize_angle(self.flee_target_yaw - self.estimated_yaw)

        if abs(yaw_err) < FLEE_TURN_TOLERANCE:
            self.get_logger().info(
                f'[Flee] Turn complete (err={math.degrees(yaw_err):+.0f}deg) '
                f'-> FLEE_RUN')
            self._enter_flee_run()
            return

        # Safety: if we've been turning too long, give up and start running
        if time.time() - self.flee_start > 4.0:
            self.get_logger().warn(
                f'[Flee] Turn timed out (err still '
                f'{math.degrees(yaw_err):+.0f}deg) - running anyway')
            self._enter_flee_run()
            return

        # Turn toward target
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

        # Track distance via integration of commanded velocity (no
        # encoders, so this is approximate but adequate)
        self.flee_dist_traveled += FLEE_LINEAR_SPEED * 0.1

        # End conditions
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

        # Drive forward
        twist = Twist()
        twist.linear.x = FLEE_LINEAR_SPEED
        # Slight drift toward the more open side to keep flowing through
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

        # Cat re-detected during scan?  _cb_position would re-enter FLEE_TURN
        # so this only fires when no cat is seen.
        if elapsed >= SCAN_AFTER_SEC:
            # Check freshness of last sighting; if recent, FLEE; else AMBUSH
            now = self.get_clock().now()
            if self.last_seen is not None:
                age = (now - self.last_seen).nanoseconds / 1e9
                if age < SCAN_AFTER_CAT_TIMEOUT:
                    self.get_logger().info(
                        f'[Scan] Cat sighting still fresh ({age:.1f}s) '
                        f'-> FLEE again')
                    self._enter_flee_turn()
                    return
            self.get_logger().info(
                '[Scan] All clear -> WAIT_AMBUSH')
            self._enter_wait_ambush()

    # ─── SEEK_CAT (LiDAR-based wander, no Nav2) ───────────────────────
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
                f'[Seek] Timed out after {SEEK_DURATION_SEC}s '
                f'-> back to WAIT_AMBUSH')
            self.ambush_cycles = 0
            self._enter_wait_ambush()
            return

        twist = Twist()
        # If currently turning to escape a wall, keep turning until done
        now = time.time()
        if self.seek_turn_until is not None and now < self.seek_turn_until:
            # Determine direction (set when we entered the turn)
            twist.angular.z = self._seek_turn_direction * SEEK_TURN_SPEED
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        if self.front_distance < SAFE_DISTANCE * 1.4:
            # Wall ahead - turn to the more open side for ~1 second
            self._seek_turn_direction = (+1 if self.front_left_dist
                                              > self.front_right_dist else -1)
            self.seek_turn_until = now + 1.0
            twist.angular.z = self._seek_turn_direction * SEEK_TURN_SPEED
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        # Otherwise just drive forward, drift toward more open side
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
                twist.linear.x = -0.05   # slight back-up while turning
            self.cmd_pub.publish(twist)
            self._update_yaw_from_cmd(twist.angular.z)
            return

        if self.front_distance > AVOID_DISTANCE:
            self._stop_motors()
            self.get_logger().info(f'[Avoid] Clear -> resume {self.pre_avoid_state}')
            if self.pre_avoid_state == FLEE_RUN:
                # Restart flee timer so we still sprint full duration
                self.flee_start = time.time()
                self.state = FLEE_RUN
            elif self.pre_avoid_state == SEEK_CAT:
                self.state = SEEK_CAT
                self.seek_turn_until = None
            elif self.pre_avoid_state == WAIT_AMBUSH:
                self._enter_wait_ambush()
            elif self.pre_avoid_state == CONFIRM_FLEE:
                # Restart the confirm hold so we get a full window
                self._enter_confirm_flee(initial_camera=self.cat_camera)
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