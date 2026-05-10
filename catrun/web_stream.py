#!/usr/bin/env python3
"""
flee_behavior.py - PLAY MODE
============================
Cat-mice mimic: the robot is the "mouse". It uses the REAR USB camera
(/dev/video1) to watch behind itself. When the cat detector confirms a
cat behind, the robot RUNS AWAY (forward) and tries to keep the cat
behind it (so the rear camera keeps the cat in view).

Geometry (REAR camera, inverted goal vs. seek_cat):
  * Camera faces backward (+X axis of base_link points away from camera).
  * cat_cx > 0.5 -> cat is on the camera-right = robot's LEFT.
                    To keep cat centered behind us we'd turn LEFT, but our
                    goal is to FLEE: we want the cat to STAY behind, so we
                    just turn slightly to keep it from drifting out of the
                    rear field of view. Turn LEFT (positive angular.z).
                    -> angular.z = +KP * (cat_cx - 0.5)   (rear-cam sign)
  * Distance:
      Closer than FLEE_PANIC_DIST -> RUN (max forward speed)
      Closer than FLEE_TARGET_DIST -> flee at speed proportional to
                                      how close the cat is
      Farther than FLEE_TARGET_DIST -> walk slowly forward (taunt)
      No depth -> constant flee speed

When the cat hasn't been seen for FLEE_LOST_TIMEOUT seconds, the robot
stops (cat is gone, the game is over). It does NOT try to navigate
anywhere - this node is purely reactive. Pair it with motor_control.py's
LiDAR obstacle avoidance to keep it from running into walls.
"""

import math
import sys
import signal
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# ─── tunables ────────────────────────────────────────────────────────────
KNOWN_CATS = {'eevee', 'raichu', 'pichu'}

# Speeds
FLEE_MAX_SPEED        = 0.30   # m/s, top forward speed when panicking
FLEE_CRUISE_SPEED     = 0.18   # m/s, when cat is at target distance
FLEE_TAUNT_SPEED      = 0.06   # m/s, when cat is far - slow forward "tease"
FLEE_NO_DEPTH_SPEED   = 0.15   # m/s, when distance unknown
ANGULAR_MAX           = 0.6    # rad/s, max turning rate

# Distances
FLEE_PANIC_DIST   = 0.50   # cat closer than this -> max speed
FLEE_TARGET_DIST  = 1.20   # try to keep cat at least this far behind
FLEE_TAUNT_DIST   = 2.00   # cat farther than this -> slow taunt

# Steering
FLEE_KP_ANG       = 1.2    # gain on cx error

# Timeouts
FLEE_LOST_TIMEOUT = 3.0    # cat unseen this long -> stop
TRIGGER_PERIOD    = 0.4    # how often to nudge the cat detector

# Obstacle safety (we're going forward fast, watch ahead)
SAFE_FRONT_DIST   = 0.30   # bail out if something this close ahead
SLOW_FRONT_DIST   = 0.60   # cap speed if something this close ahead
SAFE_SIDE_DIST    = 0.20

# states
IDLE     = 'IDLE'
WAITING  = 'WAITING'    # active but no cat seen yet (just hold still)
FLEEING  = 'FLEEING'    # cat seen behind - run!


class FleeBehavior(Node):

    def __init__(self):
        super().__init__('flee_behavior')

        self.state      = IDLE
        self.target_cat = None
        self.active     = False

        self.cat_cx    = None
        self.cat_dist  = None
        self.last_seen = None

        self.front_distance = float('inf')
        self.left_distance  = float('inf')
        self.right_distance = float('inf')

        self.last_trigger_time = 0.0

        self.create_subscription(String,       '/cat_target',   self._cb_target,   10)
        self.create_subscription(String,       '/cat_identity', self._cb_identity, 10)
        self.create_subscription(PointStamped, '/cat_position', self._cb_position, 10)
        self.create_subscription(LaserScan,    '/scan',         self._cb_scan,     10)

        self.cmd_pub    = self.create_publisher(Twist,  '/cmd_vel',              10)
        self.cam_pub    = self.create_publisher(String, '/camera_check_trigger', 10)
        self.status_pub = self.create_publisher(String, '/seek_status',          10)

        self.create_timer(0.1, self._loop)

        self.get_logger().info('=' * 50)
        self.get_logger().info('FleeBehavior ready - PLAY MODE (rear camera)')
        self.get_logger().info(f'  panic<{FLEE_PANIC_DIST}m  '
                               f'target={FLEE_TARGET_DIST}m  '
                               f'taunt>{FLEE_TAUNT_DIST}m')
        self.get_logger().info('=' * 50)

    # ─── Callbacks ────────────────────────────────────────────────────
    def _cb_target(self, msg: String):
        text = msg.data.strip().lower()
        if text in ('', 'stop', 'none'):
            self._reset()
            return
        # Play mode flees from ANY cat - accept any non-empty target.
        # We still record the requested name for log readability, but
        # we don't filter detections against it.
        self.target_cat = text if text in KNOWN_CATS else 'any'
        self.active     = True
        self.state      = WAITING
        self.cat_cx     = None
        self.cat_dist   = None
        self.last_seen  = None
        self.get_logger().info(
            f'Play mode armed (requested: {text}) - any cat triggers flee')
        self._publish_status(f'fleeing from any cat (requested: {text})')

    def _cb_identity(self, msg: String):
        # Any cat identity triggers - we don't filter by name in play mode.
        # This callback is just for logging the first sighting.
        name = msg.data.strip().lower()
        if not self.active:
            return
        if self.state == WAITING:
            self.get_logger().info(f"Spotted '{name}' behind us - fleeing!")

    def _cb_position(self, msg: PointStamped):
        self.cat_cx    = msg.point.x
        self.cat_dist  = msg.point.z if msg.point.z > 0.05 else None
        self.last_seen = self.get_clock().now()
        if self.state == WAITING:
            self.state = FLEEING

    def _cb_scan(self, msg: LaserScan):
        ranges = msg.ranges
        n      = len(ranges)

        def safe_min(indices):
            vals = [ranges[i] for i in indices
                    if msg.range_min < ranges[i] < msg.range_max
                    and math.isfinite(ranges[i])]
            return min(vals) if vals else float('inf')

        # Front sector (we're driving forward when fleeing)
        self.front_distance = safe_min(
            list(range(0, n//12)) + list(range(11*n//12, n)))
        self.left_distance  = safe_min(list(range(n//4,    5*n//12)))
        self.right_distance = safe_min(list(range(7*n//12, 3*n//4)))

    # ─── Main loop ────────────────────────────────────────────────────
    def _loop(self):
        if not self.active or self.target_cat is None:
            return

        # Nudge the detector to keep producing
        self._trigger_camera()

        if self.state == WAITING:
            # Cat hasn't been spotted yet - hold still and look behind
            self._stop_motors()
            return

        if self.state == FLEEING:
            self._loop_flee()

    def _loop_flee(self):
        now = self.get_clock().now()

        # Check whether we've lost the cat
        if self.last_seen is not None:
            elapsed = (now - self.last_seen).nanoseconds / 1e9
            if elapsed > FLEE_LOST_TIMEOUT:
                self.get_logger().info(
                    '[Flee] Cat out of sight - holding position '
                    '(escape successful?)')
                self._stop_motors()
                self.state    = WAITING
                self.cat_cx   = None
                self.cat_dist = None
                return

        if self.cat_cx is None:
            self._stop_motors()
            return

        twist = Twist()

        # ─── Angular: keep cat centered in REAR camera ────────────────
        # Rear-cam: cat_cx > 0.5 means cat appears on the camera-right,
        # which is the robot's LEFT side. To re-center the cat in the
        # rear view we need to turn LEFT.  Turning LEFT = positive
        # angular.z. So: angular.z = +KP * (cat_cx - 0.5).
        # (This is the OPPOSITE sign from the forward-camera ball
        # follower, which is correct for a rear-facing cam.)
        cx_error = self.cat_cx - 0.5
        ang = +FLEE_KP_ANG * cx_error
        twist.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, ang))

        # ─── Linear: run AWAY from cat ────────────────────────────────
        # Rear camera looks behind, robot's forward (+X) is away from cat,
        # so to flee we drive linear.x > 0 (forward).
        if self.cat_dist is None:
            base_speed = FLEE_NO_DEPTH_SPEED
        elif self.cat_dist < FLEE_PANIC_DIST:
            # Cat is right on us - panic!
            base_speed = FLEE_MAX_SPEED
        elif self.cat_dist < FLEE_TARGET_DIST:
            # Cat is closing in - flee proportional to how close
            t = ((FLEE_TARGET_DIST - self.cat_dist)
                 / max(0.01, FLEE_TARGET_DIST - FLEE_PANIC_DIST))
            base_speed = FLEE_CRUISE_SPEED + t * (FLEE_MAX_SPEED - FLEE_CRUISE_SPEED)
        elif self.cat_dist < FLEE_TAUNT_DIST:
            # Cat at comfortable distance - cruise
            base_speed = FLEE_CRUISE_SPEED
        else:
            # Cat is way back - taunt slowly so it keeps coming
            base_speed = FLEE_TAUNT_SPEED

        # Cap speed if there's something close ahead
        if self.front_distance < SAFE_FRONT_DIST:
            self.get_logger().warn(
                f'[Flee] Front blocked ({self.front_distance:.2f}m) - '
                f'turning only',
                throttle_duration_sec=1.0)
            twist.linear.x = 0.0
            # Bias the turn AWAY from the closer side wall
            if self.left_distance < SAFE_SIDE_DIST:
                twist.angular.z = -ANGULAR_MAX
            elif self.right_distance < SAFE_SIDE_DIST:
                twist.angular.z = +ANGULAR_MAX
        elif self.front_distance < SLOW_FRONT_DIST:
            # Scale speed down linearly between SAFE and SLOW
            scale = ((self.front_distance - SAFE_FRONT_DIST)
                     / max(0.01, SLOW_FRONT_DIST - SAFE_FRONT_DIST))
            twist.linear.x = base_speed * scale
        else:
            twist.linear.x = base_speed

        self.cmd_pub.publish(twist)

        dist_str = f'{self.cat_dist:.2f}m' if self.cat_dist is not None else 'n/a'
        self.get_logger().info(
            f'[Flee] cx={self.cat_cx:.2f} cat_dist={dist_str} '
            f'lin={twist.linear.x:+.2f} ang={twist.angular.z:+.2f} '
            f'(front={self.front_distance:.2f})',
            throttle_duration_sec=0.5)

    # ─── Helpers ──────────────────────────────────────────────────────
    def _trigger_camera(self):
        now = time.time()
        if now - self.last_trigger_time < TRIGGER_PERIOD:
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
        self._stop_motors()
        self.state      = IDLE
        self.active     = False
        self.target_cat = None
        self.cat_cx     = None
        self.cat_dist   = None
        self.last_seen  = None
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
