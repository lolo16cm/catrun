#!/usr/bin/env python3
"""
flee_behavior.py — AI-planned flee behavior
=============================================
When cat is detected:
1. Gets current robot position from /odom or /amcl_pose
2. Gets cat position from /cat_position
3. Calls Claude AI to plan the best escape route considering:
   - Cat's current position
   - Robot's current position  
   - Available hiding spots
   - Which direction is AWAY from the cat
4. Navigates to AI-chosen hiding spot via Nav2
"""

import json
import math
import threading
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry

# ── Hiding spots on your map ──────────────────────────────────────────────────
# Update these to real coordinates from your map
HIDING_SPOTS = [
    {'name': 'L1-Mirror',  'x':  -0.91, 'y':  0.33},
    {'name': 'L2-Bath',  'x': 2.87,   'y':  1.057},
    {'name': 'L3-TV',  'x':  1.396, 'y': 1.127},
    {'name': 'L5-Nail',      'x':  1.1,   'y':  -1.037},
    {'name': 'L6-CoffeeTable', 'x': -0.178, 'y': -1.31},
]

# ── Claude API ────────────────────────────────────────────────────────────────
CLAUDE_API_URL = 'https://api.anthropic.com/v1/messages'
CLAUDE_MODEL   = 'claude-sonnet-4-20250514'

# Minimum distance to move away from cat
MIN_FLEE_DIST  = 0.5   # m — don't flee if already far enough
FLEE_COOLDOWN  = 3.0   # s — minimum time between flee decisions


class FleeBehavior(Node):

    def __init__(self):
        super().__init__('flee_behavior')

        # Robot state
        self.robot_x     = 0.0
        self.robot_y     = 0.0
        self.cat_cx      = None   # normalised [0,1]
        self.cat_dist    = None   # metres from LiDAR
        self.is_fleeing  = False
        self.last_flee   = 0.0
        self.current_goal = None

        # Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Subscriptions
        self.create_subscription(
            PointStamped, '/cat_position', self.cat_cb, 10)
        self.create_subscription(
            Odometry, '/odom_rf2o', self.odom_cb, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)
        self.create_subscription(
            String, '/cat_target', self.target_cb, 10)

        # Status publisher
        self.status_pub = self.create_publisher(String, '/flee_status', 10)

        self.get_logger().info('FleeBehavior ready — AI-planned escape!')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def odom_cb(self, msg: Odometry):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def amcl_cb(self, msg: PoseWithCovarianceStamped):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def target_cb(self, msg: String):
        if msg.data.strip().lower() in ('stop', 'none', ''):
            self.is_fleeing = False
            if self.current_goal is not None:
                self.current_goal.cancel_goal_async()
                self.current_goal = None

    def cat_cb(self, msg: PointStamped):
        self.cat_cx   = msg.point.x
        self.cat_dist = msg.point.z if msg.point.z > 0 else None

        import time
        now = time.time()

        # Rate limit flee decisions
        if now - self.last_flee < FLEE_COOLDOWN:
            return
        if self.is_fleeing:
            return

        self.last_flee = now
        self.get_logger().info(
            f'[Flee] Cat detected at cx={self.cat_cx:.2f} '
            f'dist={self.cat_dist:.2f}m' if self.cat_dist else
            f'[Flee] Cat detected at cx={self.cat_cx:.2f}')

        # Run AI planning in separate thread (non-blocking)
        threading.Thread(
            target=self._plan_and_flee,
            daemon=True
        ).start()

    # ── AI Planning ───────────────────────────────────────────────────────────

    def _plan_and_flee(self):
        """
        Call Claude AI to choose best hiding spot.
        Runs in background thread.
        """
        self.is_fleeing = True
        self.status_pub.publish(String(data='planning'))

        try:
            spot = self._ask_claude_for_escape()
            if spot:
                self.get_logger().info(
                    f'[AI] Fleeing to {spot["name"]} '
                    f'({spot["x"]:.2f}, {spot["y"]:.2f})')
                self.status_pub.publish(
                    String(data=f'fleeing to {spot["name"]}'))
                self._navigate_to(spot['x'], spot['y'])
            else:
                self.get_logger().warn('[AI] No escape plan — using fallback')
                self._fallback_flee()
        except Exception as e:
            self.get_logger().error(f'[AI] Planning failed: {e}')
            self._fallback_flee()

    def _ask_claude_for_escape(self):
        """
        Ask Claude API to pick best hiding spot.
        Returns dict with name, x, y or None on failure.
        """
        # Build context for Claude
        cat_angle_deg = None
        if self.cat_cx is not None:
            # Convert normalised cx to angle (-30° to +30° approximately)
            cat_angle_deg = (self.cat_cx - 0.5) * HIDING_SPOTS[0]['x'] * 60

        spots_desc = '\n'.join([
            f'- {s["name"]}: x={s["x"]:.2f}, y={s["y"]:.2f}, '
            f'distance from robot={math.hypot(s["x"]-self.robot_x, s["y"]-self.robot_y):.2f}m'
            for s in HIDING_SPOTS
        ])

        prompt = f"""You are controlling an autonomous robot that is playing a cat-and-mouse game with a cat. 
The robot must flee from the cat and hide.

Current situation:
- Robot position: x={self.robot_x:.2f}, y={self.robot_y:.2f}
- Cat detected at camera position: cx={self.cat_cx:.2f} (0=far left, 0.5=center, 1=far right)
- Cat distance: {f'{self.cat_dist:.2f}m' if self.cat_dist else 'unknown'}
- Camera faces the BACK of the robot

Available hiding spots:
{spots_desc}

Choose the single best hiding spot that:
1. Is FAR from the cat's current position
2. Is in the OPPOSITE direction from where the cat is
3. Is not the spot the robot is currently near (robot at {self.robot_x:.2f}, {self.robot_y:.2f})

Respond with ONLY a JSON object, no other text:
{{"name": "spot_name", "x": 0.0, "y": 0.0, "reason": "brief reason"}}"""

        payload = json.dumps({
            'model':      CLAUDE_MODEL,
            'max_tokens': 200,
            'messages': [
                {'role': 'user', 'content': prompt}
            ]
        }).encode('utf-8')

        req = urllib.request.Request(
            CLAUDE_API_URL,
            data=payload,
            headers={
                'Content-Type':      'application/json',
                'anthropic-version': '2023-06-01',
            },
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data     = json.loads(resp.read().decode('utf-8'))
            raw_text = data['content'][0]['text'].strip()

            self.get_logger().info(f'[AI] Response: {raw_text}')

            # Parse JSON response
            result = json.loads(raw_text)

            # Validate the chosen spot exists
            for spot in HIDING_SPOTS:
                if spot['name'] == result['name']:
                    self.get_logger().info(
                        f'[AI] Chose {result["name"]}: {result.get("reason", "")}')
                    return spot

            # If name not found, use coordinates directly
            return {
                'name': result.get('name', 'ai_choice'),
                'x':    float(result['x']),
                'y':    float(result['y'])
            }

    def _fallback_flee(self):
        """
        Fallback: pick spot farthest from cat without AI.
        Used when API call fails.
        """
        if not HIDING_SPOTS:
            self.is_fleeing = False
            return

        # Estimate cat position in map frame
        # Cat is behind robot (camera faces back)
        cat_map_x = self.robot_x - (self.cat_dist or 1.0)
        cat_map_y = self.robot_y

        # Pick spot farthest from estimated cat position
        best_spot = max(
            HIDING_SPOTS,
            key=lambda s: math.hypot(
                s['x'] - cat_map_x,
                s['y'] - cat_map_y
            )
        )

        self.get_logger().info(
            f'[Fallback] Fleeing to {best_spot["name"]}')
        self._navigate_to(best_spot['x'], best_spot['y'])

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate_to(self, x, y):
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('[Flee] Nav2 not ready!')
            self.is_fleeing = False
            return

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = x
        goal.pose.pose.position.y    = y
        goal.pose.pose.orientation.w = 1.0

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_cb)

    def _goal_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('[Flee] Goal rejected!')
            self.is_fleeing = False
            return

        self.current_goal = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        status = future.result().status
        if status == 4:
            self.get_logger().info('[Flee] ✅ Reached hiding spot!')
            self.status_pub.publish(String(data='hidden'))
        else:
            self.get_logger().warn(f'[Flee] Navigation failed status={status}')
            self.status_pub.publish(String(data='flee_failed'))
        self.is_fleeing   = False
        self.current_goal = None


def main(args=None):
    rclpy.init(args=args)
    node = FleeBehavior()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
