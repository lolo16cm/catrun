# flee_behavior.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class FleeBehavior(Node):
    def __init__(self):
        super().__init__('flee_behavior')
        
        # Subscribe to cat position
        self.sub = self.create_subscription(
            PointStamped,
            '/cat_position',
            self.cat_callback,
            10
        )
        
        # Nav2 action client
        self.nav_client = ActionClient(
            self, NavigateToPose, 'navigate_to_pose'
        )
        
    def cat_callback(self, msg):
        self.get_logger().info('Cat detected! Fleeing...')
        self.send_flee_goal()
    
    def send_flee_goal(self):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        
        # Define hiding spots on your map
        # Change these to real coordinates from your map
        hiding_spots = [
            (1.0, 1.0),   # spot 1
            (-1.0, 2.0),  # spot 2
            (2.0, -1.0),  # spot 3
        ]
        
        import random
        spot = random.choice(hiding_spots)
        goal.pose.pose.position.x = spot[0]
        goal.pose.pose.position.y = spot[1]
        goal.pose.pose.orientation.w = 1.0
        
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

def main(args=None):
    rclpy.init(args=args)
    node = FleeBehavior()
    rclpy.spin(node)
    rclpy.shutdown()
