"""
catrun.launch.py - Definitive bringup for catrun robot.

Order of operations:
  t=0s   TF (base_link -> laser, base_link -> base_footprint)
  t=0s   RPLiDAR starts publishing /scan
  t=3s   rf2o laser odometry (after /scan is alive)
  t=8s   Nav2 localization (map_server + amcl)
  t=10s  Initial pose published continuously to /initialpose for 60 seconds
         (bypasses set_pose entirely; uses ros2 topic pub directly)
  t=22s  Nav2 navigation (planner, controller, bt_navigator, etc.)
         By now AMCL has been getting initial pose for 12s and is
         publishing map->odom in TF, so costmaps configure cleanly.

EDIT INITIAL_POSE_X / Y / YAW_DEG below to match where your robot starts.
"""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import math


# EDIT THESE TO MATCH YOUR ROBOT'S STARTING POSE
INITIAL_POSE_X       = 0.0    # meters in the map frame
INITIAL_POSE_Y       = 0.0
INITIAL_POSE_YAW_DEG = 90.0   # degrees: 0 = +X, 90 = +Y, 180 = -X, -90 = -Y


def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    map_file = os.path.join(
        get_package_share_directory("catrun"), "map", "my_map.yaml")

    # Convert yaw to quaternion z, w (rotation about Z axis)
    yaw_rad = math.radians(INITIAL_POSE_YAW_DEG)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)

    # Build the YAML message string for ros2 topic pub
    initial_pose_yaml = (
        "{"
        "header: {frame_id: 'map'}, "
        "pose: {"
            "pose: {"
                f"position: {{x: {INITIAL_POSE_X}, y: {INITIAL_POSE_Y}, z: 0.0}}, "
                f"orientation: {{x: 0.0, y: 0.0, z: {qz}, w: {qw}}}"
            "}, "
            "covariance: ["
                "0.25, 0.0, 0.0, 0.0, 0.0, 0.0, "
                "0.0, 0.25, 0.0, 0.0, 0.0, 0.0, "
                "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                "0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "
                "0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942"
            "]"
        "}"
        "}"
    )

    return LaunchDescription([

        # t=0s: TF tree
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_laser",
            arguments=["0", "0", "0.19", "3.14159", "0", "0",
                       "base_link", "laser"],
            output="screen",
        ),

        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_base_footprint",
            arguments=["0", "0", "0", "0", "0", "0",
                       "base_link", "base_footprint"],
            output="screen",
        ),

        # t=0s: RPLiDAR
        Node(
            package="rplidar_ros",
            executable="rplidar_composition",
            name="rplidar",
            parameters=[{
                "serial_port":      "/dev/ttyUSB0",
                "serial_baudrate":  460800,
                "frame_id":         "laser",
                "angle_compensate": True,
                "scan_mode":        "Standard",
            }],
            output="screen",
        ),

        # t=3s: rf2o laser odometry
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(
                            get_package_share_directory("rf2o_laser_odometry"),
                            "launch", "rf2o_laser_odometry.launch.py")
                    ),
                ),
            ]
        ),

        # t=8s: Nav2 localization (map_server + amcl)
        TimerAction(
            period=8.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(nav2_bringup_dir, "launch",
                                     "localization_launch.py")
                    ),
                    launch_arguments={
                        "map":          map_file,
                        "use_sim_time": "false",
                    }.items(),
                ),
            ]
        ),

        # t=10s: Publish initial pose continuously for 60 seconds
        # Bypasses the buggy set_pose.py entirely.
        # ros2 topic pub keeps the publisher alive long enough for DDS
        # discovery to complete (~1-2s), then keeps publishing to
        # guarantee AMCL receives at least one message.
        TimerAction(
            period=10.0,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2", "topic", "pub",
                        "--rate", "1",
                        "--times", "60",
                        "/initialpose",
                        "geometry_msgs/msg/PoseWithCovarianceStamped",
                        initial_pose_yaml,
                    ],
                    output="screen",
                    name="initial_pose_publisher",
                ),
            ]
        ),

        # t=22s: Nav2 navigation stack
        # By now AMCL has been receiving initial pose for 12 seconds and
        # is publishing map->odom in TF. Costmaps configure cleanly.
        TimerAction(
            period=22.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(nav2_bringup_dir, "launch",
                                     "navigation_launch.py")
                    ),
                    launch_arguments={
                        "use_sim_time": "false",
                    }.items(),
                ),
            ]
        ),
    ])
