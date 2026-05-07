from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    map_file = os.path.join(
        get_package_share_directory("catrun"), "map", "my_map.yaml")

    return LaunchDescription([

        # TF base_link -> laser
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_laser",
            arguments=["0", "0", "0.19", "3.14159", "0", "0",
                       "base_link", "laser"],
        ),

        # TF base_link -> base_footprint
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_base_footprint",
            arguments=["0", "0", "0", "0", "0", "0",
                       "base_link", "base_footprint"],
        ),

        # RPLiDAR C1
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

        # rf2o laser odometry
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("rf2o_laser_odometry"),
                    "launch", "rf2o_laser_odometry.launch.py")
            ),
        ),

        # Nav2 + AMCL + map (delayed 5s)
        TimerAction(
            period=5.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(nav2_bringup_dir, "launch",
                                     "bringup_launch.py")
                    ),
                    launch_arguments={
                        "map":          map_file,
                        "use_sim_time": "false",
                    }.items(),
                ),
            ]
        ),

        # Set initial pose (delayed 15s)
        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package="catrun",
                    executable="set_pose",
                    name="set_initial_pose",
                    output="screen",
                ),
            ]
        ),

    ])
