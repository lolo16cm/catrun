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

        # ─── t=0s: TF tree ────────────────────────────────────────────────
        # Static transforms — must be up before anything else subscribes
        # to TF, so Nav2's costmaps can resolve frames immediately.

        # TF base_link -> laser
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_laser",
            arguments=["0", "0", "0.19", "3.14159", "0", "0",
                       "base_link", "laser"],
            output="screen",
        ),

        # TF base_link -> base_footprint
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="base_link_to_base_footprint",
            arguments=["0", "0", "0", "0", "0", "0",
                       "base_link", "base_footprint"],
            output="screen",
        ),

        # ─── t=0s: RPLiDAR ────────────────────────────────────────────────
        # Starts immediately so /scan is publishing before rf2o subscribes.
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

        # ─── t=3s: rf2o laser odometry ────────────────────────────────────
        # Delayed so /scan is alive first; otherwise rf2o spams
        # "Waiting for laser_scans..." until LiDAR comes up.
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

        # ─── t=8s: Localization (map_server + amcl) ───────────────────────
        # Started FIRST so the `map` frame is being published BEFORE
        # the navigation stack's costmaps try to look it up.
        # This is the fix for controller_server hanging in inactive [2].
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

        # ─── t=14s: Set initial pose ──────────────────────────────────────
        # Done AFTER localization is up so AMCL has somewhere to put the
        # pose. Done BEFORE navigation starts so by the time the costmap
        # configures, AMCL is publishing map->odom and the `map` frame
        # is resolvable in TF.
        TimerAction(
            period=14.0,
            actions=[
                Node(
                    package="catrun",
                    executable="set_pose",
                    name="set_initial_pose",
                    output="screen",
                ),
            ]
        ),

        # ─── t=18s: Navigation (planner, controller, bt_navigator, etc.) ─
        # By now: TF tree complete, /scan flowing, /odom flowing,
        # map_server active, amcl active, initial pose set, `map` frame
        # being published in TF. Costmaps configure cleanly.
        TimerAction(
            period=18.0,
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
