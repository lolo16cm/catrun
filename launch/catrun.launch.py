"""
catrun.launch.py - Definitive bringup for catrun robot.

Now supports TWO modes via the `mode` launch argument:

  ros2 launch catrun catrun.launch.py             # default: watch
  ros2 launch catrun catrun.launch.py mode:=watch
  ros2 launch catrun catrun.launch.py mode:=play

  watch : front CSI camera + cat_detector(watch) + seek_cat
          (single camera. robot wanders, finds cat, follows like ball-follower)
  play  : front CSI + rear USB cameras + cat_detector(play) + flee_behavior
          (dual camera. robot ambushes, sees cat, smart-direction flee.
           cat_detector picks which camera feed to use based on state.)

Order of operations (unchanged):
  t=0s   TF (base_link -> laser, base_link -> base_footprint)
  t=0s   RPLiDAR starts publishing /scan
  t=3s   rf2o laser odometry (after /scan is alive)
  t=8s   Nav2 localization (map_server + amcl)        [watch only]
  t=10s  Initial pose published continuously for 60s  [watch only]
  t=22s  Nav2 navigation                              [watch only]
  t=2s   camera_node + cat_detector + behavior node   [both modes]

EDIT INITIAL_POSE_X / Y / YAW_DEG below to match where your robot starts.

EDIT focal_px values (FRONT_FOCAL_PX, REAR_FOCAL_PX) per the bbox-width
calibration described in cat_detector.py. They will likely differ
between the front CSI cam and the rear USB cam.
"""

from launch import LaunchDescription
from launch.actions import (IncludeLaunchDescription, TimerAction,
                            ExecuteProcess, DeclareLaunchArgument,
                            GroupAction, OpaqueFunction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import math


# ─── Calibration: edit these for your robot ─────────────────────────────
INITIAL_POSE_X       = 0.0     # meters in the map frame
INITIAL_POSE_Y       = 0.0
INITIAL_POSE_YAW_DEG = 90.0    # 0=+X, 90=+Y, 180=-X, -90=-Y

# Per-camera focal_px for distance estimation. Calibrate by placing the
# cat plushie at exactly 1 m, reading the bbox= value from the annotated
# stream, and computing focal_px = bbox_width_at_1m / 0.25.
#
# NOTE: When you change camera resolution, focal_px scales linearly.
# At 1280x720 the calibration was 600. At 640x360 (half), it's ~300.
# Recalibrate by placing the plushie at 1m and reading the bbox width.
FRONT_FOCAL_PX = 300.0   # CSI camera at 640x360 - recalibrate for your robot
REAR_FOCAL_PX  = 500.0   # USB camera at 640x480 - placeholder, please calibrate
# ────────────────────────────────────────────────────────────────────────


def _build_actions(context, *args, **kwargs):
    """Build the launch actions, choosing nodes based on `mode` arg."""

    mode = LaunchConfiguration('mode').perform(context).strip().lower()
    if mode not in ('watch', 'play'):
        print(f"[catrun.launch] WARNING: unknown mode '{mode}', "
              f"defaulting to 'watch'")
        mode = 'watch'

    nav2_bringup_dir = get_package_share_directory("nav2_bringup")
    map_file = os.path.join(
        get_package_share_directory("catrun"), "map", "my_map.yaml")

    # Initial pose YAML (only used in watch mode)
    yaw_rad = math.radians(INITIAL_POSE_YAW_DEG)
    qz = math.sin(yaw_rad / 2.0)
    qw = math.cos(yaw_rad / 2.0)
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

    # ─── Common nodes (both modes) ────────────────────────────────────
    common_actions = [
        # TF tree
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

        # RPLiDAR
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

        # rf2o laser odometry (t=3s)
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
    ]

    # ─── Mode-specific: camera + detector + behavior ──────────────────
    if mode == 'watch':
        # WATCH MODE: single front CSI camera publishing to legacy
        # /camera/catrun topic. cat_detector also subscribes to that.
        camera_params = {
            'camera_source': 'csi0',
            'flip_method':   2,
            'width':         640,
            'height':        360,
            'framerate':     15,
            'publish_topic': '/camera/catrun',
        }
        detector_params = {
            'mode':     'watch',
            'focal_px': FRONT_FOCAL_PX,
        }
        behavior_node = Node(
            package='catrun',
            executable='seek_cat',
            name='seek_cat',
            output='screen',
        )
        # Single camera node for watch mode
        camera_nodes = [
            Node(
                package='catrun',
                executable='camera_node',
                name='camera_node',
                parameters=[camera_params],
                output='screen',
            ),
        ]
    else:  # play
        # PLAY MODE: DUAL camera.
        #   /camera/front  - USB at /dev/video1 (physically pointing FORWARD)
        #   /camera/rear   - CSI sensor 0       (physically pointing BACKWARD)
        # NOTE: The cameras were physically swapped on the robot, so we
        # swap the device->topic mapping here to match. The TOPIC names
        # (front/rear) reflect physical direction, not which sensor.
        # cat_detector subscribes to BOTH and processes whichever
        # matches the current robot state (from /seek_status).
        # It also mirrors the active feed to /camera/catrun so the
        # web UI keeps working without changes.
        detector_params = {
            'mode':     'play',
            'focal_px': FRONT_FOCAL_PX,
        }
        behavior_node = Node(
            package='catrun',
            executable='flee_behavior',
            name='flee_behavior',
            output='screen',
        )
        camera_nodes = [
            # FRONT (forward-facing) = USB camera at /dev/video1
            Node(
                package='catrun',
                executable='camera_node',
                name='camera_front',
                parameters=[{
                    'camera_source': 'usb1',
                    'device_path':   '/dev/video1',
                    'width':         640,
                    'height':        480,
                    'framerate':     15,
                    'publish_topic': '/camera/front',
                }],
                output='screen',
            ),
            # REAR (backward-facing) = CSI IMX477 sensor 0
            Node(
                package='catrun',
                executable='camera_node',
                name='camera_rear',
                parameters=[{
                    'camera_source': 'csi0',
                    'flip_method':   2,
                    'width':         640,
                    'height':        360,
                    'framerate':     15,
                    'publish_topic': '/camera/rear',
                }],
                output='screen',
            ),
        ]

    # Camera(s) + detector + behavior come up at t=2s in both modes.
    behavior_actions = [
        TimerAction(
            period=2.0,
            actions=camera_nodes + [
                Node(
                    package='catrun',
                    executable='cat_detector',
                    name='cat_detector',
                    parameters=[detector_params],
                    output='screen',
                ),
                behavior_node,
            ]
        ),
    ]

    # ─── Watch-mode-only: full Nav2 stack ─────────────────────────────
    nav2_actions = []
    if mode == 'watch':
        nav2_actions = [
            # Nav2 localization (t=8s)
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

            # Initial pose publisher (t=10s, runs for 60s)
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

            # Nav2 navigation stack (t=22s)
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
        ]

    return common_actions + behavior_actions + nav2_actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'mode',
            default_value='watch',
            description="Robot behavior mode: 'watch' (seek/follow cat with "
                        "front CSI camera) or 'play' (flee from cat using "
                        "rear USB camera at /dev/video1)."),
        OpaqueFunction(function=_build_actions),
    ])