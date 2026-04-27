from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

# Handles: T1 LiDAR, T2 TF base_link→laser, T3 TF base_link→base_footprint,
#          T4 rf2o odometry, T5 Nav2+AMCL+map
# NOT included: T6 motors (needs sudo busybox devmem first — run manually)

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    map_file = os.path.join(
        get_package_share_directory('catrun'), 'map', 'my_map.yaml')

    return LaunchDescription([

        # T2 — TF base_link → laser
        # LiDAR is mounted 19cm above robot center, flipped 180° (3.14159 roll)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_laser',
            arguments=['0', '0', '0.19', '3.14159', '0', '0', 'base_link', 'laser'],
        ),

        # T3 — TF base_link → base_footprint
        # Nav2 needs base_footprint for costmap collision checking
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_base_footprint',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint'],
        ),

        # T1 — RPLiDAR C1
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar',
            parameters=[{
                'serial_port':     '/dev/ttyUSB0',
                'serial_baudrate': 460800,
                'frame_id':        'laser',
                'angle_compensate': True,
                'scan_mode':       'Standard',
            }],
            output='screen',
        ),

        # T4 — rf2o laser odometry (replaces wheel encoders)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('rf2o_laser_odometry'),
                    'launch', 'rf2o_laser_odometry.launch.py')
            ),
        ),

        # T5 — Nav2 + AMCL + map server
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map':          map_file,
                'use_sim_time': 'false',
            }.items(),
        ),

    ])