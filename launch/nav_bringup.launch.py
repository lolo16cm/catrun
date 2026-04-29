from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    rf2o_dir        = get_package_share_directory('rf2o_laser_odometry')
    map_file        = '/home/cc/cat_ws/src/catrun/map/my_map.yaml'

    return LaunchDescription([

        # Terminal 2 — TF base_link → laser
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_link_laser',
            arguments=['0', '0', '0.19', '3.14159', '0', '0',
                       'base_link', 'laser'],
        ),

        # Terminal 3 — TF base_link → base_footprint
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_link_footprint',
            arguments=['0', '0', '0', '0', '0', '0',
                       'base_link', 'base_footprint'],
        ),

        # Terminal 4 — rf2o odometry
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rf2o_dir, 'launch',
                             'rf2o_laser_odometry.launch.py')
            ),
        ),

        # Terminal 5 — Nav2 + AMCL + map (delayed 3s to let TF settle)
        TimerAction(
            period=3.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(nav2_bringup_dir, 'launch',
                                     'bringup_launch.py')
                    ),
                    launch_arguments={
                        'map':          map_file,
                        'use_sim_time': 'false',
                    }.items(),
                ),
            ]
        ),

    ])