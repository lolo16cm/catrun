from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    map_file = os.path.join(
        get_package_share_directory('catrun'), 'map', 'my_map.yaml')

    return LaunchDescription([

	# Robot state publisher (handles all TF from URDF)
	Node(
	    package='robot_state_publisher',
	    executable='robot_state_publisher',
	    parameters=[{
	        'robot_description': open(
	            os.path.join(
	                get_package_share_directory('catrun'),
	                'urdf', 'robot.urdf'
	            )
	        ).read()
	    }]
	),	

        # RPLiDAR
        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            parameters=[{
                'serial_port': '/dev/ttyUSB0',
                'serial_baudrate': 460800,
                'frame_id': 'laser',
                'angle_compensate': True,
                'scan_mode': 'Standard',
            }],
        ),

        # rf2o laser odometry
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('rf2o_laser_odometry'),
                    'launch', 'rf2o_laser_odometry.launch.py')
            ),
        ),

        # Nav2
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_dir, 'launch', 'bringup_launch.py')
            ),
            launch_arguments={
                'map': map_file,
                'use_sim_time': 'false',
		'params_file': os.path.join(
        get_package_share_directory('catrun'), 'map', 'nav2_params.yaml'),
            }.items(),
        ),

        # Motor control (delayed 5s)
        TimerAction(
            period=30.0,
            actions=[
                Node(
                    package='catrun',
                    executable='motor_control',
                    output='screen',
                ),
            ]
        ),

        # Navigation (delayed 15s to wait for Nav2)
        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package='catrun',
                    executable='navigation',
                    output='screen',
                ),
            ]
        ),
    ])
