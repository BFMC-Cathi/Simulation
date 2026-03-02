import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node as RosNode

def generate_launch_description():
    world_path = '/home/ros_dev/BFMC_workspace/files/simulation/world.sdf'
    models_path = '/home/ros_dev/BFMC_workspace/files/simulation/models_pkg'

    plugin_path = '/home/ros_dev/BFMC_workspace/install/traffic_light_plugin/lib/traffic_light_plugin'

    return LaunchDescription([
        SetEnvironmentVariable(name='IGN_GAZEBO_RESOURCE_PATH', value=models_path),
        SetEnvironmentVariable(name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH', value=plugin_path),

        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', '-v', '4', world_path],
            output='screen'
        ),

        ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                 '/automobile/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                 '/automobile/command@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                 '/automobile/IMU@sensor_msgs/msg/Imu[ignition.msgs.IMU',
                 '/automobile/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'],
            output='screen'
        ),

        # Launch the driving node after a short delay to let Gazebo start
        TimerAction(
            period=5.0,
            actions=[
                RosNode(
                    package='car_brain',
                    executable='yolov8_driving_node',
                    name='yolov8_driving_node',
                    output='screen',
                    parameters=[{
                        'cruise_speed': 0.2,
                        'confidence_threshold': 0.45,
                        'control_rate_hz': 20.0,
                        'steering_kp': 0.005,
                        'steering_ki': 0.0001,
                        'steering_kd': 0.002,
                        'publish_visualisation': True,
                        'stop_hold_sec': 3.0,
                        'debounce_frames': 3,
                    }],
                ),
            ],
        ),
    ])
