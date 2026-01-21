import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable

def generate_launch_description():
    world_path = '/root/BFMC_workspace/files/simulation/world.sdf'
    models_path = '/root/BFMC_workspace/files/simulation/models_pkg'

    plugin_path = '/root/BFMC_workspace/install/traffic_light_plugin/lib/traffic_light_plugin'

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
        )
    ])
