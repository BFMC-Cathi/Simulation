import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable

def generate_launch_description():
    world_path = '/root/BFMC_workspace/files/simulation/world.sdf'
    models_path = '/root/BFMC_workspace/files/simulation/models_pkg'

    # FIXED: This is the missing link!
    # Points to where colcon built your traffic light code.
    plugin_path = '/root/BFMC_workspace/install/traffic_light_plugin/lib/traffic_light_plugin'

    return LaunchDescription([
        # 1. Tell Gazebo where the Models are (Cars, Signs)
        SetEnvironmentVariable(name='IGN_GAZEBO_RESOURCE_PATH', value=models_path),

        # 2. Tell Gazebo where the Plugins are (Traffic Light Logic)
        SetEnvironmentVariable(name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH', value=plugin_path),

        # 3. Start the Simulator
        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', '-v', '4', world_path], # Added -v 4 for detailed logs
            output='screen'
        ),

        # 4. Start the ROS Bridge
        ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                 '/automobile/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                 '/automobile/command@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                 '/automobile/IMU@sensor_msgs/msg/Imu[ignition.msgs.IMU',
                 '/automobile/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'],
            output='screen'
        )
    ])
