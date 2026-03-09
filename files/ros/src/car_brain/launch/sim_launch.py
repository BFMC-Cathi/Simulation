import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch_ros.actions import Node as RosNode

def generate_launch_description():
    world_path = '/home/ros_dev/BFMC_workspace/files/simulation/world.sdf'
    models_path = '/home/ros_dev/BFMC_workspace/files/simulation/models_pkg'
    track_graph_path = '/home/ros_dev/BFMC_workspace/trackgraph.graphml'

    plugin_path = '/home/ros_dev/BFMC_workspace/install/traffic_light_plugin/lib/traffic_light_plugin'

    # Environment that ign gazebo (both server + GUI) needs to find models
    ign_env = {
        'IGN_GAZEBO_RESOURCE_PATH': models_path,
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': plugin_path,
        'SDF_PATH': models_path,
        'IGN_FILE_PATH': models_path,
    }

    return LaunchDescription([
        SetEnvironmentVariable(name='IGN_GAZEBO_RESOURCE_PATH', value=models_path),
        SetEnvironmentVariable(name='IGN_GAZEBO_SYSTEM_PLUGIN_PATH', value=plugin_path),
        SetEnvironmentVariable(name='SDF_PATH', value=models_path),
        SetEnvironmentVariable(name='IGN_FILE_PATH', value=models_path),

        ExecuteProcess(
            cmd=['ign', 'gazebo', '-r', '-v', '4', world_path],
            output='screen',
            additional_env=ign_env,
            # Use '-s' flag for headless: ['ign', 'gazebo', '-r', '-s', '-v', '4', world_path]
        ),

        ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                 '/automobile/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
                 '/automobile/command@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                 '/automobile/IMU@sensor_msgs/msg/Imu[ignition.msgs.IMU',
                 '/model/automobile/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'],
            output='screen'
        ),

        # ── Perception Node — camera → YOLO + lane detection → publish ──
        TimerAction(
            period=5.0,
            actions=[
                RosNode(
                    package='car_brain',
                    executable='perception_node',
                    name='perception_node',
                    output='screen',
                    parameters=[{
                        # YOLO
                        'confidence_threshold': 0.45,
                        'publish_visualisation': True,
                        'control_rate_hz': 30.0,
                        # Lane detection
                        'white_threshold': 160,
                        'sliding_window_count': 12,
                        'sliding_window_margin': 100,
                        'sliding_window_min_pix': 30,
                        'lane_history_frames': 5,
                        'missing_lane_timeout_sec': 2.0,
                    }],
                ),
            ],
        ),

        # ── Control + State Node — FSM + PID → /automobile/command ──
        TimerAction(
            period=6.0,
            actions=[
                RosNode(
                    package='car_brain',
                    executable='control_state_node',
                    name='control_state_node',
                    output='screen',
                    parameters=[{
                        # Track graph
                        'track_graph_path': track_graph_path,
                        'nav_source_node': '86',
                        'nav_target_node': '274',
                        # Speed
                        'cruise_speed': 0.35,
                        'slow_speed': 0.2,
                        'highway_speed': 0.6,
                        'intersection_speed': 0.25,
                        # PID steering
                        'steering_kp': 0.8,
                        'steering_ki': 0.02,
                        'steering_kd': 0.3,
                        'max_steering': 1.0,
                        'steering_alpha': 0.25,
                        # Ramping
                        'max_accel': 0.8,
                        'max_decel': 2.0,
                        'control_rate_hz': 30.0,
                        # FSM
                        'stop_hold_sec': 3.0,
                        'intersection_stop_sec': 1.5,
                        'intersection_turn_sec': 2.5,
                        'intersection_turn_steer': -0.8,
                        'debounce_frames': 3,
                        'frame_timeout_sec': 2.0,
                        'image_height': 480,
                        'image_width': 640,
                    }],
                ),
            ],
        ),
    ])
