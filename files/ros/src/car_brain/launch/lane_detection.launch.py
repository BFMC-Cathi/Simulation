from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    default_params = f"{get_package_share_directory('car_brain')}/config/lane_params.yaml"
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Absolute path to lane_params.yaml",
    )

    lane_node = Node(
        package="car_brain",
        executable="lane_node",
        name="lane_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
    )

    return LaunchDescription([
        params_file_arg,
        lane_node,
    ])
