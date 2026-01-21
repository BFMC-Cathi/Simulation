import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/BFMC_workspace/files/ros/install/my_robot_controller'
