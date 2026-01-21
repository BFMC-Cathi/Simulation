import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ros_dev/BFMC_workspace/files/ros/install/car_brain'
