import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/BFMC_workspace/files/ros/src/car_brain/install/car_brain'
