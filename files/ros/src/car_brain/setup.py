import os
from glob import glob
from setuptools import setup

package_name = 'car_brain'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    package_data={package_name: ['*.pt']},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='todo@todo.com',
    description='Car Brain Package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_node = car_brain.perception_node:main',
            'control_state_node = car_brain.control_state_node:main',
            'lane_node = car_brain.lane_node:main',
        ],
    },
)
