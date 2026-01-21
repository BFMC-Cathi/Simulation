#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

class SimpleDriver(Node):
    def __init__(self):
        super().__init__("simple_driver")

        self.subscription = self.create_subscription(
            Image, "/camera/image", self.image_callback, qos_profile_sensor_data)
        
        self.publisher_ = self.create_publisher(Twist, "/cmd_vel", 10)
        self.get_logger().info("Driver Node started. Waiting for camera data...")

    def image_callback(self, msg):
        self.get_logger().info(f"Lmao Camera active! Received frame: {msg.width}x{msg.height}")

        move_cmd = Twist()
        move_cmd.linear.x = 200.0
        move_cmd.angular.z = 100.0

        self.publisher_.publish(move_cmd)

def main(args=None):
    rclpy.init(args=args)
    driver_node = SimpleDriver()
    rclpy.spin(driver_node)

    rclpy.shutdown()