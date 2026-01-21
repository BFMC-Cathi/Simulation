#!/usr/bin/env python3
import rclpy
from rclpy.node import Node #importing node class

class MyNode(Node): #inherited the node we imported

    def __init__(self): #constructor
        super().__init__("first_node")
        self.get_logger().info("ROS")
        self.counter_ = 0
        self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        self.get_logger().info("Hello" + str(self.counter_))
        self.counter_ += 1

def main(args=None):
    rclpy.init(args = args) #starting communication
    #everything for this function will be written here
    node = MyNode() #created the node
    rclpy.spin(node) #this makes the node to continue


    rclpy.shutdown() # closing communication

if __name__ == '__main__':
    main()