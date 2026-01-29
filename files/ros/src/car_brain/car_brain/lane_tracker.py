#!/usr/bin/env python3
import rclpy
import cv2
from car_brain.base_node import CameraNode


class LaneTracker(CameraNode):
    def __init__(self):
        super().__init__('lane_tracker')
    
    def process(self, frame):
        """Lane tracking işlemleri"""
        # TODO: Lane tracking işlemleri buraya eklenecek
        pass
    
    def show_frame(self, frame, window_name='Lane Tracker'):
        """İşlenmiş frame'i ekranda gösterir"""
        super().show_frame(frame, window_name)


def main(args=None):
    rclpy.init(args=args)
    node = LaneTracker()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
