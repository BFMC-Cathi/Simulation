#!/usr/bin/env python3
import rclpy
import cv2
from ultralytics import YOLO
from car_brain.base_node import CameraNode


class TrafficSign(CameraNode):
    def __init__(self, model_path='/home/halil/Simulation/model.pt'):
        super().__init__('traffic_sign')
        
        self.model = YOLO(model_path)
        self.get_logger().info(f'YOLOv8 modeli yüklendi: {model_path}')
    
    def process(self, frame):
        """YOLOv8 inference yap"""
        results = self.model(frame, verbose=False)
        annotated_frame = results[0].plot()
        self.show_frame(annotated_frame)
    
    def show_frame(self, frame, window_name='Traffic Sign Detection'):
        """İşlenmiş frame'i ekranda gösterir"""
        super().show_frame(frame, window_name)


def main(args=None):
    rclpy.init(args=args)
    
    model_path = '/home/halil/Simulation/model.pt'
    node = TrafficSign(model_path)
    
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
