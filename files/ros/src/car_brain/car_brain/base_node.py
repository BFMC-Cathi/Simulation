#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np


class CameraNode(Node):
    """Kamera verisi alan node'lar için base class"""
    
    def __init__(self, node_name, topic='/automobile/camera/image_raw'):
        super().__init__(node_name)
        
        self.frame = None
        self.subscription = self.create_subscription(
            Image,
            topic,
            self._image_callback,
            10
        )
        self.get_logger().info(f'{node_name} başlatıldı - {topic}')
    
    def _image_callback(self, msg):
        """ROS Image mesajını OpenCV formatına çevirir"""
        if msg.encoding == 'rgb8':
            self.frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            self.frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        else:
            self.frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        
        # Alt class'ın process fonksiyonunu çağır
        self.process(self.frame)
    
    def process(self, frame):
        """Alt class'lar bu fonksiyonu override edecek"""
        pass
    
    def show_frame(self, frame, window_name='Frame'):
        """Görüntüyü ekranda gösterir"""
        if frame is not None:
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
