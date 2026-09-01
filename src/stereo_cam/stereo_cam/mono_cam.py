import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class MonoCameraNode(Node):
    def __init__(self):
        super().__init__('mono_camera_node')
        
        self.pub_cam = self.create_publisher(Image, 'camera/image_raw', 10)
        
        self.timer = self.create_timer(1.0/30, self.timer_callback)
        self.bridge = CvBridge()
        
        self.declare_parameter('camera_index', 2)
        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            self.get_logger().error("Could not open video device")
            exit()
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            time_stamp = self.get_clock().now().to_msg()
            
            try:
                msg_cam = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg_cam.header.stamp = time_stamp
                msg_cam.header.frame_id = "camera_frame"
                self.pub_cam.publish(msg_cam)
            except Exception as e:
                self.get_logger().error(f"Failed to convert image camera: {e}")
        else:
            self.get_logger().warning("Failed to capture image from camera")
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
            
def main(args=None):
    rclpy.init(args=args)
    node = MonoCameraNode()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()