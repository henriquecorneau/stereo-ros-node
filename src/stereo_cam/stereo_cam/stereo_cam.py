import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

left_camera_matrix = np.array([
    [2036.5, 1.3, 583.4],
    [0, 2028.6, 449.5],
    [0, 0, 1]
])

right_camera_matrix = np.array([
    [2036.5, 1.3, 583.4],
    [0, 2028.6, 449.5],
    [0, 0, 1]
])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.3863, 0.5091, -0.0011, 0.001, -2.6353]])
right_distortion = np.array([[-0.4256, 1.5578, -0.0023, 0.0005, -11.6757]])

# 旋转矩阵
R = np.array([
    [0.9999, 0.0012, 0.0164],
    [-0.0011, 1, -0.0019],
    [-0.0164, 0.0019, 0.9999]
])

# 平移矩阵
T = np.array([[-85.9802], [0.0723], [1.1451]])

size = (2560 // 2, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                    right_camera_matrix, right_distortion, size, R,
                                                                    T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

class StereoCameraNode(Node):
    def __init__(self):
        super().__init__('stereo_camera_node')
        
        self.pub_left = self.create_publisher(Image, 'camera/left/image_raw', 10)
        self.pub_right = self.create_publisher(Image, 'camera/right/image_raw', 10)
        
        self.pub_left_rect = self.create_publisher(Image, 'camera/left/image_rect', 10)
        self.pub_right_rect = self.create_publisher(Image, 'camera/right/image_rect', 10)
        
        self.timer = self.create_timer(1.0/30, self.timer_callback)
        self.bridge = CvBridge()
        
        self.declare_parameter('camera_index', 2)
        camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            self.get_logger().error("Could not open video device")
            exit()
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        
        if ret:
            h, w, _ = frame.shape
            half = w // 2
            left_image = frame[:, :half]
            right_image = frame[:, half:]
            
            img1_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
            img2_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
            
            imageL_rect = img1_rectified
            imageR_rect = img2_rectified
            
            time_stamp = self.get_clock().now().to_msg()
            
            try:
                msg_left = self.bridge.cv2_to_imgmsg(left_image, encoding="bgr8")
                msg_left.header.stamp = time_stamp
                msg_left.header.frame_id = "camera_left_frame"
                self.pub_left.publish(msg_left)
                
                msg_left_rect = self.bridge.cv2_to_imgmsg(imageL_rect, encoding="bgr8")
                msg_left_rect.header.stamp = time_stamp
                msg_left_rect.header.frame_id = "camera_left_rect_frame"
                self.pub_left_rect.publish(msg_left_rect)
            except Exception as e:
                self.get_logger().error(f"Failed to convert left image: {e}")
            
            try:
                msg_right = self.bridge.cv2_to_imgmsg(right_image, encoding="bgr8")
                msg_right.header.stamp = time_stamp
                msg_right.header.frame_id = "camera_right_frame"
                self.pub_right.publish(msg_right)
                
                msg_right_rect = self.bridge.cv2_to_imgmsg(imageR_rect, encoding="bgr8")
                msg_right_rect.header.stamp = time_stamp
                msg_right_rect.header.frame_id = "camera_right_rect_frame"
                self.pub_right_rect.publish(msg_right_rect)
            except Exception as e:
                self.get_logger().error(f"Failed to convert right image: {e}")
            
            
        else:
            self.get_logger().warning("Failed to capture image from camera")
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
            
def main(args=None):
    rclpy.init(args=args)
    node = StereoCameraNode()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()