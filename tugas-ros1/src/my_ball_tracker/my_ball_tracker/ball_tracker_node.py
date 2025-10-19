#!/usr/bi/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image       # image msg
from geometry_msgs.msg import Twist     # /cmd_vel msg
from cv_bridge import CvBridge          # to convert ros_image -> opencv
import cv2
import numpy as np

class BallTrackerNode(Node):
    def __init__(self):
        super().__init__('ball_tracker_node')
        self.get_logger().info('ball_tracker_node started! owo')
        
        # --- create the publisher and subscriber ---
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10)
        
        # --- create the debug publishers ---
        self.debug_img_pub = self.create_publisher(Image, '/ball_tracker/debug_image', 10) 
        self.debug_mask_pub = self.create_publisher(Image, '/ball_tracker/debug_mask', 10)
        
        # create the cv_bridge
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # --- 1. convert ros_image -> opencv ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        # --- 2. the "real-time processing" (find green) ---
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([35, 50, 50])  
        upper_green = np.array([90, 255, 255]) 
        
        mask = cv2.inRange(hsv_image, lower_green, upper_green)
        
        M = cv2.moments(mask)
        twist_msg = Twist()

        if M['m00'] > 0:
            # a green object was found!
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            
            # draw a circle on the original image
            cv2.circle(cv_image, (center_x, center_y), 20, (0, 255, 0), 3) 
            
            (h, w, d) = cv_image.shape
            error = center_x - (w / 2)
            
            twist_msg.angular.z = -float(error) * 0.01
            twist_msg.linear.x = 0.1
            
        else:
            # no green object found, spin
            twist_msg.angular.z = 0.3
            twist_msg.linear.x = 0.0
            
        # --- 4. publish the command ---
        self.cmd_vel_pub.publish(twist_msg)
        
        # --- 5. publish the debug images ---
        try:
            # publish the original image with the circle
            debug_img_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8") 
            self.debug_img_pub.publish(debug_img_msg)
            
            # publish the black/white mask
            debug_mask_msg = self.bridge.cv2_to_imgmsg(mask, "mono8") 
            self.debug_mask_pub.publish(debug_mask_msg) 
            
        except Exception as e:
            self.get_logger().error(f'cv_bridge debug pub error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BallTrackerNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
