#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
import random


class CubeCollectorNode(Node):
    def __init__(self):
        super().__init__("cube_collector_node")
        self.get_logger().info("Cube Collector Node Started! (Spawn + Smart Search)")

        # --- Parameters ---
        self.declare_parameter("target_color", "red")
        self.declare_parameter("pickup_distance", 0.45) 

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.model_states_sub = self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.model_states_callback,
            QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            ),
        )

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_img_pub = self.create_publisher(
            Image, "/cube_collector/debug_image", 10
        )
        self.debug_mask_pub = self.create_publisher(
            Image, "/cube_collector/debug_mask", 10
        )
        self.debug_lidar_pub = self.create_publisher(
            Image, "/cube_collector/debug_lidar", 10
        )

        # --- Service Clients ---
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")

        # --- State Variables ---
        self.bridge = CvBridge()
        self.state = "SEARCH"
        self.current_pose = None
        self.has_cube = False
        self.cubes_collected = 0
        self.home_pose = (0.0, 0.0)

        self.latest_model_states = None
        self.scan_ranges = []

        # --- Tracking & Memory Variables ---
        self.angular_error = 0.0  # None if target not seen
        self.last_sighting_time = 0.0 
        self.last_known_error = 0.0   

        # Create a timer for the main control loop (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # --- RESTORED: Spawn Initial Cubes Timer ---
        self.spawn_timer = self.create_timer(2.0, self.spawn_initial_cubes)

    def model_states_callback(self, msg):
        self.latest_model_states = msg

    # --- RESTORED: Spawn Function ---
    def spawn_initial_cubes(self):
        self.spawn_timer.cancel() # Run once
        n = 5
        
        # Wait for service
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Spawn service unavailable. Skipping spawn.")
            return

        sdf = """
        <?xml version='1.0'?>
        <sdf version="1.4">
            <model name="my_cube">
                <static>0</static>
                <link name="link">
                    <inertial>
                        <mass>0.1</mass>
                        <inertia>
                            <ixx>0.0001</ixx><ixy>0</ixy><ixz>0</ixz>
                            <iyy>0.0001</iyy><iyz>0</iyz><izz>0.0001</izz>
                        </inertia>
                    </inertial>
                    <collision name="collision">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                    </collision>
                    <visual name="visual">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                        <material>
                            <script>
                                <uri>file://media/materials/scripts/gazebo.material</uri>
                                <name>Gazebo/Red</name>
                            </script>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>
        """

        for i in range(n):
            req = SpawnEntity.Request()
            req.name = f"cube_{i}"
            req.xml = sdf

            # Spawn in a donut shape to avoid spawning on top of robot
            while True:
                rx = random.uniform(-1.5, 1.5)
                ry = random.uniform(-1.5, 1.5)
                if (rx**2 + ry**2) > 0.6**2: # Radius > 0.6m
                    break

            req.initial_pose.position.x = rx
            req.initial_pose.position.y = ry
            req.initial_pose.position.z = 0.5
            self.spawn_client.call_async(req)

        self.get_logger().info(f"Spawned {n} cubes.")

    def get_closest_cube_name(self):
        if not self.latest_model_states:
            return None

        # 1. Find Ground Truth Robot Pose
        robot_pose = None
        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        for i, name in enumerate(names):
            if name == "waffle_pi": 
                robot_pose = (poses[i].position.x, poses[i].position.y)
                break

        if robot_pose is None:
            return None

        rx, ry = robot_pose
        min_dist = 999.9
        best_name = None

        # 2. Find Closest Cube to GT Robot Pose
        for i, name in enumerate(names):
            if name.startswith("cube_"):
                cx = poses[i].position.x
                cy = poses[i].position.y

                dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

        if min_dist < 1.0:
            return best_name
        return None

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.current_pose = (pos.x, pos.y, yaw)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red ranges
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        if self.debug_mask_pub.get_subscription_count() > 0:
            self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), 2)
                    h, w, d = cv_image.shape
                    
                    self.angular_error = (w / 2) - cx
                    self.last_known_error = self.angular_error
                    self.last_sighting_time = time.time()
            else:
                self.angular_error = None
        else:
            self.angular_error = None

        if self.debug_img_pub.get_subscription_count() > 0:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def delete_cube(self, name):
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_client.call_async(req)

    def respawn_collected_cube(self):
        sdf = """
        <?xml version='1.0'?>
        <sdf version="1.4">
            <model name="collected_cube">
                <static>1</static>
                <link name="link">
                    <visual name="visual">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                        <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Green</name></script></material>
                    </visual>
                </link>
            </model>
        </sdf>
        """
        req = SpawnEntity.Request()
        req.name = f"collected_{self.cubes_collected}"
        req.xml = sdf
        req.initial_pose.position.x = 0.5 + (self.cubes_collected * 0.3)
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 0.1
        self.spawn_client.call_async(req)

    def get_lidar_sector(self, sector="front"):
        if len(self.scan_ranges) == 0:
            return 99.9

        ranges = self.scan_ranges
        if sector == "front":
            sector_ranges = ranges[0:20] + ranges[-20:]
        elif sector == "left":
            sector_ranges = ranges[60:120]
        elif sector == "right":
            sector_ranges = ranges[240:300]
        else:
            return 99.9

        valid_ranges = [r for r in sector_ranges if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        if not valid_ranges:
            return 99.9 
        return min(valid_ranges)

    def control_loop(self):
        if self.current_pose is None:
            return 

        twist = Twist()
        
        dist_front = self.get_lidar_sector("front")
        dist_left = self.get_lidar_sector("left")
        dist_right = self.get_lidar_sector("right")

        target_visible = self.angular_error is not None
        time_since_seen = time.time() - self.last_sighting_time

        if self.state == "SEARCH":
            # 1. Panic Stop
            if dist_front < 0.20:
                self.get_logger().warn(f"Panic Stop! Dist: {dist_front:.2f}m")
                twist.linear.x = -0.15
                twist.angular.z = 0.0
            
            # 2. Target Locked (Visual)
            elif target_visible:
                self.get_logger().info("Target Locked! Approaching...")
                self.state = "APPROACH"
                twist.linear.x = 0.0
                twist.angular.z = 0.0

            # 3. Target Lost Nearby (Reverse Recovery)
            elif dist_front < 0.50 and time_since_seen < 2.0:
                 self.get_logger().info("Target lost nearby! Reversing...")
                 twist.linear.x = -0.15
                 twist.angular.z = 0.0
                
            # 4. Obstacle Avoidance (No Visual)
            elif dist_front < 0.60:
                self.get_logger().warn(f"Obstacle ({dist_front:.2f}m). Avoiding.")
                twist.linear.x = 0.0
                if dist_left > dist_right:
                    twist.angular.z = 0.6 
                else:
                    twist.angular.z = -0.6
            
            # 5. Memory Recovery (Scan last position)
            elif time_since_seen < 3.0:
                self.get_logger().info("Scanning for lost target...")
                twist.linear.x = 0.0
                direction = 1.0 if self.last_known_error > 0 else -1.0
                twist.angular.z = direction * 0.6 

            # 6. Wander
            else:
                twist.linear.x = 0.2
                twist.angular.z = 0.1 

        elif self.state == "APPROACH":
            if not target_visible:
                self.get_logger().warn("Visual Lost during Approach!")
                self.state = "SEARCH"
                twist.linear.x = 0.0
            else:
                k_p = 0.005
                twist.angular.z = k_p * self.angular_error
                twist.linear.x = 0.15

                if dist_front < 0.40:
                    self.get_logger().info(f"Grabbing Range ({dist_front:.2f}m). EXECUTE.")
                    self.state = "GRAB"
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

        elif self.state == "GRAB":
            target_name = self.get_closest_cube_name()
            if target_name:
                self.get_logger().info(f"Grabbing {target_name}...")
                self.delete_cube(target_name)
                self.has_cube = True
                self.state = "RETURN"
            else:
                self.get_logger().info("Grab Failed. Retry Search.")
                twist.linear.x = -0.15
                self.state = "SEARCH"

        elif self.state == "RETURN":
            rx, ry, rtheta = self.current_pose
            dist_to_home = math.sqrt(rx**2 + ry**2)
            
            if dist_front < 0.3:
                 self.get_logger().warn("Obstacle on Return path!")
                 twist.linear.x = 0.0 
                 twist.angular.z = 0.5 
            else:
                angle_to_home = math.atan2(-ry, -rx)
                angle_diff = angle_to_home - rtheta

                while angle_diff > math.pi: angle_diff -= 2 * math.pi
                while angle_diff < -math.pi: angle_diff += 2 * math.pi

                if dist_to_home < 0.3:
                    self.state = "DROP"
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    twist.angular.z = 1.5 * angle_diff
                    if abs(angle_diff) < 0.5:
                        twist.linear.x = 0.2

        elif self.state == "DROP":
            self.get_logger().info("Dropping Cube")
            self.respawn_collected_cube()
            self.cubes_collected += 1
            self.has_cube = False
            twist.linear.x = -0.2
            self.cmd_vel_pub.publish(twist)
            time.sleep(1.0)
            self.state = "SEARCH"

        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = CubeCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()