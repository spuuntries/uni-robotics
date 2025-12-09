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


class CubeCollectorNode(Node):
    def __init__(self):
        super().__init__("cube_collector_node")
        self.get_logger().info("Cube Collector Node Started! (Lidar + Vision Only)")

        # --- Parameters ---
        self.declare_parameter("target_color", "red")
        self.declare_parameter("pickup_distance", 0.45)  # Distance to trigger grab

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
        self.state = "SEARCH"  # SEARCH, APPROACH, RETURN, DROP
        self.current_pose = None  # (x, y, theta)
        self.has_cube = False
        self.cubes_collected = 0
        self.home_pose = (0.0, 0.0)

        # Real-time state (for "Virtual Gripper" logic only)
        self.latest_model_states = None

        # Sensor Data
        self.scan_ranges = []

        # Control vars
        self.last_img_time = time.time()
        self.angular_error = 0.0  # None if target not seen

        # Create a timer for the main control loop (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Spawn Cubes asynchronously after startup (prevents blocking __init__)
        self.spawn_timer = self.create_timer(2.0, self.spawn_initial_cubes)

    def model_states_callback(self, msg):
        self.latest_model_states = msg

    def spawn_initial_cubes(self):
        # Cancel the timer so this only runs once
        self.spawn_timer.cancel()
        n = 5

        import random

        # Wait for service (with timeout so we don't hang forever)
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                "Spawn service not available after waiting. Skipping spawn."
            )
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
            name = f"cube_{i}"
            req.name = name
            req.xml = sdf

            # Spawn in a donut shape
            while True:
                rx = random.uniform(-1.5, 1.5)
                ry = random.uniform(-1.5, 1.5)
                if (rx**2 + ry**2) > 0.5**2:
                    break

            req.initial_pose.position.x = rx
            req.initial_pose.position.y = ry
            req.initial_pose.position.z = 0.5

            # Fire and forget - we do NOT store the location
            self.spawn_client.call_async(req)

        self.get_logger().info(f"Spawned {n} cubes blindly.")

    def get_closest_cube_name(self):
        # "Virtual Gripper" logic
        # Returns the name of a cube ONLY if it is physically close to the robot
        if not self.latest_model_states:
            return None

        # 1. Find Ground Truth Robot Pose
        robot_pose = None
        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        for i, name in enumerate(names):
            if name == "waffle_pi":  # Default Turtlebot3 name in Gazebo
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

        # Threshold for "touching" (0.6m covers the robot radius + cube radius)
        # Using GT vs GT, this should be very accurate.
        if min_dist < 0.5:
            return best_name
        return None

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges

    def odom_callback(self, msg):
        # Extract x, y, theta
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        # quaternion to euler (yaw)
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = (pos.x, pos.y, yaw)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_img_time = time.time()
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Detect Red Cubes
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
                    # Error: Center of image vs Center of blob
                    self.angular_error = (w / 2) - cx
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

    def get_obstacle_avoidance(self, dist_limit=0.6):
        # Returns (angular_correction, min_front_dist)
        if not self.scan_ranges:
            return 0.0, 99.9

        n = len(self.scan_ranges)

        # Helper to get min distance in a degree range
        def get_min_in_range(start_deg, end_deg):
            vals = []
            for deg in range(start_deg, end_deg + 1):
                idx = deg % 360
                if idx < n:
                    r = self.scan_ranges[idx]
                    if not math.isinf(r) and not math.isnan(r) and r > 0.05:
                        vals.append(r)
            return min(vals) if vals else 99.9

        front = get_min_in_range(-20, 20)
        left = get_min_in_range(20, 60)
        right = get_min_in_range(300, 340)  # -60 to -20

        turn = 0.0

        # If obstacle is detected within limit
        if front < dist_limit:
            # Blocked in front, turn away from closest side
            if left < right:
                turn = -0.8  # Turn Right strongly
            else:
                turn = 0.8  # Turn Left strongly
        elif left < dist_limit * 0.7:
            turn = -0.4  # Nudge Right
        elif right < dist_limit * 0.7:
            turn = 0.4  # Nudge Left

        return turn, front

    def control_loop(self):
        if self.current_pose is None:
            return  # Wait for odom

        twist = Twist()
        # Default avoidance check
        avoid_turn, front_dist = self.get_obstacle_avoidance(dist_limit=0.6)

        if self.state == "SEARCH":
            if self.angular_error is not None:
                self.get_logger().info("Target Sight! Switching to APPROACH")
                self.state = "APPROACH"
            else:
                # Wander
                if abs(avoid_turn) > 0.1:
                    # Obstacle! Turn
                    twist.angular.z = avoid_turn
                    twist.linear.x = 0.0
                else:
                    twist.linear.x = 0.2
                    twist.angular.z = 0.1  # Slight curve

        elif self.state == "APPROACH":
            if self.angular_error is None:
                self.state = "SEARCH"  # Lost visual
                twist.linear.x = 0.0
            else:
                # Obstacle Avoidance blending
                # Use slightly tighter limit for approach to allow getting close to objects
                avoid_turn, front_dist = self.get_obstacle_avoidance(dist_limit=0.5)

                # Visual Servoing
                # P-Controller for rotation
                k_p = 0.005
                vis_turn = k_p * self.angular_error

                # Combine: Priority to avoidance if critical
                if abs(avoid_turn) > 0.1:
                    twist.angular.z = avoid_turn + (
                        vis_turn * 0.5
                    )  # Dampen visual tracking if avoiding
                    twist.linear.x = 0.05  # Slow down
                else:
                    twist.angular.z = vis_turn
                    twist.linear.x = 0.15

                # Check if we are close enough to grab
                # Condition: We see red (angular_error is not None) AND Lidar says close
                if front_dist < 0.40:
                    self.get_logger().info(
                        f"Close enough! (Lidar: {front_dist:.2f}m). GRABBING."
                    )
                    self.state = "GRAB"
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

        elif self.state == "GRAB":
            # self.get_logger().info(self.latest_model_states)
            # Targeted Grasp using Virtual Gripper logic
            target_name = self.get_closest_cube_name()

            if target_name:
                self.get_logger().info(f"Grabbing {target_name}...")
                self.delete_cube(target_name)

                self.has_cube = True
                self.state = "RETURN"
            else:
                self.get_logger().info("Grabbing failed - nothing in range.")
                # Back up slightly to try again
                twist.linear.x = -0.1
                self.state = "SEARCH"

        elif self.state == "RETURN":
            rx, ry, rtheta = self.current_pose
            dist_to_home = math.sqrt(rx**2 + ry**2)

            angle_to_home = math.atan2(-ry, -rx)
            angle_diff = angle_to_home - rtheta

            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Wiggle room logic:
            # If close to home, reduce avoidance sensitivity
            avoid_limit = 0.6
            if dist_to_home < 1.0:
                avoid_limit = (
                    0.35  # Allow getting closer to pillars near tracking goal/home
                )

            avoid_turn, front_dist = self.get_obstacle_avoidance(dist_limit=avoid_limit)

            if dist_to_home < 0.3:
                self.state = "DROP"
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                nav_turn = 1.5 * angle_diff

                if abs(avoid_turn) > 0.1:
                    # Blend obstacle avoidance with homing
                    twist.angular.z = avoid_turn + (nav_turn * 0.3)
                    twist.linear.x = 0.05
                else:
                    twist.angular.z = nav_turn
                    if abs(angle_diff) < 0.5:
                        twist.linear.x = 0.2

        elif self.state == "DROP":
            self.get_logger().info("Dropping Cube")
            self.respawn_collected_cube()
            self.cubes_collected += 1
            self.has_cube = False

            # Back up a bit to avoid hitting the dropped cube immediately
            twist.linear.x = -0.2
            self.cmd_vel_pub.publish(twist)
            time.sleep(1.0)  # Blocking sleep is okay-ish here for simple logic

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
