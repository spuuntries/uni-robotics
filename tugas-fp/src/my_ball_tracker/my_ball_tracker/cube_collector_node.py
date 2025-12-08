#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
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
        self.get_logger().info("Cube Collector Node Started!")

        # --- Parameters ---
        self.declare_parameter(
            "target_color", "red"
        )  # not fully used yet, hardcoded to red
        self.declare_parameter("pickup_distance", 0.5)

        # --- Subscribers ---
        # Gazebo ModelStates often requires Best Effort
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, depth=10
        )

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
            qos_best_effort,
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
        self.state = "SEARCH"  # 4 states defined, SEARCH, APPROACH, RETURN, DROP
        self.current_pose = None  # (x, y, theta)
        self.target_cube_name = None  # Name of the cube we are targeting
        self.has_cube = False
        self.cubes_collected = 0
        self.cube_locations = {}  # {name: (x, y)} from model states
        self.home_pose = (0.0, 0.0)

        # Sensor Data
        self.scan_ranges = []

        # Control vars
        self.last_img_time = time.time()
        self.angular_error = 0.0
        self.linear_error = 0.0

        # Create a timer for the main control loop (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges

    def odom_callback(self, msg):
        # Log verbosely for first few times to debug connection
        if not hasattr(self, "odom_count"):
            self.odom_count = 0
        self.odom_count += 1

        if self.odom_count <= 10:
            self.get_logger().info(f"Odom received! ({self.odom_count})")

        # Log throttled heartbeat thereafter
        now = time.time()
        if not hasattr(self, "last_odom_log") or (now - self.last_odom_log > 5.0):
            self.get_logger().info("Odom received (heartbeat)")
            self.last_odom_log = now

        # Extract x, y, theta
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        # quaternion to euler (yaw)
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = (pos.x, pos.y, yaw)

    def model_states_callback(self, msg):
        # Update locations of all 'cube_*' entities and the robot
        if not hasattr(self, "logged_entities"):
            self.get_logger().info(f"Model States Entities: {msg.name}")
            self.logged_entities = True

        for i, name in enumerate(msg.name):
            if name.startswith("cube_"):
                pos = msg.pose[i].position
                self.cube_locations[name] = (pos.x, pos.y)
            elif name == "waffle_pi":
                # Ground truth robot pose
                pos = msg.pose[i].position
                self.ground_truth_pose = (pos.x, pos.y)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_img_time = time.time()
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Detect Red Cubes
        # Red can wrap around 180 in HSV, so we need two ranges
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Lower Red
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        # Upper Red
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        # Publish Debug Mask
        if self.debug_mask_pub.get_subscription_count() > 0:
            self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 100:  # Min area threshold
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw for debug
                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), 2)

                    # Calculate errors for control
                    h, w, d = cv_image.shape
                    self.angular_error = (w / 2) - cx
                    # Simple distance estimation (inverse of area or y-position)
                    # For "APPROACH", we just want to get close enough.
                    # As we get closer, cy (y-position of center) moves down (higher value) or area gets bigger.
                    self.linear_error = (
                        1.0  # Constant forward speed, stop when "close" logic triggers
                    )
            else:
                self.angular_error = None
        else:
            self.angular_error = None

        # Publish Debug Image
        if self.debug_img_pub.get_subscription_count() > 0:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def get_closest_cube(self):
        # Use ground truth if available, else fallback to Odom
        ref_pose = getattr(self, "ground_truth_pose", None)
        source = "GT"
        if not ref_pose:
            source = "Odom"
            if self.current_pose:
                ref_pose = (self.current_pose[0], self.current_pose[1])
            else:
                return None, 9999.9

        if not self.cube_locations:
            return None, 9999.9

        min_dist = 9999.9
        closest_name = None

        rx, ry = ref_pose

        for name, (cx, cy) in self.cube_locations.items():
            dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        # Debug spam?
        # self.get_logger().info(f"Closest: {closest_name}, Dist: {min_dist:.2f} ({source})")
        return closest_name, min_dist

    def delete_cube(self, name):
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_client.call_async(req)

    def respawn_collected_cube(self):
        # Spawn a visual marker at home to show we collected it
        # Just stacking them or placing them nearby
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
        # Offset them slightly so they don't overlap perfectly
        req.initial_pose.position.x = 0.5 + (self.cubes_collected * 0.3)
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 0.1
        self.spawn_client.call_async(req)

    def draw_lidar_debug(
        self, left_sector, right_sector, target_turn, avoid_turn, dist_to_target
    ):
        # Create a blank image (300x300)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        center_x, center_y = 150, 150
        scale = 50.0  # pixels per meter

        # Draw Robot
        cv2.circle(img, (center_x, center_y), 10, (255, 0, 0), -1)

        # Draw Scan Points
        # Right sector: -45 to 0 (indices -45 to end)
        for i, r in enumerate(right_sector):
            if math.isinf(r) or r == 0:
                continue
            angle = math.radians((i - 45))  # approx -45 to 0
            px = int(
                center_x + r * scale * math.cos(angle - 1.57)
            )  # -1.57 to rotate to face up
            py = int(center_y + r * scale * math.sin(angle - 1.57))
            cv2.circle(img, (px, py), 2, (0, 255, 255), -1)

        # Left sector: 0 to 45 (indices 0 to 45)
        for i, r in enumerate(left_sector):
            if math.isinf(r) or r == 0:
                continue
            angle = math.radians(i)
            px = int(center_x + r * scale * math.cos(angle - 1.57))
            py = int(center_y + r * scale * math.sin(angle - 1.57))
            cv2.circle(img, (px, py), 2, (0, 255, 255), -1)

        # Draw Target Distance Circle
        if dist_to_target < 5.0:
            r_px = int(dist_to_target * scale)
            cv2.circle(img, (center_x, center_y), r_px, (0, 255, 0), 1)

        # Draw Vectors
        # Target Turn (Green)
        t_len = 50
        tx = int(center_x - target_turn * 100)  # Inverted X for image coords
        cv2.line(img, (center_x, center_y), (tx, center_y - t_len), (0, 255, 0), 2)

        # Avoid Turn (Red)
        ax = int(center_x - avoid_turn * 100)
        cv2.line(img, (center_x, center_y), (ax, center_y - t_len), (0, 0, 255), 2)

        # Final Result (White)
        final = target_turn + avoid_turn
        fx = int(center_x - final * 100)
        cv2.line(img, (center_x, center_y), (fx, center_y - t_len), (255, 255, 255), 3)

        # Text
        cv2.putText(
            img,
            f"Tgt: {dist_to_target:.2f}m",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            img,
            f"Avd: {avoid_turn:.2f}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

        return img

    def control_loop(self):
        if self.current_pose is None:
            now = time.time()
            if not hasattr(self, "last_wait_log") or (now - self.last_wait_log > 2.0):
                self.get_logger().info("Waiting for Odom data...")
                self.last_wait_log = now
            return

        twist = Twist()

        if self.state == "SEARCH":
            # If we see a target, switch to approach
            if self.angular_error is not None:
                self.get_logger().info("Target Found! Switching to APPROACH")
                self.state = "APPROACH"
            else:
                # Wander Logic: Drive forward, turn if obstacle
                min_dist = 10.0
                if len(self.scan_ranges) > 0:
                    # check front 60 degrees (approx)
                    # Ranges array size varies, usually 360 or so
                    # We assume 0 is front
                    mid = len(self.scan_ranges) // 2
                    left_range = self.scan_ranges[0:20]
                    right_range = self.scan_ranges[-20:]
                    front_ranges = left_range + right_range

                    # filter infinite/nan
                    valid_ranges = [
                        r
                        for r in front_ranges
                        if not math.isinf(r) and not math.isnan(r) and r > 0.0
                    ]
                    if valid_ranges:
                        min_dist = min(valid_ranges)

                if min_dist < 0.5:
                    # Obstacle ahead, turn
                    twist.angular.z = 0.5
                    twist.linear.x = 0.0
                else:
                    # Clear, drive forward
                    twist.linear.x = 0.2
                    # Add slight wobble to cover more area
                    twist.angular.z = math.sin(time.time()) * 0.1

        elif self.state == "APPROACH":
            # Visual Servoing
            if self.angular_error is None:
                self.state = "SEARCH"  # Lost it
            else:
                # 1. Attraction (Visual Servoing)
                target_turn = 0.005 * self.angular_error
                twist.linear.x = 0.1

                # 2. Repulsion (Lidar Avoidance)
                avoid_turn = 0.0
                target_name, dist_to_target = self.get_closest_cube()

                left_sector = []
                right_sector = []
                min_front = 10.0

                if len(self.scan_ranges) > 0:
                    left_sector = self.scan_ranges[0:45]
                    right_sector = self.scan_ranges[-45:]

                    min_left = min(
                        [r for r in left_sector if not math.isinf(r) and r > 0.0],
                        default=10.0,
                    )
                    min_right = min(
                        [r for r in right_sector if not math.isinf(r) and r > 0.0],
                        default=10.0,
                    )
                    min_front = min(min_left, min_right)

                # Debug Logging (Throttled)
                if not hasattr(self, "last_debug_time"):
                    self.last_debug_time = 0
                if time.time() - self.last_debug_time > 1.0:
                    pose_source = (
                        "GT" if getattr(self, "ground_truth_pose", None) else "Odom"
                    )
                    self.get_logger().info(
                        f"Approach: Tgt={dist_to_target:.2f}m ({pose_source}), Lidar={min_front:.2f}m"
                    )
                    self.last_debug_time = time.time()

                if len(self.scan_ranges) > 0:
                    # Avoidance Logic:
                    # Only avoid if obstacle is significantly closer than the target.
                    # CRITICAL FIX: If dist_to_target is unknown (9999), DISABLE avoidance to prevent dodging the target itself.
                    if (
                        dist_to_target < 100.0
                        and min_front < 1.0
                        and min_front < (dist_to_target - 0.3)
                    ):
                        if min_left < min_right:
                            avoid_turn = -0.4 * (1.0 - min_left)
                        else:
                            avoid_turn = 0.4 * (1.0 - min_right)
                        twist.linear.x = 0.05

                # 3. Blend
                twist.angular.z = target_turn + avoid_turn

                # 4. Debug Visualization
                if self.debug_lidar_pub.get_subscription_count() > 0:
                    debug_lidar_img = self.draw_lidar_debug(
                        left_sector,
                        right_sector,
                        target_turn,
                        avoid_turn,
                        dist_to_target,
                    )
                    self.debug_lidar_pub.publish(
                        self.bridge.cv2_to_imgmsg(debug_lidar_img, "bgr8")
                    )

                # 5. Grab Logic
                # Trigger if ground truth says we are close OR if lidar says we hit something while facing target
                is_touching = min_front < 0.40 and abs(target_turn) < 0.2

                should_grab = False
                if target_name and dist_to_target < 0.6:
                    should_grab = True
                elif is_touching:
                    should_grab = True

                if should_grab:
                    self.get_logger().info(
                        f"Grabbing! (GT Dist: {dist_to_target:.2f}, Lidar: {min_front:.2f})"
                    )
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                    if target_name:
                        self.delete_cube(target_name)
                        if target_name in self.cube_locations:
                            del self.cube_locations[target_name]
                    else:
                        # Blind Grasp: Try to delete all potential cubes
                        self.get_logger().info("Blind Grasping all cubes...")
                        for i in range(10):
                            self.delete_cube(f"cube_{i}")

                    self.has_cube = True
                    self.state = "RETURN"

        elif self.state == "RETURN":
            # Go to (0,0)
            rx, ry, rtheta = self.current_pose
            dist_to_home = math.sqrt(rx**2 + ry**2)

            if not hasattr(self, "last_return_log"):
                self.last_return_log = 0
            if time.time() - self.last_return_log > 2.0:
                self.get_logger().info(f"Returning... Dist: {dist_to_home:.2f}m")
                self.last_return_log = time.time()

            # Angle to home
            angle_to_home = math.atan2(
                -ry, -rx
            )  # Vector from robot to (0,0) is (-rx, -ry)

            # Angular difference
            angle_diff = angle_to_home - rtheta
            # Normalize angle
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            if dist_to_home < 0.3:
                self.state = "DROP"
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                # Simple P controller for navigation
                twist.angular.z = 1.5 * angle_diff
                if abs(angle_diff) < 0.5:  # Face mostly towards home before driving
                    twist.linear.x = 0.2

        elif self.state == "DROP":
            self.get_logger().info("Dropping Cube")
            self.respawn_collected_cube()
            self.cubes_collected += 1
            self.has_cube = False

            # Back up a bit?
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
