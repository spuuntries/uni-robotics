#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from collections import deque
from enum import Enum, auto


class RobotState(Enum):
    """State machine states for better code clarity"""
    SEARCH = auto()
    APPROACH = auto()
    GRAB = auto()
    RETURN = auto()
    DROP = auto()
    AVOID_OBSTACLE = auto()


class LidarZones:
    """Structured lidar zone data"""
    def __init__(self):
        self.front = float('inf')
        self.front_left = float('inf')
        self.front_right = float('inf')
        self.left = float('inf')
        self.right = float('inf')
        self.back = float('inf')
        # Narrow front for precise detection
        self.front_narrow = float('inf')


class CubeCollectorNode(Node):
    def __init__(self):
        super().__init__("cube_collector_node")
        self.get_logger().info("Cube Collector Node Started! (Optimized Version)")

        # --- Parameters ---
        self.declare_parameter("target_color", "red")
        self.declare_parameter("pickup_distance", 0.45)
        
        # Obstacle avoidance parameters
        self.SAFE_DISTANCE = 0.5          # Minimum safe distance dari obstacle
        self.CRITICAL_DISTANCE = 0.3      # Distance untuk emergency stop
        self.APPROACH_SAFE_DISTANCE = 0.35  # Safe distance saat approach
        self.WALL_FOLLOW_DISTANCE = 0.6   # Target distance untuk wall following
        
        # Speed parameters
        self.MAX_LINEAR_SPEED = 0.22      # TurtleBot3 max speed
        self.MAX_ANGULAR_SPEED = 1.5
        self.SEARCH_LINEAR_SPEED = 0.18
        self.APPROACH_LINEAR_SPEED = 0.12
        self.RETURN_LINEAR_SPEED = 0.2

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

        # --- Service Clients ---
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")

        # --- State Variables ---
        self.bridge = CvBridge()
        self.state = RobotState.SEARCH
        self.previous_state = None  # For returning after obstacle avoidance
        self.current_pose = None
        self.has_cube = False
        self.cubes_collected = 0
        self.home_pose = (0.0, 0.0)

        # Model states for virtual gripper
        self.latest_model_states = None

        # Lidar data
        self.lidar_zones = LidarZones()
        self.raw_scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0

        # Vision data
        self.angular_error = None
        self.target_area = 0  # Area of detected cube
        self.last_img_time = time.time()
        
        # Search optimization
        self.search_direction = 1  # 1 for left, -1 for right
        self.search_start_time = time.time()
        self.last_cube_direction = 0  # Remember where we last saw cube
        self.stuck_counter = 0
        self.last_position = None
        self.position_history = deque(maxlen=50)  # Track recent positions
        
        # Obstacle avoidance state
        self.avoid_direction = 0  # -1 left, 1 right
        self.avoid_start_time = 0
        self.consecutive_obstacles = 0

        # Create control timer (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Spawn cubes after startup
        self.spawn_timer = self.create_timer(2.0, self.spawn_initial_cubes)

    # ==================== CALLBACKS ====================
    
    def model_states_callback(self, msg):
        self.latest_model_states = msg

    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation

        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = (pos.x, pos.y, yaw)
        
        # Track position for stuck detection
        self.position_history.append((pos.x, pos.y, time.time()))

    def scan_callback(self, msg):
        """Process lidar scan into useful zones"""
        self.raw_scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        
        # Process into zones
        self._process_lidar_zones()

    def _process_lidar_zones(self):
        """
        Divide lidar into zones for better obstacle detection.
        Assumes 360 degree scan with index 0 at front.
        """
        if len(self.raw_scan_ranges) == 0:
            return
            
        n = len(self.raw_scan_ranges)
        
        def get_min_in_range(start_idx, end_idx):
            """Get minimum valid reading in index range (handles wraparound)"""
            indices = []
            if start_idx < end_idx:
                indices = range(start_idx, end_idx)
            else:  # Wraparound case
                indices = list(range(start_idx, n)) + list(range(0, end_idx))
            
            valid = []
            for i in indices:
                r = self.raw_scan_ranges[i]
                if not math.isinf(r) and not math.isnan(r) and 0.05 < r < 10.0:
                    valid.append(r)
            
            return min(valid) if valid else float('inf')
        
        # Calculate zone boundaries (in indices)
        # Assuming 360 points for 360 degrees
        deg_to_idx = n / 360.0
        
        # Front narrow: -15 to +15 degrees
        self.lidar_zones.front_narrow = get_min_in_range(
            int(n - 15 * deg_to_idx), int(15 * deg_to_idx)
        )
        
        # Front wide: -30 to +30 degrees  
        self.lidar_zones.front = get_min_in_range(
            int(n - 30 * deg_to_idx), int(30 * deg_to_idx)
        )
        
        # Front-left: 30 to 60 degrees
        self.lidar_zones.front_left = get_min_in_range(
            int(30 * deg_to_idx), int(60 * deg_to_idx)
        )
        
        # Front-right: -60 to -30 degrees (300 to 330)
        self.lidar_zones.front_right = get_min_in_range(
            int(300 * deg_to_idx), int(330 * deg_to_idx)
        )
        
        # Left: 60 to 120 degrees
        self.lidar_zones.left = get_min_in_range(
            int(60 * deg_to_idx), int(120 * deg_to_idx)
        )
        
        # Right: -120 to -60 degrees (240 to 300)
        self.lidar_zones.right = get_min_in_range(
            int(240 * deg_to_idx), int(300 * deg_to_idx)
        )
        
        # Back: 150 to 210 degrees
        self.lidar_zones.back = get_min_in_range(
            int(150 * deg_to_idx), int(210 * deg_to_idx)
        )

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_img_time = time.time()
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # Detect Red Cubes with optimized HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red color ranges (improved thresholds)
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations untuk noise reduction
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        if self.debug_mask_pub.get_subscription_count() > 0:
            self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 150:  # Minimum area threshold
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 2)

                    h, w, _ = cv_image.shape
                    self.angular_error = (w / 2) - cx
                    self.target_area = area
                    
                    # Remember direction for search optimization
                    self.last_cube_direction = 1 if self.angular_error > 0 else -1
                else:
                    self.angular_error = None
                    self.target_area = 0
            else:
                self.angular_error = None
                self.target_area = 0
        else:
            self.angular_error = None
            self.target_area = 0

        # Draw debug info
        cv2.putText(cv_image, f"State: {self.state.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(cv_image, f"Front: {self.lidar_zones.front:.2f}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if self.debug_img_pub.get_subscription_count() > 0:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    # ==================== OBSTACLE AVOIDANCE ====================
    
    def check_obstacle_critical(self) -> bool:
        """Check if there's a critical obstacle requiring immediate stop"""
        return self.lidar_zones.front_narrow < self.CRITICAL_DISTANCE
    
    def check_obstacle_front(self) -> bool:
        """Check if there's an obstacle in front requiring avoidance"""
        return (self.lidar_zones.front < self.SAFE_DISTANCE or
                self.lidar_zones.front_left < self.SAFE_DISTANCE * 0.8 or
                self.lidar_zones.front_right < self.SAFE_DISTANCE * 0.8)
    
    def check_path_clear(self) -> bool:
        """Check if the path ahead is clear for movement"""
        return (self.lidar_zones.front > self.SAFE_DISTANCE and
                self.lidar_zones.front_left > self.SAFE_DISTANCE * 0.7 and
                self.lidar_zones.front_right > self.SAFE_DISTANCE * 0.7)
    
    def get_best_avoidance_direction(self) -> int:
        """
        Determine best direction to avoid obstacle.
        Returns: 1 for left turn, -1 for right turn
        """
        # Compare space on left vs right
        left_space = min(self.lidar_zones.front_left, self.lidar_zones.left)
        right_space = min(self.lidar_zones.front_right, self.lidar_zones.right)
        
        # If we're approaching a cube, prefer direction that keeps cube in view
        if self.angular_error is not None and self.state == RobotState.APPROACH:
            # Cube is on left (positive error), prefer slight right to keep it in view
            if self.angular_error > 50:
                return -1 if right_space > self.CRITICAL_DISTANCE else 1
            elif self.angular_error < -50:
                return 1 if left_space > self.CRITICAL_DISTANCE else -1
        
        # Default: go towards more open space
        if left_space > right_space + 0.1:
            return 1  # Turn left
        elif right_space > left_space + 0.1:
            return -1  # Turn right
        else:
            # Equal space, maintain previous direction or use search direction
            return self.search_direction

    def compute_obstacle_avoidance_twist(self) -> Twist:
        """
        Compute velocity command for obstacle avoidance using VFH-lite approach.
        """
        twist = Twist()
        
        # Emergency: very close obstacle
        if self.lidar_zones.front_narrow < self.CRITICAL_DISTANCE:
            twist.linear.x = -0.1  # Back up
            twist.angular.z = self.get_best_avoidance_direction() * self.MAX_ANGULAR_SPEED
            return twist
        
        # Calculate repulsive forces from obstacles
        front_force = max(0, (self.SAFE_DISTANCE - self.lidar_zones.front) / self.SAFE_DISTANCE)
        front_left_force = max(0, (self.SAFE_DISTANCE - self.lidar_zones.front_left) / self.SAFE_DISTANCE)
        front_right_force = max(0, (self.SAFE_DISTANCE - self.lidar_zones.front_right) / self.SAFE_DISTANCE)
        
        # Angular velocity: turn away from obstacles
        angular_z = (front_right_force - front_left_force) * self.MAX_ANGULAR_SPEED
        
        # If front is blocked, add stronger rotation
        if front_force > 0.3:
            direction = self.get_best_avoidance_direction()
            angular_z += direction * front_force * self.MAX_ANGULAR_SPEED
        
        # Linear velocity: slow down near obstacles
        max_force = max(front_force, front_left_force * 0.7, front_right_force * 0.7)
        linear_x = self.SEARCH_LINEAR_SPEED * (1.0 - max_force)
        linear_x = max(0.0, linear_x)  # Don't go backwards unless critical
        
        twist.linear.x = linear_x
        twist.angular.z = np.clip(angular_z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
        
        return twist

    def is_stuck(self) -> bool:
        """Detect if robot is stuck (not making progress)"""
        if len(self.position_history) < 20:
            return False
            
        # Check movement over last 2 seconds (20 samples at 10Hz)
        recent = list(self.position_history)[-20:]
        first_pos = recent[0]
        last_pos = recent[-1]
        
        dist_moved = math.sqrt(
            (last_pos[0] - first_pos[0])**2 + 
            (last_pos[1] - first_pos[1])**2
        )
        
        # If moved less than 5cm in 2 seconds while trying to move, we're stuck
        return dist_moved < 0.05

    def handle_stuck(self) -> Twist:
        """Generate recovery behavior when stuck"""
        twist = Twist()
        self.stuck_counter += 1
        
        if self.stuck_counter < 10:
            # Try backing up
            twist.linear.x = -0.15
            twist.angular.z = self.get_best_avoidance_direction() * 0.5
        elif self.stuck_counter < 20:
            # Rotate in place
            twist.angular.z = self.get_best_avoidance_direction() * self.MAX_ANGULAR_SPEED
        else:
            # Reset counter and try different direction
            self.stuck_counter = 0
            self.search_direction *= -1
            
        return twist

    # ==================== SEARCH BEHAVIOR ====================
    
    def compute_search_twist(self) -> Twist:
        """
        Optimized search behavior with obstacle avoidance.
        Uses a combination of spiral and reactive behaviors.
        """
        twist = Twist()
        
        # Check for stuck condition
        if self.is_stuck():
            return self.handle_stuck()
        else:
            self.stuck_counter = 0
        
        # Check for obstacles
        if self.check_obstacle_critical():
            twist.linear.x = -0.1
            twist.angular.z = self.get_best_avoidance_direction() * self.MAX_ANGULAR_SPEED
            return twist
        
        if self.check_obstacle_front():
            return self.compute_obstacle_avoidance_twist()
        
        # Path is clear - explore
        search_time = time.time() - self.search_start_time
        
        # Adaptive search pattern
        if search_time < 3.0:
            # First, turn towards last known cube direction
            if self.last_cube_direction != 0:
                twist.angular.z = self.last_cube_direction * 0.4
                twist.linear.x = 0.1
            else:
                # Slow rotation to scan
                twist.angular.z = self.search_direction * 0.5
                twist.linear.x = 0.05
        else:
            # Move forward with slight curve
            twist.linear.x = self.SEARCH_LINEAR_SPEED
            
            # Add some exploration behavior
            phase = (search_time % 10.0) / 10.0
            if phase < 0.3:
                twist.angular.z = self.search_direction * 0.3
            elif phase < 0.7:
                twist.angular.z = 0.0  # Straight
            else:
                twist.angular.z = -self.search_direction * 0.2
            
            # Change search direction periodically
            if int(search_time) % 15 == 0 and int(search_time) > 0:
                self.search_direction *= -1
        
        # Gentle wall following if we're near a wall on one side
        if self.lidar_zones.left < self.WALL_FOLLOW_DISTANCE:
            twist.angular.z -= 0.2  # Slight turn right
        elif self.lidar_zones.right < self.WALL_FOLLOW_DISTANCE:
            twist.angular.z += 0.2  # Slight turn left
        
        return twist

    # ==================== APPROACH BEHAVIOR ====================
    
    def compute_approach_twist(self) -> Twist:
        """
        Visual servoing approach with obstacle avoidance.
        """
        twist = Twist()
        
        if self.angular_error is None:
            # Lost visual - return to search
            self.state = RobotState.SEARCH
            self.search_start_time = time.time()
            return twist
        
        # Check for critical obstacles (but not the cube itself)
        # If front is very close and we see the cube centered, it's probably the cube
        is_cube_centered = abs(self.angular_error) < 50
        min_front = self.lidar_zones.front_narrow
        
        # If we see large obstacle but cube is not centered, need to avoid
        if min_front < self.CRITICAL_DISTANCE and not is_cube_centered:
            twist.linear.x = 0.0
            twist.angular.z = self.get_best_avoidance_direction() * 0.8
            return twist
        
        # Visual servoing with P-controller
        kp_angular = 0.004
        twist.angular.z = kp_angular * self.angular_error
        twist.angular.z = np.clip(twist.angular.z, -0.8, 0.8)
        
        # Adaptive forward speed based on distance and alignment
        alignment_factor = 1.0 - min(abs(self.angular_error) / 200.0, 0.5)
        
        # Slow down as we get closer
        distance_factor = min(min_front / self.SAFE_DISTANCE, 1.0)
        
        twist.linear.x = self.APPROACH_LINEAR_SPEED * alignment_factor * distance_factor
        
        # Add gentle obstacle avoidance without losing target
        if self.lidar_zones.front_left < self.APPROACH_SAFE_DISTANCE:
            twist.angular.z -= 0.3  # Gentle right adjustment
        if self.lidar_zones.front_right < self.APPROACH_SAFE_DISTANCE:
            twist.angular.z += 0.3  # Gentle left adjustment
        
        # Check if close enough to grab
        if min_front < 0.38 and is_cube_centered:
            self.get_logger().info(f"Close enough! (Lidar: {min_front:.2f}m). GRABBING.")
            self.state = RobotState.GRAB
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        return twist

    # ==================== RETURN BEHAVIOR ====================
    
    def compute_return_twist(self) -> Twist:
        """
        Return to home with obstacle avoidance.
        Uses potential field approach.
        """
        twist = Twist()
        
        if self.current_pose is None:
            return twist
            
        rx, ry, rtheta = self.current_pose
        
        # Calculate distance and angle to home
        dist_to_home = math.sqrt(rx**2 + ry**2)
        angle_to_home = math.atan2(-ry, -rx)
        angle_diff = self._normalize_angle(angle_to_home - rtheta)
        
        # Check if arrived
        if dist_to_home < 0.35:
            self.state = RobotState.DROP
            return twist
        
        # Attractive force towards home
        attract_linear = min(self.RETURN_LINEAR_SPEED, dist_to_home * 0.5)
        attract_angular = 1.2 * angle_diff
        
        # Repulsive forces from obstacles
        repulse_angular = 0.0
        repulse_linear = 0.0
        
        if self.check_obstacle_critical():
            # Emergency avoidance
            twist.linear.x = -0.1
            twist.angular.z = self.get_best_avoidance_direction() * self.MAX_ANGULAR_SPEED
            return twist
        
        if self.lidar_zones.front < self.SAFE_DISTANCE:
            # Front obstacle - strong repulsion
            repulse = (self.SAFE_DISTANCE - self.lidar_zones.front) / self.SAFE_DISTANCE
            repulse_angular = self.get_best_avoidance_direction() * repulse * 1.5
            repulse_linear = -repulse * 0.15
            
        if self.lidar_zones.front_left < self.SAFE_DISTANCE:
            repulse = (self.SAFE_DISTANCE - self.lidar_zones.front_left) / self.SAFE_DISTANCE
            repulse_angular -= repulse * 0.8  # Turn right
            
        if self.lidar_zones.front_right < self.SAFE_DISTANCE:
            repulse = (self.SAFE_DISTANCE - self.lidar_zones.front_right) / self.SAFE_DISTANCE
            repulse_angular += repulse * 0.8  # Turn left
        
        # Combine attractive and repulsive forces
        twist.angular.z = attract_angular + repulse_angular
        twist.angular.z = np.clip(twist.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
        
        # Only move forward if roughly facing the right direction
        if abs(angle_diff) < 0.8:
            twist.linear.x = attract_linear + repulse_linear
            twist.linear.x = max(0.0, min(twist.linear.x, self.MAX_LINEAR_SPEED))
        else:
            twist.linear.x = 0.0  # Rotate in place first
        
        return twist

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    # ==================== MAIN CONTROL LOOP ====================
    
    def control_loop(self):
        if self.current_pose is None:
            return

        twist = Twist()

        if self.state == RobotState.SEARCH:
            # Check if we found a target
            if self.angular_error is not None:
                self.get_logger().info("Target in sight! Switching to APPROACH")
                self.state = RobotState.APPROACH
            else:
                twist = self.compute_search_twist()

        elif self.state == RobotState.APPROACH:
            twist = self.compute_approach_twist()

        elif self.state == RobotState.GRAB:
            twist = self._handle_grab_state()

        elif self.state == RobotState.RETURN:
            twist = self.compute_return_twist()

        elif self.state == RobotState.DROP:
            self._handle_drop_state()
            return  # Drop handles its own publishing

        self.cmd_vel_pub.publish(twist)

    def _handle_grab_state(self) -> Twist:
        """Handle the GRAB state logic"""
        twist = Twist()
        target_name = self.get_closest_cube_name()

        if target_name:
            self.get_logger().info(f"Grabbing {target_name}...")
            self.delete_cube(target_name)
            self.has_cube = True
            self.state = RobotState.RETURN
        else:
            self.get_logger().info("Grab failed - nothing in range. Retrying...")
            # Move forward a tiny bit and try again
            twist.linear.x = 0.05
            
            # If can't grab after a moment, go back to search
            if self.lidar_zones.front_narrow > 0.5:
                self.state = RobotState.SEARCH
                self.search_start_time = time.time()
        
        return twist

    def _handle_drop_state(self):
        """Handle the DROP state logic"""
        self.get_logger().info(f"Dropping Cube #{self.cubes_collected + 1}")
        self.respawn_collected_cube()
        self.cubes_collected += 1
        self.has_cube = False

        # Back up smoothly
        twist = Twist()
        twist.linear.x = -0.15
        
        for _ in range(10):  # Back up for 1 second
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        
        self.state = RobotState.SEARCH
        self.search_start_time = time.time()
        self.last_cube_direction = 0  # Reset search preference

    # ==================== UTILITY FUNCTIONS ====================
    
    def get_closest_cube_name(self):
        """Virtual Gripper logic - find closest cube using ground truth"""
        if not self.latest_model_states:
            return None

        robot_pose = None
        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        for i, name in enumerate(names):
            if name in ["waffle_pi", "burger", "turtlebot3_waffle_pi"]:
                robot_pose = (poses[i].position.x, poses[i].position.y)
                break

        if robot_pose is None:
            return None

        rx, ry = robot_pose
        min_dist = float('inf')
        best_name = None

        for i, name in enumerate(names):
            if name.startswith("cube_"):
                cx = poses[i].position.x
                cy = poses[i].position.y
                dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                
                if dist < min_dist:
                    min_dist = dist
                    best_name = name

        # Threshold for grabbing (robot radius + cube radius + buffer)
        if min_dist < 0.45:
            return best_name
        return None

    def delete_cube(self, name):
        req = DeleteEntity.Request()
        req.name = name
        self.delete_client.call_async(req)

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

    def spawn_initial_cubes(self):
        self.spawn_timer.cancel()
        
        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Spawn service not available.")
            return

        import random
        
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

        n_cubes = 5
        for i in range(n_cubes):
            req = SpawnEntity.Request()
            req.name = f"cube_{i}"
            req.xml = sdf

            while True:
                rx = random.uniform(-2.0, 2.0)
                ry = random.uniform(-2.0, 2.0)
                if (rx**2 + ry**2) > 0.7**2:
                    break

            req.initial_pose.position.x = rx
            req.initial_pose.position.y = ry
            req.initial_pose.position.z = 0.5
            self.spawn_client.call_async(req)

        self.get_logger().info(f"Spawned {n_cubes} cubes.")


def main(args=None):
    rclpy.init(args=args)
    node = CubeCollectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot before shutdown
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()