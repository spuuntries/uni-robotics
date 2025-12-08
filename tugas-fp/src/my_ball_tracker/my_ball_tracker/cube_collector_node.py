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
    """State machine states"""
    SEARCH = auto()
    APPROACH = auto()
    GRAB = auto()
    RETURN = auto()
    DROP = auto()
    EMERGENCY_AVOID = auto()  # New state for emergency avoidance


class LidarZones:
    """Structured lidar zone data with blind spot detection"""
    def __init__(self):
        self.front = float('inf')
        self.front_left = float('inf')
        self.front_right = float('inf')
        self.left = float('inf')
        self.right = float('inf')
        self.back = float('inf')
        self.front_narrow = float('inf')
        
        # Blind spot indicators (True = potential blind spot detected)
        self.front_blind = False
        self.front_left_blind = False
        self.front_right_blind = False
        
        # Raw counts for analysis
        self.front_invalid_count = 0
        self.front_valid_count = 0


class VelocityMonitor:
    """Monitor actual vs commanded velocity to detect collisions"""
    def __init__(self, window_size=20):
        self.cmd_history = deque(maxlen=window_size)
        self.pos_history = deque(maxlen=window_size)
        self.collision_suspected = False
        
    def update(self, cmd_vel: Twist, position: tuple, timestamp: float):
        self.cmd_history.append((cmd_vel.linear.x, cmd_vel.angular.z, timestamp))
        self.pos_history.append((position[0], position[1], timestamp))
        
    def check_collision(self) -> bool:
        """
        Detect collision by comparing commanded velocity with actual movement.
        If we're commanding forward motion but not moving, we might be stuck.
        """
        if len(self.cmd_history) < 10 or len(self.pos_history) < 10:
            return False
            
        # Get recent command history
        recent_cmds = list(self.cmd_history)[-10:]
        recent_pos = list(self.pos_history)[-10:]
        
        # Calculate average commanded linear velocity
        avg_cmd_linear = sum(c[0] for c in recent_cmds) / len(recent_cmds)
        
        # Calculate actual displacement
        if len(recent_pos) >= 2:
            dx = recent_pos[-1][0] - recent_pos[0][0]
            dy = recent_pos[-1][1] - recent_pos[0][1]
            dt = recent_pos[-1][2] - recent_pos[0][2]
            
            if dt > 0.1:  # At least 100ms of data
                actual_speed = math.sqrt(dx**2 + dy**2) / dt
                
                # If commanding significant forward velocity but barely moving
                if avg_cmd_linear > 0.1 and actual_speed < 0.03:
                    self.collision_suspected = True
                    return True
                    
        self.collision_suspected = False
        return False


class CubeCollectorNode(Node):
    def __init__(self):
        super().__init__("cube_collector_node")
        self.get_logger().info("Cube Collector Node Started! (Anti-Collision Version)")

        # --- Parameters ---
        self.declare_parameter("target_color", "red")
        self.declare_parameter("pickup_distance", 0.45)
        
        # ============ OBSTACLE AVOIDANCE PARAMETERS ============
        # Lidar characteristics (TurtleBot3 LDS-01)
        self.LIDAR_MIN_RANGE = 0.12        # Minimum reliable range
        self.LIDAR_MAX_RANGE = 3.5         # Maximum reliable range
        
        # Safety distances - LEBIH BESAR untuk pilar
        self.PILLAR_SAFE_DISTANCE = 0.55   # Jarak aman dari pilar (lebih jauh)
        self.PILLAR_CRITICAL_DISTANCE = 0.40  # Jarak kritis dari pilar
        self.CUBE_APPROACH_DISTANCE = 0.35  # Boleh lebih dekat ke cube
        
        # General safety
        self.SAFE_DISTANCE = 0.50
        self.CRITICAL_DISTANCE = 0.35
        self.EMERGENCY_DISTANCE = 0.25     # Sangat bahaya - immediate reverse
        
        # Blind spot detection thresholds
        self.BLIND_SPOT_INVALID_RATIO = 0.4  # If >40% readings invalid, suspect blind spot
        self.MIN_VALID_READINGS = 3          # Minimum valid readings needed
        
        # Speed parameters
        self.MAX_LINEAR_SPEED = 0.20
        self.MAX_ANGULAR_SPEED = 1.5
        self.SEARCH_LINEAR_SPEED = 0.15
        self.APPROACH_LINEAR_SPEED = 0.10
        self.RETURN_LINEAR_SPEED = 0.18
        self.EMERGENCY_REVERSE_SPEED = -0.15

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
        self.previous_state = None
        self.current_pose = None
        self.has_cube = False
        self.cubes_collected = 0
        self.home_pose = (0.0, 0.0)

        # Model states
        self.latest_model_states = None

        # Lidar data
        self.lidar_zones = LidarZones()
        self.raw_scan_ranges = []
        self.scan_angle_min = 0.0
        self.scan_angle_increment = 0.0

        # Vision data
        self.angular_error = None
        self.target_area = 0
        self.target_in_view = False
        self.target_bbox = None  # Bounding box of detected cube
        self.last_img_time = time.time()
        
        # Search optimization
        self.search_direction = 1
        self.search_start_time = time.time()
        self.last_cube_direction = 0
        
        # Collision/stuck detection
        self.velocity_monitor = VelocityMonitor()
        self.position_history = deque(maxlen=50)
        self.stuck_counter = 0
        self.emergency_start_time = 0
        self.emergency_direction = 1
        self.consecutive_emergency_count = 0
        
        # Last commanded velocity (untuk monitoring)
        self.last_cmd_vel = Twist()

        # Create control timer (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # Spawn cubes
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
        
        # Update monitors
        current_time = time.time()
        self.position_history.append((pos.x, pos.y, current_time))
        self.velocity_monitor.update(self.last_cmd_vel, (pos.x, pos.y), current_time)

    def scan_callback(self, msg):
        """Process lidar scan with blind spot detection"""
        self.raw_scan_ranges = list(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment
        self._process_lidar_zones_with_blind_detection()

    def _process_lidar_zones_with_blind_detection(self):
        """
        Process lidar data dengan deteksi blind spot.
        Jika banyak reading yang invalid (inf, nan, atau < min_range),
        kemungkinan ada obstacle sangat dekat (blind spot).
        """
        if len(self.raw_scan_ranges) == 0:
            return
            
        n = len(self.raw_scan_ranges)
        
        def analyze_zone(start_deg, end_deg):
            """
            Analyze a zone and return (min_distance, blind_spot_detected, invalid_count, valid_count)
            """
            deg_to_idx = n / 360.0
            
            # Convert degrees to indices (handle wraparound)
            start_idx = int(start_deg * deg_to_idx) % n
            end_idx = int(end_deg * deg_to_idx) % n
            
            # Get indices in range
            if start_idx <= end_idx:
                indices = range(start_idx, end_idx)
            else:  # Wraparound
                indices = list(range(start_idx, n)) + list(range(0, end_idx))
            
            valid_readings = []
            invalid_count = 0
            too_close_count = 0
            
            for i in indices:
                r = self.raw_scan_ranges[i]
                
                # Check for invalid readings
                if math.isinf(r) or math.isnan(r):
                    invalid_count += 1
                elif r < self.LIDAR_MIN_RANGE:
                    # Reading below minimum range - something VERY close
                    too_close_count += 1
                    invalid_count += 1
                elif r < self.LIDAR_MAX_RANGE:
                    valid_readings.append(r)
                else:
                    invalid_count += 1
            
            total_readings = len(list(indices))
            valid_count = len(valid_readings)
            
            # Detect blind spot condition
            blind_spot = False
            if total_readings > 0:
                invalid_ratio = invalid_count / total_readings
                
                # Blind spot suspected if:
                # 1. High ratio of invalid readings
                # 2. Too many "too close" readings
                # 3. Very few valid readings
                if (invalid_ratio > self.BLIND_SPOT_INVALID_RATIO or 
                    too_close_count > 2 or
                    (valid_count < self.MIN_VALID_READINGS and total_readings > 5)):
                    blind_spot = True
            
            min_dist = min(valid_readings) if valid_readings else float('inf')
            
            return min_dist, blind_spot, invalid_count, valid_count
        
        # Analyze each zone
        # Front narrow: -15 to +15 degrees (350 to 10)
        min_d, blind, inv, val = analyze_zone(345, 15)
        self.lidar_zones.front_narrow = min_d
        self.lidar_zones.front_blind = blind
        self.lidar_zones.front_invalid_count = inv
        self.lidar_zones.front_valid_count = val
        
        # Front wide: -30 to +30 degrees (330 to 30)
        self.lidar_zones.front, _, _, _ = analyze_zone(330, 30)
        
        # Front-left: 30 to 60 degrees
        self.lidar_zones.front_left, self.lidar_zones.front_left_blind, _, _ = analyze_zone(30, 60)
        
        # Front-right: 300 to 330 degrees
        self.lidar_zones.front_right, self.lidar_zones.front_right_blind, _, _ = analyze_zone(300, 330)
        
        # Left: 60 to 120 degrees
        self.lidar_zones.left, _, _, _ = analyze_zone(60, 120)
        
        # Right: 240 to 300 degrees
        self.lidar_zones.right, _, _, _ = analyze_zone(240, 300)
        
        # Back: 150 to 210 degrees
        self.lidar_zones.back, _, _, _ = analyze_zone(150, 210)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.last_img_time = time.time()
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        h, w, _ = cv_image.shape
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red color detection
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        if self.debug_mask_pub.get_subscription_count() > 0:
            self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.target_in_view = False
        self.angular_error = None
        self.target_area = 0
        self.target_bbox = None

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 150:
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Get bounding box
                    x, y, bw, bh = cv2.boundingRect(largest_contour)
                    self.target_bbox = (x, y, bw, bh)

                    cv2.circle(cv_image, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.rectangle(cv_image, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

                    self.angular_error = (w / 2) - cx
                    self.target_area = area
                    self.target_in_view = True
                    self.last_cube_direction = 1 if self.angular_error > 0 else -1

        # Draw debug info
        self._draw_debug_info(cv_image)

        if self.debug_img_pub.get_subscription_count() > 0:
            self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def _draw_debug_info(self, cv_image):
        """Draw debugging information on image"""
        h, w, _ = cv_image.shape
        
        # State info
        cv2.putText(cv_image, f"State: {self.state.name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Lidar info
        front_color = (0, 0, 255) if self.lidar_zones.front_blind else (255, 255, 0)
        cv2.putText(cv_image, f"Front: {self.lidar_zones.front_narrow:.2f}m", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, front_color, 1)
        
        # Blind spot warning
        if self.lidar_zones.front_blind:
            cv2.putText(cv_image, "!! BLIND SPOT !!", (w//2 - 80, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Collision warning
        if self.velocity_monitor.collision_suspected:
            cv2.putText(cv_image, "!! COLLISION !!", (w//2 - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Target info
        if self.target_in_view:
            cv2.putText(cv_image, f"Target Area: {self.target_area}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # ==================== OBSTACLE DETECTION ====================
    
    def is_obstacle_pillar(self) -> bool:
        """
        Determine if the obstacle in front is a PILLAR (not a cube).
        
        Logic:
        - If we see red cube in camera AND it's centered, obstacle = cube
        - If we see obstacle but NO red cube, obstacle = pillar
        - If blind spot detected but no red cube, assume pillar
        """
        has_obstacle = (
            self.lidar_zones.front_narrow < self.PILLAR_SAFE_DISTANCE or
            self.lidar_zones.front_blind or
            self.velocity_monitor.collision_suspected
        )
        
        if not has_obstacle:
            return False
            
        # If we see a red cube centered in view, it's probably the cube
        if self.target_in_view and abs(self.angular_error) < 100:
            # Additional check: is the cube taking up significant space?
            # Large area + centered = definitely cube
            if self.target_area > 500:
                return False  # It's a cube, not pillar
        
        # No cube visible or cube not centered = pillar
        return True
    
    def detect_imminent_collision(self) -> bool:
        """
        Detect if collision is imminent or already happening.
        Uses multiple signals:
        1. Lidar blind spot
        2. Velocity mismatch (commanding forward but not moving)
        3. Very close readings
        """
        # Blind spot in front - something very close
        if self.lidar_zones.front_blind:
            return True
            
        # Velocity-based collision detection
        if self.velocity_monitor.check_collision():
            return True
            
        # Emergency distance breach
        if self.lidar_zones.front_narrow < self.EMERGENCY_DISTANCE:
            return True
            
        # Multiple blind spots
        blind_count = sum([
            self.lidar_zones.front_blind,
            self.lidar_zones.front_left_blind,
            self.lidar_zones.front_right_blind
        ])
        if blind_count >= 2:
            return True
            
        return False
    
    def check_pillar_proximity(self) -> tuple:
        """
        Check proximity to pillar (not cube).
        Returns: (is_near_pillar, danger_level, suggested_direction)
        
        danger_level: 0=safe, 1=caution, 2=danger, 3=critical
        """
        if not self.is_obstacle_pillar():
            return False, 0, 0
            
        # Get minimum front distance
        front_dist = self.lidar_zones.front_narrow
        
        # If blind spot, assume very close
        if self.lidar_zones.front_blind:
            front_dist = self.EMERGENCY_DISTANCE
            
        # Determine danger level
        if front_dist < self.EMERGENCY_DISTANCE:
            danger_level = 3  # Critical
        elif front_dist < self.PILLAR_CRITICAL_DISTANCE:
            danger_level = 2  # Danger
        elif front_dist < self.PILLAR_SAFE_DISTANCE:
            danger_level = 1  # Caution
        else:
            danger_level = 0  # Safe
            
        # Suggest avoidance direction
        left_space = min(self.lidar_zones.front_left, self.lidar_zones.left)
        right_space = min(self.lidar_zones.front_right, self.lidar_zones.right)
        
        # Prefer direction with more space
        if left_space > right_space + 0.1:
            direction = 1  # Turn left
        elif right_space > left_space + 0.1:
            direction = -1  # Turn right
        else:
            direction = self.search_direction
            
        return True, danger_level, direction

    # ==================== EMERGENCY AVOIDANCE ====================
    
    def handle_emergency_avoidance(self) -> Twist:
        """
        Emergency collision avoidance behavior.
        Called when collision is imminent or happening.
        """
        twist = Twist()
        
        current_time = time.time()
        
        # Initialize emergency if just started
        if self.state != RobotState.EMERGENCY_AVOID:
            self.previous_state = self.state
            self.state = RobotState.EMERGENCY_AVOID
            self.emergency_start_time = current_time
            self.consecutive_emergency_count += 1
            
            # Determine escape direction
            left_clear = not self.lidar_zones.front_left_blind and self.lidar_zones.front_left > 0.3
            right_clear = not self.lidar_zones.front_right_blind and self.lidar_zones.front_right > 0.3
            
            if left_clear and not right_clear:
                self.emergency_direction = 1
            elif right_clear and not left_clear:
                self.emergency_direction = -1
            elif self.lidar_zones.left > self.lidar_zones.right:
                self.emergency_direction = 1
            else:
                self.emergency_direction = -1
                
            self.get_logger().warn(f"EMERGENCY AVOIDANCE! Direction: {'LEFT' if self.emergency_direction > 0 else 'RIGHT'}")
        
        elapsed = current_time - self.emergency_start_time
        
        # Phase 1: Back up (0-1.5s)
        if elapsed < 1.5:
            twist.linear.x = self.EMERGENCY_REVERSE_SPEED
            twist.angular.z = self.emergency_direction * 0.5
            
        # Phase 2: Rotate away (1.5-3s)
        elif elapsed < 3.0:
            twist.linear.x = 0.0
            twist.angular.z = self.emergency_direction * self.MAX_ANGULAR_SPEED
            
        # Phase 3: Check if clear and exit
        else:
            if self.is_path_clear_for_exit():
                self.get_logger().info("Emergency avoidance complete. Resuming.")
                self.state = self.previous_state if self.previous_state else RobotState.SEARCH
                self.search_start_time = time.time()
                
                # If too many consecutive emergencies, reset search
                if self.consecutive_emergency_count > 3:
                    self.get_logger().warn("Too many emergencies. Full reset.")
                    self.state = RobotState.SEARCH
                    self.consecutive_emergency_count = 0
            else:
                # Continue rotating
                twist.angular.z = self.emergency_direction * self.MAX_ANGULAR_SPEED
                
                # If stuck in emergency for too long, try opposite direction
                if elapsed > 5.0:
                    self.emergency_direction *= -1
                    self.emergency_start_time = current_time
        
        return twist
    
    def is_path_clear_for_exit(self) -> bool:
        """Check if path is clear to exit emergency avoidance"""
        return (
            self.lidar_zones.front > self.PILLAR_SAFE_DISTANCE and
            not self.lidar_zones.front_blind and
            self.lidar_zones.front_left > self.CRITICAL_DISTANCE and
            self.lidar_zones.front_right > self.CRITICAL_DISTANCE and
            not self.velocity_monitor.collision_suspected
        )

    # ==================== SEARCH BEHAVIOR ====================
    
    def compute_search_twist(self) -> Twist:
        """Search behavior with pillar avoidance"""
        twist = Twist()
        
        # Check for pillar proximity
        near_pillar, danger_level, avoid_dir = self.check_pillar_proximity()
        
        if danger_level >= 2:  # Danger or critical
            # Strong avoidance
            twist.linear.x = 0.0 if danger_level == 3 else 0.05
            twist.angular.z = avoid_dir * self.MAX_ANGULAR_SPEED
            return twist
            
        if danger_level == 1:  # Caution
            # Gentle avoidance while continuing search
            twist.linear.x = self.SEARCH_LINEAR_SPEED * 0.5
            twist.angular.z = avoid_dir * 0.8
            return twist
        
        # Check stuck condition
        if self.is_stuck():
            return self.handle_stuck()
        else:
            self.stuck_counter = 0
            self.consecutive_emergency_count = 0  # Reset if moving well
        
        # Normal search behavior
        search_time = time.time() - self.search_start_time
        
        # Adaptive search
        if search_time < 3.0:
            if self.last_cube_direction != 0:
                twist.angular.z = self.last_cube_direction * 0.4
                twist.linear.x = 0.08
            else:
                twist.angular.z = self.search_direction * 0.5
                twist.linear.x = 0.05
        else:
            twist.linear.x = self.SEARCH_LINEAR_SPEED
            
            phase = (search_time % 8.0) / 8.0
            if phase < 0.25:
                twist.angular.z = self.search_direction * 0.3
            elif phase < 0.75:
                twist.angular.z = 0.0
            else:
                twist.angular.z = -self.search_direction * 0.2
            
            if int(search_time) % 12 == 0 and int(search_time) > 0:
                self.search_direction *= -1
        
        # Proactive wall/pillar avoidance
        twist = self.apply_reactive_avoidance(twist)
        
        return twist
    
    def apply_reactive_avoidance(self, twist: Twist) -> Twist:
        """Apply reactive obstacle avoidance to any twist command"""
        
        # Skip if in emergency
        if self.state == RobotState.EMERGENCY_AVOID:
            return twist
            
        # Front obstacle
        if self.lidar_zones.front < self.SAFE_DISTANCE:
            scale = (self.lidar_zones.front - self.CRITICAL_DISTANCE) / (self.SAFE_DISTANCE - self.CRITICAL_DISTANCE)
            scale = max(0.0, min(1.0, scale))
            twist.linear.x *= scale
            
            # Add rotation away
            _, _, avoid_dir = self.check_pillar_proximity()
            twist.angular.z += avoid_dir * (1.0 - scale) * 0.8
        
        # Side obstacles
        if self.lidar_zones.front_left < self.SAFE_DISTANCE:
            twist.angular.z -= 0.3
        if self.lidar_zones.front_right < self.SAFE_DISTANCE:
            twist.angular.z += 0.3
            
        # Clamp values
        twist.linear.x = max(0.0, min(twist.linear.x, self.MAX_LINEAR_SPEED))
        twist.angular.z = max(-self.MAX_ANGULAR_SPEED, min(twist.angular_z, self.MAX_ANGULAR_SPEED))
        
        return twist

    def is_stuck(self) -> bool:
        """Detect if robot is stuck"""
        if len(self.position_history) < 20:
            return False
            
        recent = list(self.position_history)[-20:]
        first_pos = recent[0]
        last_pos = recent[-1]
        
        dist_moved = math.sqrt(
            (last_pos[0] - first_pos[0])**2 + 
            (last_pos[1] - first_pos[1])**2
        )
        
        return dist_moved < 0.05

    def handle_stuck(self) -> Twist:
        """Recovery when stuck"""
        twist = Twist()
        self.stuck_counter += 1
        
        _, _, avoid_dir = self.check_pillar_proximity()
        
        if self.stuck_counter < 15:
            twist.linear.x = -0.12
            twist.angular.z = avoid_dir * 0.5
        elif self.stuck_counter < 30:
            twist.angular.z = avoid_dir * self.MAX_ANGULAR_SPEED
        else:
            self.stuck_counter = 0
            self.search_direction *= -1
            
        return twist

    # ==================== APPROACH BEHAVIOR ====================
    
    def compute_approach_twist(self) -> Twist:
        """Approach cube with pillar avoidance"""
        twist = Twist()
        
        if not self.target_in_view:
            self.state = RobotState.SEARCH
            self.search_start_time = time.time()
            return twist
        
        # Check if obstacle is pillar (not the cube)
        near_pillar, danger_level, avoid_dir = self.check_pillar_proximity()
        
        if near_pillar and danger_level >= 2:
            # Obstacle is pillar, not cube - avoid it
            self.get_logger().info("Pillar detected during approach - avoiding")
            twist.linear.x = 0.0
            twist.angular.z = avoid_dir * 0.8
            return twist
        
        # Visual servoing
        kp_angular = 0.003
        twist.angular.z = kp_angular * self.angular_error
        twist.angular_z = np.clip(twist.angular.z, -0.6, 0.6)
        
        # Forward speed based on alignment
        alignment_factor = 1.0 - min(abs(self.angular_error) / 200.0, 0.5)
        twist.linear.x = self.APPROACH_LINEAR_SPEED * alignment_factor
        
        # Check distance to target
        front_dist = self.lidar_zones.front_narrow
        
        # If close to something and it's the cube (centered, red visible)
        is_cube_centered = abs(self.angular_error) < 60
        
        if front_dist < 0.38 and is_cube_centered and self.target_area > 300:
            self.get_logger().info(f"Close to cube! (Lidar: {front_dist:.2f}m). GRABBING.")
            self.state = RobotState.GRAB
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        
        # Apply gentle avoidance for side obstacles
        if self.lidar_zones.front_left < self.CUBE_APPROACH_DISTANCE:
            twist.angular.z -= 0.2
        if self.lidar_zones.front_right < self.CUBE_APPROACH_DISTANCE:
            twist.angular.z += 0.2
        
        return twist

    # ==================== RETURN BEHAVIOR ====================
    
    def compute_return_twist(self) -> Twist:
        """Return home with pillar avoidance"""
        twist = Twist()
        
        if self.current_pose is None:
            return twist
            
        rx, ry, rtheta = self.current_pose
        
        dist_to_home = math.sqrt(rx**2 + ry**2)
        angle_to_home = math.atan2(-ry, -rx)
        angle_diff = self._normalize_angle(angle_to_home - rtheta)
        
        if dist_to_home < 0.35:
            self.state = RobotState.DROP
            return twist
        
        # Check for pillar
        near_pillar, danger_level, avoid_dir = self.check_pillar_proximity()
        
        if danger_level >= 2:
            # Pillar in way - avoid first
            twist.linear.x = 0.0 if danger_level == 3 else 0.05
            twist.angular.z = avoid_dir * self.MAX_ANGULAR_SPEED * 0.8
            return twist
        
        # Navigation towards home with obstacle avoidance
        # Attractive force
        attract_angular = 1.0 * angle_diff
        attract_linear = min(self.RETURN_LINEAR_SPEED, dist_to_home * 0.4)
        
        # Repulsive force from obstacles
        repulse_angular = 0.0
        
        if self.lidar_zones.front < self.SAFE_DISTANCE:
            repulse = (self.SAFE_DISTANCE - self.lidar_zones.front) / self.SAFE_DISTANCE
            repulse_angular = avoid_dir * repulse * 1.2
            attract_linear *= (1.0 - repulse * 0.5)
            
        if self.lidar_zones.front_left < self.SAFE_DISTANCE * 0.8:
            repulse_angular -= 0.5
        if self.lidar_zones.front_right < self.SAFE_DISTANCE * 0.8:
            repulse_angular += 0.5
        
        twist.angular.z = attract_angular + repulse_angular
        twist.angular.z = np.clip(twist.angular.z, -self.MAX_ANGULAR_SPEED, self.MAX_ANGULAR_SPEED)
        
        if abs(angle_diff) < 0.7:
            twist.linear.x = max(0.0, min(attract_linear, self.MAX_LINEAR_SPEED))
        else:
            twist.linear.x = 0.0
        
        return twist

    @staticmethod
    def _normalize_angle(angle: float) -> float:
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
        
        # === PRIORITY 1: Emergency collision detection ===
        # Check for imminent collision (except when grabbing cube)
        if self.state not in [RobotState.GRAB, RobotState.DROP]:
            if self.detect_imminent_collision():
                # Check if it's a pillar (not cube)
                if self.is_obstacle_pillar():
                    twist = self.handle_emergency_avoidance()
                    self.cmd_vel_pub.publish(twist)
                    self.last_cmd_vel = twist
                    return
        
        # === PRIORITY 2: Continue emergency avoidance if active ===
        if self.state == RobotState.EMERGENCY_AVOID:
            twist = self.handle_emergency_avoidance()
            self.cmd_vel_pub.publish(twist)
            self.last_cmd_vel = twist
            return

        # === Normal state machine ===
        if self.state == RobotState.SEARCH:
            if self.target_in_view:
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
            return

        self.cmd_vel_pub.publish(twist)
        self.last_cmd_vel = twist

    def _handle_grab_state(self) -> Twist:
        twist = Twist()
        target_name = self.get_closest_cube_name()

        if target_name:
            self.get_logger().info(f"Grabbing {target_name}...")
            self.delete_cube(target_name)
            self.has_cube = True
            self.state = RobotState.RETURN
        else:
            self.get_logger().info("Grab failed - retrying...")
            twist.linear.x = 0.03
            
            if self.lidar_zones.front_narrow > 0.5 and not self.target_in_view:
                self.state = RobotState.SEARCH
                self.search_start_time = time.time()
        
        return twist

    def _handle_drop_state(self):
        self.get_logger().info(f"Dropping Cube #{self.cubes_collected + 1}")
        self.respawn_collected_cube()
        self.cubes_collected += 1
        self.has_cube = False

        twist = Twist()
        twist.linear.x = -0.12
        
        for _ in range(12):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        twist.linear.x = 0.0
        self.cmd_vel_pub.publish(twist)
        
        self.state = RobotState.SEARCH
        self.search_start_time = time.time()
        self.last_cube_direction = 0
        self.consecutive_emergency_count = 0

    # ==================== UTILITY FUNCTIONS ====================
    
    def get_closest_cube_name(self):
        if not self.latest_model_states:
            return None

        robot_pose = None
        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        for i, name in enumerate(names):
            if name in ["waffle_pi", "burger", "turtlebot3_waffle_pi", "turtlebot3_burger"]:
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
        twist = Twist()
        node.cmd_vel_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()