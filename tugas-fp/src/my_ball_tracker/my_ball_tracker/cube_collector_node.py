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
        self.get_logger().info("Cube Collector Node Started! (Anti-Stuck v2)")

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
            ModelStates, "/gazebo/model_states", self.model_states_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        )

        # --- Publishers ---
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_img_pub = self.create_publisher(Image, "/cube_collector/debug_image", 10)
        self.debug_mask_pub = self.create_publisher(Image, "/cube_collector/debug_mask", 10)

        # --- Services ---
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")

        # --- State Variables ---
        self.bridge = CvBridge()
        self.state = "SEARCH"
        self.prev_state = "SEARCH"

        self.current_pose = None
        self.has_cube = False
        self.cubes_collected = 0

        self.latest_model_states = None
        self.scan_ranges = []

        self.angular_error = None
        self.last_sighting_time = 0.0
        self.last_known_error = 0.0

        # Avoidance behavior
        self.avoid_until = 0.0
        self.avoid_direction = 1.0

        self.timer = self.create_timer(0.1, self.control_loop)
        self.spawn_timer = self.create_timer(2.0, self.spawn_initial_cubes)

    # ----------------------------------------------------------------------
    # Gazebo Models
    # ----------------------------------------------------------------------
    def model_states_callback(self, msg):
        self.latest_model_states = msg

    def spawn_initial_cubes(self):
        self.spawn_timer.cancel()
        n = 5

        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SpawnEntity service unavailable.")
            return

        sdf = """
        <?xml version='1.0'?>
        <sdf version="1.4">
            <model name="cube_model">
                <static>0</static>
                <link name="link">
                    <collision name="col">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                    </collision>
                    <visual name="vis">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                        <material><script>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                            <name>Gazebo/Red</name>
                        </script></material>
                    </visual>
                </link>
            </model>
        </sdf>
        """

        for i in range(n):
            req = SpawnEntity.Request()
            req.name = f"cube_{i}"
            req.xml = sdf

            while True:
                x = random.uniform(-1.5, 1.5)
                y = random.uniform(-1.5, 1.5)
                if x*x + y*y > 0.6**2:
                    break

            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.5

            self.spawn_client.call_async(req)

        self.get_logger().info(f"Spawned {n} cubes.")

    # ----------------------------------------------------------------------
    # LIDAR Processing
    # ----------------------------------------------------------------------
    def scan_callback(self, msg):
        self.scan_ranges = list(msg.ranges)

    def get_lidar_sector(self, sector="front"):
        if len(self.scan_ranges) == 0:
            return 99.9

        ranges = self.scan_ranges
        n = len(ranges)
        center = n // 2     # 0 rad
        window = 25

        if sector == "front":
            idxs = range(center - window, center + window)
        elif sector == "left":
            offset = n // 4
            idxs = range(center + offset - window, center + offset + window)
        elif sector == "right":
            offset = n // 4
            idxs = range(center - offset - window, center - offset + window)
        else:
            return 99.9

        vals = []
        for i in idxs:
            r = ranges[i % n]
            if (not math.isinf(r)) and (not math.isnan(r)) and r > 0.05:
                vals.append(r)

        if not vals:
            return 99.9
        return min(vals)

    # ----------------------------------------------------------------------
    # Camera
    # ----------------------------------------------------------------------
    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        low1 = np.array([0, 70, 50])
        up1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, low1, up1)

        low2 = np.array([170, 70, 50])
        up2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, low2, up2)

        mask = cv2.bitwise_or(mask1, mask2)

        if mask is not None:
            self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.angular_error = None

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > 120:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    h, w, _ = img.shape

                    cv2.circle(img, (cx, int(M["m01"]/M["m00"])), 8, (0,255,0), 2)

                    self.angular_error = (w/2) - cx
                    self.last_known_error = self.angular_error
                    self.last_sighting_time = time.time()

        self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

    # ----------------------------------------------------------------------
    # Odometry
    # ----------------------------------------------------------------------
    def odom_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        siny = 2 * (ori.w * ori.z + ori.x * ori.y)
        cosy = 1 - 2 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny, cosy)

        self.current_pose = (pos.x, pos.y, yaw)

    # ----------------------------------------------------------------------
    # Grab / Delete cube
    # ----------------------------------------------------------------------
    def get_closest_cube_name(self):
        if self.latest_model_states is None:
            return None

        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        robot_pos = None
        for i, n in enumerate(names):
            if n == "waffle_pi":
                robot_pos = (poses[i].position.x, poses[i].position.y)
                break

        if robot_pos is None:
            return None

        rx, ry = robot_pos
        best = None
        best_d = 99

        for i, n in enumerate(names):
            if n.startswith("cube_"):
                cx = poses[i].position.x
                cy = poses[i].position.y
                d = math.dist((rx,ry),(cx,cy))
                if d < best_d:
                    best_d = d
                    best = n

        if best_d < 1.0:
            return best
        return None

    def delete_cube(self, name):
        req = DeleteEntity.Request()
        req.name = name
        self.delete_client.call_async(req)

    def respawn_collected_cube(self):
        sdf = """
        <?xml version='1.0'?>
        <sdf version="1.4">
            <model name="collected">
                <static>1</static>
                <link name="link">
                    <visual name="vis">
                        <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                        <material><script>
                            <uri>file://media/materials/scripts/gazebo.material</uri>
                            <name>Gazebo/Green</name>
                        </script></material>
                    </visual>
                </link>
            </model>
        </sdf>
        """

        req = SpawnEntity.Request()
        req.name = f"collected_{self.cubes_collected}"
        req.xml = sdf
        req.initial_pose.position.x = 0.3 + (self.cubes_collected * 0.3)
        req.initial_pose.position.y = 0.0
        req.initial_pose.position.z = 0.1
        self.spawn_client.call_async(req)

    # ----------------------------------------------------------------------
    # MAIN CONTROL LOOP (with Anti-Stuck)
    # ----------------------------------------------------------------------
    def control_loop(self):
        if self.current_pose is None:
            return

        twist = Twist()

        # LIDAR distances
        df = self.get_lidar_sector("front")
        dl = self.get_lidar_sector("left")
        dr = self.get_lidar_sector("right")

        target_visible = self.angular_error is not None
        time_since_seen = time.time() - self.last_sighting_time

        # ==============================================================
        # GLOBAL: AVOID STATE (anti-stuck logic)
        # ==============================================================
        if self.state == "AVOID":
            if df > 0.7 and time.time() > self.avoid_until:
                self.state = self.prev_state
            else:
                twist.linear.x = -0.05
                twist.angular.z = 0.6 * self.avoid_direction
                self.cmd_vel_pub.publish(twist)
                return

        # ==============================================================    
        # NORMAL BEHAVIOR
        # ==============================================================    
        if self.state == "SEARCH":

            # 1. Hard-stop
            if df < 0.25:
                twist.linear.x = -0.1

            # 2. Visual lock
            elif target_visible:
                self.state = "APPROACH"

            # 3. Target recently lost
            elif df < 0.45 and time_since_seen < 1.0:
                twist.linear.x = -0.12

            # 4. Obstacle avoid → enter AVOID state
            elif df < 0.75:
                self.prev_state = "SEARCH"
                self.state = "AVOID"
                self.avoid_direction = -1.0 if dl > dr else 1.0
                self.avoid_until = time.time() + 1.0
                return

            # 5. Wandering
            else:
                twist.linear.x = 0.22
                twist.angular.z = 0.0

        elif self.state == "APPROACH":
            if not target_visible:
                self.state = "SEARCH"
            else:
                kp = 0.005
                twist.angular.z = kp * self.angular_error
                twist.linear.x = 0.15

                if df < 0.40:
                    self.state = "GRAB"

        elif self.state == "GRAB":
            tgt = self.get_closest_cube_name()
            if tgt:
                self.delete_cube(tgt)
                self.has_cube = True
                self.state = "RETURN"
            else:
                twist.linear.x = -0.1
                self.state = "SEARCH"

        elif self.state == "RETURN":
            x,y,yaw = self.current_pose
            dist = math.sqrt(x*x + y*y)

            if df < 0.3:
                # If blocked → Avoid
                self.prev_state = "RETURN"
                self.state = "AVOID"
                self.avoid_direction = -1.0 if dl > dr else 1.0
                self.avoid_until = time.time() + 1.2
                return

            ang = math.atan2(-y, -x)
            diff = ang - yaw
            diff = (diff + math.pi) % (2*math.pi) - math.pi

            twist.angular.z = 1.4 * diff
            if abs(diff) < 0.4:
                twist.linear.x = 0.18

            if dist < 0.35:
                self.state = "DROP"

        elif self.state == "DROP":
            self.respawn_collected_cube()
            self.cubes_collected += 1
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
