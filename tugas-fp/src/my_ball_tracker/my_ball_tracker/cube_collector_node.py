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
import random


class CubeCollectorNode(Node):
    def __init__(self):
        super().__init__("cube_collector_node")
        self.get_logger().info("Cube Collector Node Started! (FINAL ANTI-STUCK VERSION)")

        # ---- STATE VARIABLES ----
        self.state = "SEARCH"
        self.angular_error = None
        self.last_sighting_time = 0.0
        self.last_known_error = 0.0
        self.current_pose = None
        self.latest_model_states = None
        self.scan_ranges = []
        self.cubes_collected = 0

        self.bridge = CvBridge()

        # ---- SUBSCRIBERS ----
        self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(
            ModelStates, "/gazebo/model_states", self.model_states_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # ---- PUBLISHERS ----
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_img_pub = self.create_publisher(Image, "/cube_collector/debug_image", 10)
        self.debug_mask_pub = self.create_publisher(Image, "/cube_collector/debug_mask", 10)

        # ---- SERVICES ----
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")

        # ---- TIMERS ----
        self.create_timer(0.1, self.control_loop)
        self.spawn_timer = self.create_timer(2.0, self.spawn_initial_cubes)

    # -------------------------------------------------------------------------
    # SPAWN CUBES
    # -------------------------------------------------------------------------
    def spawn_initial_cubes(self):
        self.spawn_timer.cancel()
        n = 5

        if not self.spawn_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Spawn service unavailable")
            return

        sdf = """
        <?xml version='1.0'?><sdf version="1.4">
        <model name="cube">
        <static>0</static>
        <link name="link">
            <visual name="vis">
                <geometry><box><size>0.2 0.2 0.2</size></box></geometry>
                <material><script><uri>file://media/materials/scripts/gazebo.material</uri><name>Gazebo/Red</name></script></material>
            </visual>
        </link>
        </model></sdf>
        """

        for i in range(n):
            req = SpawnEntity.Request()
            req.name = f"cube_{i}"
            req.xml = sdf

            while True:
                x = random.uniform(-1.4, 1.4)
                y = random.uniform(-1.4, 1.4)
                if x*x + y*y > 0.7:
                    break

            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.3
            self.spawn_client.call_async(req)

        self.get_logger().info("Spawned cubes successfully")

    # -------------------------------------------------------------------------
    # MODEL STATES
    # -------------------------------------------------------------------------
    def model_states_callback(self, msg):
        self.latest_model_states = msg

    # -------------------------------------------------------------------------
    # LIDAR PROCESSING
    # -------------------------------------------------------------------------
    def scan_callback(self, msg):
        self.scan_ranges = list(msg.ranges)

    def get_lidar_sector(self, sector):
        if len(self.scan_ranges) == 0:
            return 99.0

        ranges = self.scan_ranges
        n = len(ranges)
        center = n // 2
        w = 30

        if sector == "front":
            idxs = range(center - w, center + w)
        elif sector == "left":
            idxs = range(center + n//4 - w, center + n//4 + w)
        elif sector == "right":
            idxs = range(center - n//4 - w, center - n//4 + w)
        else:
            return 99.0

        vals = []
        for i in idxs:
            d = ranges[i % n]
            if not math.isinf(d) and not math.isnan(d) and d > 0.05:
                vals.append(d)

        return min(vals) if vals else 99.0

    # -------------------------------------------------------------------------
    # CAMERA
    # -------------------------------------------------------------------------
    def image_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, np.array([0,70,50]), np.array([10,255,255]))
        mask2 = cv2.inRange(hsv, np.array([170,70,50]), np.array([180,255,255]))
        mask = cv2.bitwise_or(mask1, mask2)

        self.debug_mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))

        cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.angular_error = None

        if cnts:
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 120:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"]/M["m00"])
                    h, w, _ = img.shape
                    self.angular_error = (w/2) - cx
                    self.last_known_error = self.angular_error
                    self.last_sighting_time = time.time()

        self.debug_img_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))

    # -------------------------------------------------------------------------
    # ODOMETRY
    # -------------------------------------------------------------------------
    def odom_callback(self, msg):
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        yaw = math.atan2(2*(o.w*o.z + o.x*o.y), 1 - 2*(o.y*o.y + o.z*o.z))
        self.current_pose = (p.x, p.y, yaw)

    # -------------------------------------------------------------------------
    # GRAB SYSTEM
    # -------------------------------------------------------------------------
    def get_closest_cube_name(self):
        if self.latest_model_states is None:
            return None

        names = self.latest_model_states.name
        poses = self.latest_model_states.pose

        robot_pos = None
        for i, name in enumerate(names):
            if name == "waffle_pi":
                robot_pos = (poses[i].position.x, poses[i].position.y)
                break

        if robot_pos is None:
            return None

        rx, ry = robot_pos
        best = None
        best_d = 9

        for i, name in enumerate(names):
            if name.startswith("cube_"):
                cx = poses[i].position.x
                cy = poses[i].position.y
                d = math.dist((rx,ry),(cx,cy))
                if d < best_d:
                    best_d = d
                    best = name

        return best if best_d < 1.2 else None

    def delete_cube(self, name):
        req = DeleteEntity.Request()
        req.name = name
        self.delete_client.call_async(req)

    # -------------------------------------------------------------------------
    # MAIN CONTROL LOOP — THE REAL FIX
    # -------------------------------------------------------------------------
    def control_loop(self):
        if self.current_pose is None:
            return

        twist = Twist()

        df = self.get_lidar_sector("front")
        dl = self.get_lidar_sector("left")
        dr = self.get_lidar_sector("right")
        visible = self.angular_error is not None
        time_since = time.time() - self.last_sighting_time

        # ---------------------------------------------------------------------
        # 🔥 GLOBAL SAFETY (ONLY BRAKE, NEVER STOP MISSION)
        # ---------------------------------------------------------------------
        if df < 0.18:
            twist.linear.x = -0.05
            self.cmd_vel_pub.publish(twist)
            return

        # ---------------------------------------------------------------------
        # 🔥 GLOBAL ANTI-STUCK ESCAPE (REV + TURN)
        # ---------------------------------------------------------------------
        if df < 0.30 and not visible:
            self.get_logger().warn("STUCK DETECTED → ESCAPE!")

            # reverse
            t0 = time.time()
            while time.time() - t0 < 0.23:
                esc = Twist()
                esc.linear.x = -0.12
                self.cmd_vel_pub.publish(esc)

            # turn toward clearer direction
            turn_dir = 1 if dl > dr else -1
            t0 = time.time()
            while time.time() - t0 < 0.35:
                esc = Twist()
                esc.angular.z = 0.8 * turn_dir
                self.cmd_vel_pub.publish(esc)

            return

        # ---------------------------------------------------------------------
        # 🔥 STATE: SEARCH
        # ---------------------------------------------------------------------
        if self.state == "SEARCH":

            # Priority: go to APPROACH if target appears
            if visible:
                self.state = "APPROACH"
                return

            # Light obstacle avoidance (non-blocking)
            if df < 0.45:
                twist.angular.z = 0.5 if dl > dr else -0.5
            else:
                twist.linear.x = 0.25

        # ---------------------------------------------------------------------
        # 🔥 STATE: APPROACH (FOLLOW CENTROID)
        # ---------------------------------------------------------------------
        elif self.state == "APPROACH":

            if not visible:
                self.state = "SEARCH"
                return

            kp = 0.005
            twist.angular.z = kp * self.angular_error

            # avoid obstacle but keep approaching!
            if df < 0.35:
                twist.linear.x = 0.05
                twist.angular.z += 0.4 * (1 if self.angular_error > 0 else -1)
            else:
                twist.linear.x = 0.18

            if df < 0.40:
                self.state = "GRAB"

        # ---------------------------------------------------------------------
        # 🔥 STATE: GRAB
        # ---------------------------------------------------------------------
        elif self.state == "GRAB":
            tgt = self.get_closest_cube_name()
            if tgt:
                self.delete_cube(tgt)
                self.has_cube = True
                self.state = "RETURN"
            else:
                twist.linear.x = -0.12
                self.state = "SEARCH"

        # ---------------------------------------------------------------------
        # 🔥 STATE: RETURN
        # ---------------------------------------------------------------------
        elif self.state == "RETURN":
            x, y, yaw = self.current_pose
            dist = math.sqrt(x*x + y*y)

            # avoid obstacle
            if df < 0.3:
                twist.angular.z = 0.6 if dl > dr else -0.6
                self.cmd_vel_pub.publish(twist)
                return

            ang = math.atan2(-y, -x)
            diff = (ang - yaw + math.pi) % (2*math.pi) - math.pi

            twist.angular.z = 1.0 * diff
            if abs(diff) < 0.4:
                twist.linear.x = 0.18

            if dist < 0.35:
                self.state = "DROP"

        # ---------------------------------------------------------------------
        # 🔥 STATE: DROP
        # ---------------------------------------------------------------------
        elif self.state == "DROP":
            twist.linear.x = -0.2
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.8)
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
