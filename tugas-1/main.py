from typing import Optional
from pybricks.hubs import InventorHub
from pybricks.robotics import DriveBase
from pybricks.pupdevices import Motor, ColorSensor, UltrasonicSensor
from pybricks.tools import StopWatch, wait
from pybricks.parameters import Port, Color, Stop, Direction
from ucollections import deque  # type: ignore


class RobotConfig:
    def __init__(
        self,
        wheel_diameter=56,
        axle_track=112,
        drive_velocity=200,
        drive_acceleration=20,
        turn_velocity=100,
        turn_acceleration=10,
        turn_left_limit=-55,
        turn_right_limit=55,
        yaw_velocity=300,
        yaw_left_limit=-115,
        yaw_right_limit=-55,
        pitch_velocity=200,
        pitch_up_limit=25,
        pitch_down_limit=-25,
    ):
        self.wheel_diameter = wheel_diameter
        self.axle_track = axle_track

        self.drive_velocity = drive_velocity
        self.drive_acceleration = drive_acceleration
        self.turn_velocity = turn_velocity
        self.turn_acceleration = turn_acceleration

        self.turn_left_limit = turn_left_limit
        self.turn_right_limit = turn_right_limit

        self.yaw_velocity = yaw_velocity
        self.yaw_left_limit = yaw_left_limit
        self.yaw_right_limit = yaw_right_limit

        self.pitch_velocity = pitch_velocity
        self.pitch_up_limit = pitch_up_limit
        self.pitch_down_limit = pitch_down_limit


class Robot(DriveBase):
    def __init__(
        self,
        hub: InventorHub,
        left_motor: Motor,
        right_motor: Motor,
        pitch_motor: Motor,
        yaw_motor: Motor,
        color_sensor: ColorSensor,
        ultrasonic_sensor: UltrasonicSensor,
        config: RobotConfig = RobotConfig(),
    ):
        self.config = config

        super().__init__(
            left_motor,
            right_motor,
            self.config.wheel_diameter,
            self.config.axle_track,
        )

        self.hub = hub

        self.pitch_motor = pitch_motor
        self.yaw_motor = yaw_motor
        self.color_sensor = color_sensor
        self.ultrasonic_sensor = ultrasonic_sensor

        self.step_state = "follow"
        self.prev_step_state = self.step_state

        self.stopwatch = StopWatch()

        self.t = 0.0
        self.step_state_time = 0.0

        self.initial = True

        self.lost_look_side = "right"
        self.lost_new_side = True

        self.move_hist = deque(maxlen=20)
        self.sensor_rec = []

        self.settings(straight_speed=self.config.drive_velocity)
        self.settings(turn_rate=self.config.turn_velocity)

    def straight(self, distance: float, then: Stop = Stop.HOLD, wait: bool = True):
        """
        Drives straight for a given distance and then stops.

        Arguments:
            distance (Number, mm): Distance to travel.
            then (Stop): What to do after coming to a standstill.
            wait (bool): Wait for the maneuver to complete before continuing.
        """
        log_entry = {"action": "straight", "value": distance}
        self.move_hist.append(log_entry)
        super().straight(distance, then=then, wait=wait)

    def scan_for_line(self, target_color=Color.GREEN) -> list[int]:
        """
        Sweeps the yaw motor from left to right and records all angles
        where the target color is detected. Returns a list of angles.
        """
        print("Scanning for line...")
        found_angles = []

        self.yaw_motor.run_target(
            self.config.yaw_velocity, self.config.yaw_right_limit, wait=False
        )

        while not self.yaw_motor.done():
            if self.color_sensor.color() == target_color:
                angle = self.yaw_motor.angle()
                if (
                    not found_angles or abs(angle - found_angles[-1]) > 5
                ):  # 5-degree buffer
                    print(f"  > Found line at angle: {angle}")
                    found_angles.append(angle)
            wait(10)

        self.yaw_motor.run_target(self.config.yaw_velocity, self.config.yaw_left_limit)

        return found_angles

    def turn(self, angle: float, then: Stop = Stop.HOLD, wait: bool = True):
        """
        Turns in place by a given angle and then stops.

        Arguments:
            angle (Number, deg): Angle of the turn.
            then (Stop): What to do after coming to a standstill.
            wait (bool): Wait for the maneuver to complete before continuing.
        """
        log_entry = {"action": "turn", "value": angle}
        self.move_hist.append(log_entry)
        super().turn(angle, then=then, wait=wait)

    def drive(self, speed: float, turn_rate: float):
        """
        Starts driving at the specified speed and turn rate. Both values are measured at the center point between the wheels of the robot.

        Arguments:
            speed (Number, mm/s): Speed of the robot.
            turn_rate (Number, deg/s): Turn rate of the robot.
        """
        log_entry = {"action": "drive", "value": (speed, turn_rate)}
        self.move_hist.append(log_entry)
        super().drive(speed, turn_rate)

    def sensor_move_to_target(
        self, yaw_target: Optional[float] = None, pitch_target: Optional[float] = None
    ):
        if yaw_target is not None:
            if yaw_target < self.config.yaw_left_limit:
                yaw_target = self.config.yaw_left_limit
            if yaw_target > self.config.yaw_right_limit:
                yaw_target = self.config.yaw_right_limit

            log_entry = {
                "timestamp": self.t,
                "action": "sensor_yaw",
                "target": yaw_target,
            }
            self.move_hist.append(log_entry)
            self.yaw_motor.run_target(self.config.yaw_velocity, yaw_target)

        if pitch_target is not None:
            if pitch_target < self.config.pitch_down_limit:
                pitch_target = self.config.pitch_down_limit
            if pitch_target > self.config.pitch_up_limit:
                pitch_target = self.config.pitch_up_limit

            log_entry = {
                "timestamp": self.t,
                "action": "sensor_pitch",
                "target": pitch_target,
            }
            self.move_hist.append(log_entry)
            self.pitch_motor.run_target(self.config.yaw_velocity, pitch_target)

    def process_scan(self):
        if self.lost_look_side == "right":
            if self.lost_new_side:
                self.sensor_move_to_target(yaw_target=self.config.yaw_right_limit)
                self.lost_new_side = False

            if self.yaw_motor.done():
                self.lost_look_side = "left"
                self.lost_new_side = True

        if self.lost_look_side == "left":
            if self.lost_new_side:
                self.sensor_move_to_target(yaw_target=self.config.yaw_left_limit)
                self.lost_new_side = False

            if self.yaw_motor.done():
                self.lost_look_side = "right"
                self.lost_new_side = True

    def step_follow(self):
        distance_step = 50
        self.straight(distance_step)

        if self.color_sensor.color() != Color.GREEN:
            self.step_state = "backtrack"
            self.stop()

    def step_lost(self):
        if self.initial:
            print("Oh no, I'm lost! Backing up...")
            self.straight(-50)
            self.initial = False

        self.process_scan()

    def step_backtrack(self):
        if not self.move_hist:
            print("Backtracking failed, history is empty! Switching to wide scan.")
            self.step_state = "lost"  # Fall back to the old scan-in-place
            return

        # Pop the last move we made
        last_move = self.move_hist.pop()
        print(f"Backtracking move: {last_move}")

        # Invert the action and execute it
        if last_move["action"] == "straight":
            # Move backwards by the same amount
            self.straight(-last_move["value"])
        elif last_move["action"] == "turn":
            # Turn in the opposite direction
            self.turn(-last_move["value"])

    def step(self):
        dt = self.stopwatch.time() / 1000
        self.stopwatch.reset()
        self.t += dt
        self.step_state_time += dt

        if self.prev_step_state != self.step_state:
            self.initial = True
            self.step_state_time = 0.0
            self.prev_step_state = self.step_state

        if self.step_state == "follow":
            self.step_follow()
        elif self.step_state == "backtrack":
            self.step_backtrack()
        elif self.step_state == "lost":
            self.step_lost()

        if self.move_hist:
            print(f"Time: {self.t:.2f}, Last Move: {self.move_hist[-1]}")


robot_config = RobotConfig()
hub = InventorHub()
left_motor = Motor(Port.F, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.B)
pitch_motor = Motor(Port.C)
yaw_motor = Motor(Port.D)
color_sensor = ColorSensor(Port.E)
ultrasonic_sensor = UltrasonicSensor(Port.A)

robot = Robot(
    hub,
    left_motor,
    right_motor,
    pitch_motor,
    yaw_motor,
    color_sensor,
    ultrasonic_sensor,
    robot_config,
)

while True:
    robot.step()

    wait(80)
