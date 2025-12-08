#!/bin/bash
set -e

# 0. Cleanup existing Gazebo processes to ensure a clean world
echo "Cleaning up existing Gazebo processes..."
killall -9 gzserver gzclient 2>/dev/null || true

# 1. Source ROS 2 (Humble detected from environment, defaulting to typical path if not set)
if [ -z "$ROS_DISTRO" ]; then
    export ROS_DISTRO=humble
fi

# Prevent Python from looking in ~/.local/lib/python
export PYTHONNOUSERSITE=1

if [ -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    source "/opt/ros/$ROS_DISTRO/setup.bash"
    echo "Sourced ROS 2 $ROS_DISTRO"
else
    echo "Warning: /opt/ros/$ROS_DISTRO/setup.bash not found. Ensure ROS 2 is installed and sourced."
fi

# 2. Activate Virtual Environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated virtual environment."
fi

# 3. Resolve dependencies
# rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y

# 4. Build the package
echo "Building package..."
colcon build

# 5. Source the local overlay
if [ -f "install/setup.bash" ]; then
    source install/setup.bash
    echo "Sourced local workspace."
else
    echo "Error: install/setup.bash not found!"
    exit 1
fi

# 6. Launch TurtleBot3 Simulation (Background)
if [ -z "$TURTLEBOT3_MODEL" ]; then
    export TURTLEBOT3_MODEL=waffle_pi
fi
echo "Launching TurtleBot3 ($TURTLEBOT3_MODEL) simulation..."
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py &
SIM_PID=$!
sleep 8 # Give Gazebo time to start

# Retry spawn logic (in case the launch file's internal spawn failed)
echo "Ensuring robot is spawned (retrying if needed)..."
for i in {1..3}; do
    ros2 run gazebo_ros spawn_entity.py \
        -entity waffle_pi \
        -file /opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf \
        -x -2.0 -y -0.5 -z 0.01 && break
    echo "Spawn attempt $i failed or entity already exists. Retrying in 5s..."
    sleep 5
done || true

# 7. Spawn Cubes
echo "Spawning Cubes..."
ros2 run my_ball_tracker cube_spawner

# 8. Run the collector node
echo "Running cube_collector..."
ros2 run my_ball_tracker cube_collector

# Cleanup background process on exit
kill $SIM_PID
