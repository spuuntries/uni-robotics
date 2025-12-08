import random
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

class CubeSpawner(Node):
    def __init__(self):
        super().__init__('cube_spawner')
        self.client = self.create_client(SpawnEntity, '/spawn_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.get_logger().info('Spawn Entity Service Available')

    def spawn_cubes(self, n=5):
        # Red Cube SDF
        sdf = """
        <?xml version='1.0'?>
        <sdf version="1.4">
            <model name="my_cube">
                <static>0</static>
                <link name="link">
                    <inertial>
                        <mass>0.1</mass>
                        <inertia>
                            <ixx>0.0001</ixx>
                            <ixy>0</ixy>
                            <ixz>0</ixz>
                            <iyy>0.0001</iyy>
                            <iyz>0</iyz>
                            <izz>0.0001</izz>
                        </inertia>
                    </inertial>
                    <collision name="collision">
                        <geometry>
                            <box>
                                <size>0.2 0.2 0.2</size>
                            </box>
                        </geometry>
                    </collision>
                    <visual name="visual">
                        <geometry>
                            <box>
                                <size>0.2 0.2 0.2</size>
                            </box>
                        </geometry>
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
            request = SpawnEntity.Request()
            request.name = f'cube_{i}'
            request.xml = sdf
            
            # Spawn in a donut shape or safe box avoiding (0,0)
            while True:
                rx = random.uniform(-1.5, 1.5)
                ry = random.uniform(-1.5, 1.5)
                if (rx**2 + ry**2) > 0.5**2: # Ensure it's at least 0.5m away from center
                    break
            
            request.initial_pose.position.x = rx
            request.initial_pose.position.y = ry
            request.initial_pose.position.z = 0.5
            
            self.future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, self.future)
            if self.future.result() is not None:
                self.get_logger().info(f'Spawned {request.name}')
            else:
                self.get_logger().error(f'Failed to spawn {request.name}')

def main(args=None):
    rclpy.init(args=args)
    node = CubeSpawner()
    node.spawn_cubes(5) # Spawn 5 cubes
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
