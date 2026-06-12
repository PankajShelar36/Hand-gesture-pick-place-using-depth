import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
import time
import math 

class SceneSpawner(Node):
    def __init__(self):
        super().__init__('scene_spawner')
        # Publisher for static objects (floor, table)
        self.publisher_ = self.create_publisher(CollisionObject, '/collision_object', 10)
        
        # Publisher for attached objects (gripper)
        self.attached_publisher_ = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        
        # Give the publishers a second to connect to the MoveIt network
        time.sleep(1.0)
        self.spawn_objects()

    def spawn_objects(self):
        # ==========================================
        # 1. THE FLOOR
        # ==========================================
        floor = CollisionObject()
        floor.header.frame_id = "base"
        floor.id = "safety_floor"
        floor.operation = CollisionObject.ADD

        floor_box = SolidPrimitive()
        floor_box.type = SolidPrimitive.BOX
        floor_box.dimensions = [4.0, 4.0, 0.1] # 4m x 4m wide, 10cm thick

        floor_pose = Pose()
        floor_pose.position.x = 0.0
        floor_pose.position.y = 0.0
        floor_pose.position.z = -0.10 

        floor.primitives.append(floor_box)
        floor.primitive_poses.append(floor_pose)

        # ==========================================
        # 2. THE TABLE
        # ==========================================
        table = CollisionObject()
        table.header.frame_id = "base"
        table.id = "work_table"
        
        table_box = SolidPrimitive()
        table_box.type = SolidPrimitive.BOX
        table_box.dimensions = [0.39, 0.92, 0.69] 

        table_pose = Pose()
        table_pose.position.x = 0.0 
        table_pose.position.y = -0.65 
        table_pose.position.z = 0.295 
        
        yaw = math.pi / 2.0
        table_pose.orientation.x = 0.0
        table_pose.orientation.y = 0.0
        table_pose.orientation.z = math.sin(yaw / 2.0)
        table_pose.orientation.w = math.cos(yaw / 2.0)

        table.primitives.append(table_box)
        table.primitive_poses.append(table_pose)
        table.operation = CollisionObject.ADD

        # ==========================================
        # 3. THE GRIPPER (ATTACHED)
        # ==========================================
        aco = AttachedCollisionObject()
        aco.link_name = "tool0" # The flange of the UR robot
        
        gripper_obj = CollisionObject()
        gripper_obj.header.frame_id = "tool0"
        gripper_obj.id = "robotiq_3f_gripper"
        gripper_obj.operation = CollisionObject.ADD

        gripper_box = SolidPrimitive()
        gripper_box.type = SolidPrimitive.BOX
        gripper_box.dimensions = [0.15, 0.15, 0.27] # 12cm x 12cm x 27cm

        gripper_pose = Pose()
        gripper_pose.position.x = 0.0
        gripper_pose.position.y = 0.0
        gripper_pose.position.z = 0.135 # Half of 27cm so it sits flush on the flange
        gripper_pose.orientation.w = 1.0

        gripper_obj.primitives.append(gripper_box)
        gripper_obj.primitive_poses.append(gripper_pose)
        
        aco.object = gripper_obj

        # ==========================================
        # PUBLISH EVERYTHING
        # ==========================================
        self.publisher_.publish(floor)
        self.publisher_.publish(table)
        self.attached_publisher_.publish(aco)
        
        self.get_logger().info("Scene spawned: Floor, Table, and Gripper attached!")

def main(args=None):
    rclpy.init(args=args)
    node = SceneSpawner()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
