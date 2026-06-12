#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

class GripperTestNode(Node):
    def __init__(self):
        super().__init__('gripper_test_node')
        self.gripper_pub = self.create_publisher(Int32MultiArray, '/gripper_control', 10)
        self.get_logger().info("Gripper Test Node Started.")

    def send_command(self, pos, speed, force):
        msg = Int32MultiArray()
        msg.data = [pos, speed, force]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f"Sent command: pos={pos}, speed={speed}, force={force}")

def main(args=None):
    rclpy.init(args=args)
    node = GripperTestNode()

    print("\n" + "="*30)
    print("   GRIPPER TEST INTERFACE   ")
    print("="*30)
    print(" o : Open Gripper  (pos = 0)")
    print(" c : Close Gripper (pos = 255)")
    print(" q : Quit")
    print("="*30 + "\n")

    try:
        while rclpy.ok():
            # Blocks and waits for user input
            cmd = input("Enter command (o / c / q): ").strip().lower()
            
            if cmd == 'o':
                node.send_command(pos=0, speed=150, force=100)
            elif cmd == 'c':
                node.send_command(pos=255, speed=150, force=20)
            elif cmd == 'q':
                print("Exiting test interface...")
                break
            else:
                print("Invalid command. Please enter 'o', 'c', or 'q'.")
                
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
