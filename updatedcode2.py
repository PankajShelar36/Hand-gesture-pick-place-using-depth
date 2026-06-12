import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from pymodbus.client import ModbusTcpClient
import time

# --- CONFIGURATION ---
GRIPPER_IP = '192.168.1.105'
PORT = 502

class Robotiq3FAdvanced(Node):
    def __init__(self):
        super().__init__('robotiq_3f_advanced')
        self.get_logger().info(f'Connecting to {GRIPPER_IP}...')

        self.client = ModbusTcpClient(GRIPPER_IP, port=PORT)
        
        # 1. Activate
        if self.connect_and_activate():
            self.get_logger().info('>>> SYSTEM READY <<<')
            self.get_logger().info('Use: ros2 topic pub --once /gripper_control std_msgs/msg/Int32MultiArray "data: [255, 255, 50]"')
            self.get_logger().info('Format: [Position (0-255), Speed (0-255), Force (0-255)]')
        else:
            self.get_logger().error('Connection Failed. Will retry on command.')

        # 2. Status Monitor (Reads "Object Detected" signal)
        self.timer = self.create_timer(1.0, self.status_monitor)

        # 3. Control Subscriber
        self.create_subscription(Int32MultiArray, '/gripper_control', self.control_callback, 10)

    def connect_and_activate(self):
        try:
            self.client.connect()
            self.client.write_register(0, 0)
            time.sleep(0.5)
            self.client.write_register(0, 256) # Activate
            time.sleep(2.0) 
            return True
        except Exception as e:
            self.get_logger().error(f"Activation Error: {e}")
            return False

    def status_monitor(self):
        """Reads gripper status to see if we are holding an object"""
        try:
            # Read Input Register 0 (Status)
            rr = self.client.read_input_registers(0, 1)
            if rr.isError(): return

            # Decode gOBJ (Object Detection) - Bits 6-7 of Low Byte
            # Actually on 3F it's often Bits 0-1 of High Byte of Input Reg 0
            # Let's look at the raw value to be safe.
            raw = rr.registers[0]
            
            # gOBJ Mapping for 3F:
            # 0 = Moving
            # 1 = Stopped (Object Detected while Opening)
            # 2 = Stopped (Object Detected while Closing) <--- WE WANT THIS
            # 3 = At Requested Position (No object detected)
            
            # Shift logic depends on endianness. 
            # Usually gOBJ is bits 12-13 in the 16-bit word? No, that's 2F.
            # 3F Manual: gOBJ is Bit 6-7 of Byte 0 (Input Reg 0 Low Byte).
            
            gOBJ = (raw >> 6) & 0x03 

            if gOBJ == 0:
                pass # Moving...
            elif gOBJ == 1:
                self.get_logger().info("STATUS: Stopped (Opening Blocked)")
            elif gOBJ == 2:
                self.get_logger().info(">>> STATUS: OBJECT DETECTED! (Holding Sponge) <<<")
            elif gOBJ == 3:
                # self.get_logger().info("STATUS: Position Reached (No Object)")
                pass

        except:
            pass

    def control_callback(self, msg):
        """Expects [Position, Speed, Force]"""
        if len(msg.data) < 3:
            self.get_logger().error("Input must be [Pos, Speed, Force]")
            return

        # Clamp values 0-255
        Pos = max(0, min(255, msg.data[0]))
        Spd = max(0, min(255, msg.data[1]))
        For = max(0, min(255, msg.data[2]))

        self.get_logger().info(f"Target -> Pos:{Pos} Spd:{Spd} Force:{For}")

        # --- PACKING FOR BASIC MODE ---
        # Reg 0: Action (0x09) | Options (0x00) -> 0x0900 (2304)
        # Reg 1: Options2 (0x00) | Position (Pos) -> Pos
        # Reg 2: Speed (High Byte) | Force (Low Byte)
        
        reg0 = 2304
        reg1 = Pos
        reg2 = (Spd << 8) | For # Pack Speed and Force together

        payload = [
            reg0, 
            reg1, 
            reg2, 
            65535, 65535, 65535, 65535, 65280 # Ignored registers
        ]

        # --- SELF-HEALING SEND ---
        for attempt in range(3):
            try:
                result = self.client.write_registers(0, payload)
                if not result.isError():
                    break 
            except:
                self.get_logger().warn("Reconnecting...")
                self.client.close()
                self.client.connect()

def main(args=None):
    rclpy.init(args=args)
    node = Robotiq3FAdvanced()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
