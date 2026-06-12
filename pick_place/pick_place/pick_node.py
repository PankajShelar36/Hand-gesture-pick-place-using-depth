import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, Float64
from builtin_interfaces.msg import Duration as BuiltinDuration

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs

from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from moveit_msgs.msg import Constraints, JointConstraint

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration


# ============================================================
# Quaternion helper
# ============================================================
def quat_multiply(q1, q2):
    """Multiply two quaternions stored as (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def gripper_orientation_for_angle(angle_deg):
    """
    Return (qx, qy, qz, qw) for a downward-pointing gripper rotated
    by angle_deg around the tool approach axis (world -Z when pointing down).

    Base orientation – gripper pointing straight down:
        q_base = (w=0, x=1, y=0, z=0)   ← 180° around world X

    To spin the gripper around its OWN approach axis we POST-multiply:
        q_result = q_base * q_rot_z(angle)
    """
    angle_rad = math.radians(angle_deg)
    half = angle_rad / 2.0

    q_rot  = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base = (0.0, 1.0, 0.0, 0.0)

    w, x, y, z = quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ============================================================
# Tolerance constants
# ============================================================
# Loose tolerance for hover / lift motions — faster planning
LOOSE_TOLERANCE = 0.02

# Tight tolerance for plunge motions — ensures consistent Z depth
# This is the key fix: MoveIt must stop VERY close to the IK solution,
# otherwise small joint errors across 6 joints accumulate into 1-3 cm
# of Cartesian Z error at the end-effector.
TIGHT_TOLERANCE = 0.003


class PickNode(Node):
    def __init__(self):
        super().__init__('pick_node')

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.is_moving    = False
        self.motion_stage = 'IDLE'
        self.sequence_timer = None

        self.last_target_x = None
        self.last_target_y = None
        self.current_goal_x = None
        self.current_goal_y = None

        self.current_joint_map = {}
        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        self.retarget_threshold   = 0.05
        self.stable_wait_sec      = 3.0

        self.pending_target_x          = None
        self.pending_target_y          = None
        self.pending_target_start_time = None

        # ── Orientation state ─────────────────────────────────────────
        self.current_box_angle = 0.0
        self.goal_box_angle    = 0.0
        self.last_pick_angle   = None
        # ─────────────────────────────────────────────────────────────

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.subscription     = self.create_subscription(
            Point, '/red_box_ray', self.ray_callback, 10)
        self.joint_state_sub  = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 20)
        self.angle_sub        = self.create_subscription(
            Float64, '/red_box_angle', self.angle_callback, 10)

        self.gripper_pub = self.create_publisher(
            Int32MultiArray, '/gripper_control', 10)

        self.table_top_z    = 0.64
        self.hover_gap      = 0.15
        self.gripper_length = 0.27
        self.target_flange_z = self.table_top_z + self.hover_gap + self.gripper_length

        # How far below hover to descend when plunging (pick and place)
        self.plunge_depth = 0.08

        self.get_logger().info(
            "Continuous Pick Node Active (with orientation)! Waiting for target...")

    # ─────────────────────────────────────────────────────────────────
    def angle_callback(self, msg):
        self.current_box_angle = msg.data

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def distance_2d(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def have_all_joint_states(self):
        return all(name in self.current_joint_map for name in self.ur_joint_names)

    def control_gripper(self, pos, speed, force):
        msg = Int32MultiArray()
        msg.data = [pos, speed, force]
        self.gripper_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────
    def ray_callback(self, msg):
        if self.is_moving or not self.have_all_joint_states():
            return

        try:
            t = self.tf_buffer.lookup_transform(
                'base', 'oak_rgb_camera_optical_frame',
                rclpy.time.Time(), timeout=Duration(seconds=1.0))

            origin_cam = tf2_geometry_msgs.PointStamped()
            origin_cam.header.frame_id = 'oak_rgb_camera_optical_frame'
            ray_cam = tf2_geometry_msgs.PointStamped()
            ray_cam.header.frame_id = 'oak_rgb_camera_optical_frame'
            ray_cam.point = msg

            origin_base = tf2_geometry_msgs.do_transform_point(origin_cam, t)
            ray_base    = tf2_geometry_msgs.do_transform_point(ray_cam, t)

            dx = ray_base.point.x - origin_base.point.x
            dy = ray_base.point.y - origin_base.point.y
            dz = ray_base.point.z - origin_base.point.z

            if abs(dz) < 1e-6:
                return

            scale = (self.table_top_z - origin_base.point.z) / dz
            raw_x = origin_base.point.x + scale * dx
            raw_y = origin_base.point.y + scale * dy

            final_x = raw_x
            final_y = raw_y

            now_sec = time.time()

            # ── First ever pick ───────────────────────────────────────
            if self.last_target_x is None or self.last_target_y is None:
                self.current_goal_x = final_x
                self.current_goal_y = final_y
                self.goal_box_angle = self.current_box_angle
                self.get_logger().info(
                    f"First pick → angle frozen at {self.goal_box_angle:.0f} deg")
                self.is_moving = True
                self.start_hover_motion(final_x, final_y)
                return

            # ── Check if same spot as last drop ──────────────────────
            dist_from_last = self.distance_2d(
                final_x, final_y, self.last_target_x, self.last_target_y)

            if dist_from_last < self.retarget_threshold:
                angle_changed = (self.last_pick_angle is not None and
                                 self.current_box_angle != self.last_pick_angle)

                if not angle_changed:
                    self.pending_target_x = self.pending_target_y = \
                        self.pending_target_start_time = None
                    return

                if self.pending_target_x is None:
                    self.pending_target_x = final_x
                    self.pending_target_y = final_y
                    self.pending_target_start_time = now_sec
                    return

                dist_from_pending = self.distance_2d(
                    final_x, final_y,
                    self.pending_target_x, self.pending_target_y)
                if dist_from_pending >= self.retarget_threshold:
                    self.pending_target_x = final_x
                    self.pending_target_y = final_y
                    self.pending_target_start_time = now_sec
                    return

                if (now_sec - self.pending_target_start_time) < self.stable_wait_sec:
                    return

                self.get_logger().info(
                    f"Orientation changed {self.last_pick_angle:.0f}°→"
                    f"{self.current_box_angle:.0f}° at same position. Re-picking!")
                self.current_goal_x = self.pending_target_x
                self.current_goal_y = self.pending_target_y
                self.pending_target_x = self.pending_target_y = \
                    self.pending_target_start_time = None
                self.goal_box_angle = self.current_box_angle
                self.is_moving = True
                self.start_hover_motion(self.current_goal_x, self.current_goal_y)
                return

            # ── Start / reset pending timer ───────────────────────────
            if self.pending_target_x is None:
                self.pending_target_x = final_x
                self.pending_target_y = final_y
                self.pending_target_start_time = now_sec
                return

            dist_from_pending = self.distance_2d(
                final_x, final_y, self.pending_target_x, self.pending_target_y)
            if dist_from_pending >= self.retarget_threshold:
                self.pending_target_x = final_x
                self.pending_target_y = final_y
                self.pending_target_start_time = now_sec
                return

            if (now_sec - self.pending_target_start_time) < self.stable_wait_sec:
                return

            # ── Stable new target confirmed ───────────────────────────
            self.current_goal_x = self.pending_target_x
            self.current_goal_y = self.pending_target_y
            self.pending_target_x = self.pending_target_y = \
                self.pending_target_start_time = None

            self.goal_box_angle = self.current_box_angle
            self.get_logger().info(
                f"New target → angle frozen at {self.goal_box_angle:.0f} deg")

            self.is_moving = True
            self.start_hover_motion(self.current_goal_x, self.current_goal_y)

        except TransformException as ex:
            self.get_logger().info(f"TF Error: {ex}")

    # ─────────────────────────────────────────────────────────────────
    def start_hover_motion(self, x, y):
        self.motion_stage = 'HOVER'
        # Hover and lift use LOOSE tolerance — speed matters more than precision
        self.execute_ik_motion(x, y, self.target_flange_z, tight=False)

    # ─────────────────────────────────────────────────────────────────
    def execute_ik_motion(self, x, y, z, tight=False):
        """
        Send an IK request then plan to the joint goal.

        tight=False  → LOOSE_TOLERANCE (0.02 rad) for hover/lift — fast planning
        tight=True   → TIGHT_TOLERANCE (0.003 rad) for plunges — consistent Z depth

        WHY tight matters:
            MoveIt plans in joint space. With 6 joints each allowed ±0.02 rad of
            error, the cumulative Cartesian Z error at the tool tip can be 1-3 cm.
            Because every plunge starts from a slightly different lift pose, the
            planner picks different joint solutions each cycle — causing the robot
            to stop at a different depth each time even though the target Z is
            hardcoded.  Tightening to 0.003 rad eliminates this accumulation.
        """
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass

        qx, qy, qz, qw = gripper_orientation_for_angle(self.goal_box_angle)
        self.get_logger().info(
            f"IK z={z:.3f}  box_angle={self.goal_box_angle:.0f}°  "
            f"tight={tight}  q=({qx:.4f},{qy:.4f},{qz:.4f},{qw:.4f})")

        target_pose = PoseStamped()
        target_pose.header.frame_id = "base"
        target_pose.pose.position.x = x
        target_pose.pose.position.y = y
        target_pose.pose.position.z = z
        target_pose.pose.orientation.x = qx
        target_pose.pose.orientation.y = qy
        target_pose.pose.orientation.z = qz
        target_pose.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name    = "ur_manipulator"
        req.ik_request.ik_link_name  = "tool0"
        req.ik_request.pose_stamped  = target_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout       = BuiltinDuration(sec=1, nanosec=0)

        req.ik_request.robot_state.joint_state.name = self.ur_joint_names
        req.ik_request.robot_state.joint_state.position = [
            self.current_joint_map[n] for n in self.ur_joint_names]

        # Store tight flag so the IK callback can pass it to plan_to_joint_goal
        self._pending_tight = tight

        future = self.ik_client.call_async(req)
        future.add_done_callback(self.ik_callback)

    def ik_callback(self, future):
        response = future.result()
        if response is None or response.error_code.val != 1:
            self.get_logger().error("IK failed. Unlocking.")
            self.is_moving    = False
            self.motion_stage = 'IDLE'
            return

        ik_map = dict(zip(response.solution.joint_state.name,
                          response.solution.joint_state.position))
        try:
            goal_positions = [ik_map[n] for n in self.ur_joint_names]
            # Pass through the tight flag that was set before the IK call
            self.plan_to_joint_goal(goal_positions, tight=self._pending_tight)
        except KeyError:
            self.get_logger().error("IK solution missing joint")
            self.is_moving    = False
            self.motion_stage = 'IDLE'

    def plan_to_joint_goal(self, goal_positions, tight=False):
        """
        Plan to a joint goal.

        tight=True uses TIGHT_TOLERANCE (0.003 rad) so plunge moves always
        stop at the exact IK-solved joint configuration, giving repeatable
        Cartesian depth at the tool tip.

        tight=False uses LOOSE_TOLERANCE (0.02 rad) for faster planning on
        hover and lift moves where sub-mm precision is not needed.
        """
        tolerance = TIGHT_TOLERANCE if tight else LOOSE_TOLERANCE
        self.get_logger().info(
            f"Planning with tolerance={tolerance:.4f} rad  tight={tight}")

        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name            = "ur_manipulator"
        req.motion_plan_request.num_planning_attempts = 5 if tight else 3
        req.motion_plan_request.allowed_planning_time = 3.0 if tight else 2.5
        req.motion_plan_request.max_velocity_scaling_factor     = 0.08
        req.motion_plan_request.max_acceleration_scaling_factor = 0.08

        req.motion_plan_request.start_state.joint_state.name     = self.ur_joint_names
        req.motion_plan_request.start_state.joint_state.position = [
            self.current_joint_map[n] for n in self.ur_joint_names]

        c = Constraints()
        for jn, jp in zip(self.ur_joint_names, goal_positions):
            c.joint_constraints.append(
                JointConstraint(
                    joint_name=jn,
                    position=jp,
                    tolerance_above=tolerance,   # ← key fix
                    tolerance_below=tolerance,   # ← key fix
                    weight=1.0))
        req.motion_plan_request.goal_constraints.append(c)

        future = self.plan_client.call_async(req)
        future.add_done_callback(self.plan_callback)

    def plan_callback(self, future):
        response = future.result()
        if response is None or response.motion_plan_response.error_code.val != 1:
            self.get_logger().error("MoveIt planning failed! Unlocking.")
            self.is_moving    = False
            self.motion_stage = 'IDLE'
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = \
            response.motion_plan_response.trajectory.joint_trajectory

        self.traj_client.wait_for_server()
        f = self.traj_client.send_goal_async(goal_msg)
        f.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error("Trajectory rejected! Unlocking.")
            self.is_moving    = False
            self.motion_stage = 'IDLE'
            return
        gh.get_result_async().add_done_callback(self.get_result_callback)

    # ─────────────────────────────────────────────────────────────────
    def _clear_timer(self):
        if self.sequence_timer is not None:
            self.destroy_timer(self.sequence_timer)
            self.sequence_timer = None

    # ─────────────────────────────────────────────────────────────────
    def get_result_callback(self, future):
        if self.motion_stage == 'HOVER':
            self.get_logger().info("Hover complete. Opening gripper in 1 s...")
            self.sequence_timer = self.create_timer(1.0, self.step_open_and_plunge)

        elif self.motion_stage == 'PLUNGE':
            self.get_logger().info("Plunge complete. Closing gripper...")
            self.sequence_timer = self.create_timer(0.5, self.step_close_and_lift)

        elif self.motion_stage == 'LIFT':
            self.get_logger().info("Lift complete. Placing back in 1 s...")
            self.sequence_timer = self.create_timer(1.0, self.trigger_place_plunge)

        elif self.motion_stage == 'PLACE_PLUNGE':
            self.get_logger().info("Returned to pick position. Opening gripper...")
            self.sequence_timer = self.create_timer(0.5, self.step_open_and_retract)

        elif self.motion_stage == 'PLACE_LIFT':
            self.get_logger().info("Retract complete. Waiting 3 s for next target...")
            self.sequence_timer = self.create_timer(3.0, self.reset_for_next_pick)

    # ── Step methods ──────────────────────────────────────────────────

    def step_open_and_plunge(self):
        self._clear_timer()
        self.get_logger().info("Opening gripper...")
        self.control_gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self.trigger_plunge_motion)

    def trigger_plunge_motion(self):
        self._clear_timer()
        plunge_z = self.target_flange_z - self.plunge_depth
        self.get_logger().info(f"Plunging down to z={plunge_z:.4f} m (tight tolerance)...")
        self.motion_stage = 'PLUNGE'
        # tight=True → robot stops very close to the IK joint solution
        # so the actual Cartesian Z is repeatable within ~1 mm
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            plunge_z, tight=True)

    def step_close_and_lift(self):
        self._clear_timer()
        self.get_logger().info("Closing gripper...")
        self.control_gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self.trigger_lift_motion)

    def trigger_lift_motion(self):
        self._clear_timer()
        self.get_logger().info("Lifting to hover height...")
        self.motion_stage = 'LIFT'
        # tight=False → fast planning for lift, precision not critical
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            self.target_flange_z, tight=False)

    def trigger_place_plunge(self):
        self._clear_timer()
        plunge_z = self.target_flange_z - self.plunge_depth
        self.get_logger().info(f"Moving down to drop at z={plunge_z:.4f} m (tight tolerance)...")
        self.motion_stage = 'PLACE_PLUNGE'
        # tight=True → consistent drop height every cycle regardless of
        # which lift pose the robot is coming from
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            plunge_z, tight=True)

    def step_open_and_retract(self):
        self._clear_timer()
        self.get_logger().info("Dropping object...")
        self.control_gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self.trigger_place_lift)

    def trigger_place_lift(self):
        self._clear_timer()
        self.get_logger().info("Retracting up...")
        self.motion_stage = 'PLACE_LIFT'
        # tight=False → fast planning for retract
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            self.target_flange_z, tight=False)

    def reset_for_next_pick(self):
        self._clear_timer()
        self.get_logger().info("Ready for new target! Move the box.")
        self.last_target_x = self.current_goal_x
        self.last_target_y = self.current_goal_y
        self.last_pick_angle = self.goal_box_angle
        self.pending_target_x = self.pending_target_y = \
            self.pending_target_start_time = None
        self.is_moving    = False
        self.motion_stage = 'IDLE'


def main(args=None):
    rclpy.init(args=args)
    node = PickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
