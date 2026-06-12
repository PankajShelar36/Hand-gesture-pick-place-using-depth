import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, Float64, Bool
from builtin_interfaces.msg import Duration as BuiltinDuration

from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from moveit_msgs.msg import Constraints, JointConstraint

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration


# ─────────────────────────────────────────────────────────────────────────────
# Home / observation position  (teach-pendant degrees → radians)
# ─────────────────────────────────────────────────────────────────────────────
HOME_JOINTS = [math.radians(d) for d in [77.03, -88.01, 63.03, -63.82, -88.73, 266.75]]
#   shoulder_pan=77.03°  shoulder_lift=-88.01°  elbow=63.03°
#   wrist_1=-63.82°      wrist_2=-88.73°        wrist_3=266.75°

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants  (metres)
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_LENGTH      = 0.27    # tool0 flange → fingertip
HOVER_GAP           = 0.15    # clearance above object top before descending
GRASP_DEPTH         = 0.05    # how far below object top the fingers should be when gripping
GRIPPER_ANGLE_OFFSET = -90.0  # PCA angle is from image-X; gripper zero is 90° offset from that

# ─────────────────────────────────────────────────────────────────────────────
# MoveIt planning tolerances
# ─────────────────────────────────────────────────────────────────────────────
LOOSE_TOLERANCE = 0.02    # hover / lift — fast planning
TIGHT_TOLERANCE = 0.003   # plunge only  — repeatable Z depth

# ─────────────────────────────────────────────────────────────────────────────
# Detection stability gate
# ─────────────────────────────────────────────────────────────────────────────
STABLE_SEC      = 3.0    # how long the detection must be spatially consistent
RETARGET_DIST_M = 0.05   # XY move > 5 cm resets the stability timer


# ─────────────────────────────────────────────────────────────────────────────
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def gripper_quat(angle_deg=0.0):
    """(qx, qy, qz, qw) for gripper pointing straight down, rotated angle_deg around Z."""
    half   = math.radians(angle_deg) / 2.0
    q_rot  = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base = (0.0, 1.0, 0.0, 0.0)   # 180° around world X = pointing down
    w, x, y, z = quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ─────────────────────────────────────────────────────────────────────────────
class PickNodeAnyPose(Node):
    """
    Pick-and-place node for any-object pose-based detection.

    Subscribes to:
        /object_pose   (PoseStamped, frame=base) — from any_pose_detector_rs
        /object_angle  (Float64, degrees 0-180)  — PCA-derived exact angle
        /joint_states  (JointState)

    Publishes:
        /gripper_control (Int32MultiArray) → updatedcode2.py

    Grasp heights (base frame, Z axis up):
        hover_flange_z  = obj_z + GRIPPER_LENGTH + HOVER_GAP
        plunge_flange_z = obj_z + GRIPPER_LENGTH - GRASP_DEPTH   ← FULLY DYNAMIC

    State machine: IDLE → HOVER → PLUNGE → LIFT → PLACE_PLUNGE → PLACE_LIFT → IDLE
    Place returns the object to the same (x, y, z) it was picked from.
    """

    def __init__(self):
        super().__init__('pick_node_any_pose')

        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        self.current_joint_map = {}

        self.is_moving      = True    # blocked until startup home move completes
        self.motion_stage   = 'IDLE'
        self.sequence_timer = None
        self._pending_tight = False

        self.goal_x     = None
        self.goal_y     = None
        self.goal_z     = None
        self.goal_angle = 0.0

        self.hover_z  = None
        self.plunge_z = None

        self.last_x = None
        self.last_y = None
        self.pending_x     = None
        self.pending_y     = None
        self.pending_z     = None
        self.pending_start = None

        self._latest_x     = None
        self._latest_y     = None
        self._latest_z     = None
        self._latest_angle = 0.0

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.create_subscription(
            PoseStamped, '/object_pose',  self._pose_cb,  10)
        self.create_subscription(
            Float64,     '/object_angle', self._angle_cb, 10)
        self.create_subscription(
            JointState,  '/joint_states', self._joint_cb, 20)

        self.gripper_pub      = self.create_publisher(Int32MultiArray, '/gripper_control', 10)
        self.pick_active_pub  = self.create_publisher(Bool, '/pick_active', 10)

        # Move to home on startup; 3 s delay lets MoveIt finish loading
        self._startup_timer = self.create_timer(3.0, self._startup_home_cb)

        self.get_logger().info(
            f'Pick Node Any-Pose ready  '
            f'(hover_gap={HOVER_GAP} m  grasp_depth={GRASP_DEPTH} m  stable={STABLE_SEC} s)  '
            f'-- moving to home position before detection starts')

    # ─────────────────────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def _angle_cb(self, msg):
        self._latest_angle = msg.data

    def _have_joints(self):
        return all(n in self.current_joint_map for n in self.ur_joint_names)

    def _gripper(self, pos, speed=150, force=100):
        msg = Int32MultiArray()
        msg.data = [pos, speed, force]
        self.gripper_pub.publish(msg)

    def _set_pick_active(self, active: bool):
        msg = Bool()
        msg.data = active
        self.pick_active_pub.publish(msg)

    def _clear_timer(self):
        if self.sequence_timer is not None:
            self.destroy_timer(self.sequence_timer)
            self.sequence_timer = None

    # ─────────────────────────────────────────────────────────────────────────
    def _startup_home_cb(self):
        """One-shot timer: move to home before allowing any detection."""
        self.destroy_timer(self._startup_timer)

        if not self._have_joints():
            # Joints not yet received — retry in 1 s
            self._startup_timer = self.create_timer(1.0, self._startup_home_cb)
            self.get_logger().info('Waiting for joint states before homing...')
            return

        self._set_pick_active(True)
        self.get_logger().info('Moving to home position...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False)

    # ─────────────────────────────────────────────────────────────────────────
    def _pose_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        self._latest_x = x
        self._latest_y = y
        self._latest_z = z

        if self.is_moving or not self._have_joints():
            return

        now = time.time()

        if self.last_x is None:
            if self.pending_x is None:
                self.pending_x     = x
                self.pending_y     = y
                self.pending_z     = z
                self.pending_start = now
                return
            if self._dist2d(x, y, self.pending_x, self.pending_y) >= RETARGET_DIST_M:
                self.pending_x, self.pending_y = x, y
                self.pending_z = z
                self.pending_start = now
                return
            if (now - self.pending_start) >= STABLE_SEC:
                self._commit_pick(self.pending_x, self.pending_y, self.pending_z)
            return

        if self._dist2d(x, y, self.last_x, self.last_y) < RETARGET_DIST_M:
            self.pending_x = self.pending_y = self.pending_z = self.pending_start = None
            return

        if self.pending_x is None:
            self.pending_x, self.pending_y = x, y
            self.pending_z = z
            self.pending_start = now
            return

        if self._dist2d(x, y, self.pending_x, self.pending_y) >= RETARGET_DIST_M:
            self.pending_x, self.pending_y = x, y
            self.pending_z = z
            self.pending_start = now
            return

        if (now - self.pending_start) >= STABLE_SEC:
            self._commit_pick(self.pending_x, self.pending_y, self.pending_z)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _dist2d(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # ─────────────────────────────────────────────────────────────────────────
    def _commit_pick(self, x, y, z):
        self.goal_x     = x
        self.goal_y     = y
        self.goal_z     = z
        self.goal_angle = (self._latest_angle + GRIPPER_ANGLE_OFFSET) % 180.0

        self.hover_z  = z + GRIPPER_LENGTH + HOVER_GAP
        self.plunge_z = z + GRIPPER_LENGTH - GRASP_DEPTH

        self.pending_x = self.pending_y = self.pending_z = self.pending_start = None
        self._set_pick_active(True)

        self.get_logger().info(
            f'Pick committed  obj=({x:.3f}, {y:.3f}, {z:.3f})  '
            f'hover_flange_z={self.hover_z:.3f}  plunge_flange_z={self.plunge_z:.3f}  '
            f'angle={self.goal_angle:.1f} deg')

        self.is_moving = True
        self._go_hover()

    # ─────────────────────────────────────────────────────────────────────────
    def _ik_move(self, x, y, z, tight=False):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass

        qx, qy, qz, qw = gripper_quat(self.goal_angle)

        target = PoseStamped()
        target.header.frame_id = 'base'
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name       = 'ur_manipulator'
        req.ik_request.ik_link_name     = 'tool0'
        req.ik_request.pose_stamped     = target
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout          = BuiltinDuration(sec=1, nanosec=0)
        req.ik_request.robot_state.joint_state.name     = self.ur_joint_names
        req.ik_request.robot_state.joint_state.position = [
            self.current_joint_map[n] for n in self.ur_joint_names]

        self._pending_tight = tight
        self.ik_client.call_async(req).add_done_callback(self._ik_cb)

    def _ik_cb(self, future):
        resp = future.result()
        if resp is None or resp.error_code.val != 1:
            self.get_logger().error(f'IK failed at stage {self.motion_stage}. Unlocking.')
            self._unlock()
            return
        ik_map = dict(zip(resp.solution.joint_state.name,
                          resp.solution.joint_state.position))
        try:
            joints = [ik_map[n] for n in self.ur_joint_names]
            self._plan_joints(joints, tight=self._pending_tight)
        except KeyError:
            self.get_logger().error('IK solution missing joint. Unlocking.')
            self._unlock()

    def _plan_joints(self, goal_positions, tight=False):
        tol = TIGHT_TOLERANCE if tight else LOOSE_TOLERANCE
        req = GetMotionPlan.Request()
        mp  = req.motion_plan_request
        mp.group_name            = 'ur_manipulator'
        mp.num_planning_attempts = 5 if tight else 3
        mp.allowed_planning_time = 3.0 if tight else 2.5
        mp.max_velocity_scaling_factor     = 0.08
        mp.max_acceleration_scaling_factor = 0.08
        mp.start_state.joint_state.name     = self.ur_joint_names
        mp.start_state.joint_state.position = [
            self.current_joint_map[n] for n in self.ur_joint_names]

        c = Constraints()
        for jn, jp in zip(self.ur_joint_names, goal_positions):
            c.joint_constraints.append(
                JointConstraint(joint_name=jn, position=jp,
                                tolerance_above=tol, tolerance_below=tol, weight=1.0))
        mp.goal_constraints.append(c)

        self.plan_client.call_async(req).add_done_callback(self._plan_cb)

    def _plan_cb(self, future):
        resp = future.result()
        if resp is None or resp.motion_plan_response.error_code.val != 1:
            self.get_logger().error(f'Planning failed at {self.motion_stage}. Unlocking.')
            self._unlock()
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = resp.motion_plan_response.trajectory.joint_trajectory
        self.traj_client.wait_for_server()
        self.traj_client.send_goal_async(goal).add_done_callback(self._goal_resp_cb)

    def _goal_resp_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(f'Trajectory rejected at {self.motion_stage}. Unlocking.')
            self._unlock()
            return
        gh.get_result_async().add_done_callback(self._result_cb)

    def _unlock(self):
        self.is_moving    = False
        self.motion_stage = 'IDLE'
        self._set_pick_active(False)

    # ─────────────────────────────────────────────────────────────────────────
    def _result_cb(self, future):
        if   self.motion_stage == 'HOME':
            self.get_logger().info('At home position. Detection active.')
            self._unlock()   # sets is_moving=False and pick_active=False
        elif self.motion_stage == 'HOVER':
            self.get_logger().info('Hover reached. Opening gripper...')
            self.sequence_timer = self.create_timer(0.8, self._open_then_plunge)
        elif self.motion_stage == 'PLUNGE':
            self.get_logger().info('Plunge done. Closing gripper...')
            self.sequence_timer = self.create_timer(0.5, self._close_then_lift)
        elif self.motion_stage == 'LIFT':
            self.get_logger().info('Lift done. Moving to place position...')
            self.sequence_timer = self.create_timer(0.8, self._go_place_plunge)
        elif self.motion_stage == 'PLACE_PLUNGE':
            self.get_logger().info('At place depth. Opening gripper...')
            self.sequence_timer = self.create_timer(0.5, self._open_then_retract)
        elif self.motion_stage == 'PLACE_LIFT':
            self.get_logger().info('Retracted. Returning to home...')
            self.sequence_timer = self.create_timer(1.0, self._finish_cycle)

    # ── Steps ─────────────────────────────────────────────────────────────────
    def _go_hover(self):
        self.get_logger().info(f'Moving to hover  z={self.hover_z:.3f} m ...')
        self.motion_stage = 'HOVER'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _open_then_plunge(self):
        self._clear_timer()
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._do_plunge)

    def _do_plunge(self):
        self._clear_timer()
        self.get_logger().info(
            f'Plunging to z={self.plunge_z:.3f} m  '
            f'(obj_z={self.goal_z:.3f} + gripper={GRIPPER_LENGTH} - grasp_depth={GRASP_DEPTH})  tight...')
        self.motion_stage = 'PLUNGE'
        self._ik_move(self.goal_x, self.goal_y, self.plunge_z, tight=True)

    def _close_then_lift(self):
        self._clear_timer()
        self._gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self._do_lift)

    def _do_lift(self):
        self._clear_timer()
        self.get_logger().info('Lifting to hover height...')
        self.motion_stage = 'LIFT'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _go_place_plunge(self):
        self._clear_timer()
        self.get_logger().info(f'Descending to place z={self.plunge_z:.3f} m (tight)...')
        self.motion_stage = 'PLACE_PLUNGE'
        self._ik_move(self.goal_x, self.goal_y, self.plunge_z, tight=True)

    def _open_then_retract(self):
        self._clear_timer()
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._do_retract)

    def _do_retract(self):
        self._clear_timer()
        self.get_logger().info('Retracting to hover height...')
        self.motion_stage = 'PLACE_LIFT'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _finish_cycle(self):
        self._clear_timer()
        self.last_x = self.goal_x
        self.last_y = self.goal_y
        self.get_logger().info('Returning to home position...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False)
        # is_moving stays True, pick_active stays True until HOME motion completes


def main(args=None):
    rclpy.init(args=args)
    node = PickNodeAnyPose()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
