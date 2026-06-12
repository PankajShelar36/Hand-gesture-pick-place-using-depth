import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Int32MultiArray, Float64, Bool
from builtin_interfaces.msg import Duration as BuiltinDuration
from cv_bridge import CvBridge

from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from moveit_msgs.msg import Constraints, JointConstraint

from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

# ─────────────────────────────────────────────────────────────────────────────
# Poses  (teach-pendant degrees → radians)
# ─────────────────────────────────────────────────────────────────────────────
HOME_JOINTS     = [math.radians(d) for d in [77.03, -88.01, 63.03, -63.82, -88.73, 266.75]]
HANDOVER_JOINTS = [math.radians(d) for d in [78.85, -89.08, 65.74, -76.01,   2.16, 277.87]]

# ─────────────────────────────────────────────────────────────────────────────
# Geometry  (metres)
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_LENGTH       = 0.27
HOVER_GAP            = 0.15
GRASP_DEPTH          = 0.05
GRIPPER_ANGLE_OFFSET = -90.0

# ─────────────────────────────────────────────────────────────────────────────
# MoveIt tolerances
# ─────────────────────────────────────────────────────────────────────────────
LOOSE_TOLERANCE = 0.02
TIGHT_TOLERANCE = 0.003

# ─────────────────────────────────────────────────────────────────────────────
# Object detection stability gate
# ─────────────────────────────────────────────────────────────────────────────
STABLE_SEC      = 3.0
RETARGET_DIST_M = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Handover trigger  (depth-based foreground detection)
# ─────────────────────────────────────────────────────────────────────────────
HANDOVER_SETTLE_SEC  = 1.0   # wait after arm reaches pose before capturing BG
HANDOVER_BG_FRAMES   = 10    # frames to build "arm-at-handover" background
HANDOVER_BG_THRESH_M = 0.08  # foreground if depth < bg − 8 cm
HANDOVER_TRIG_PX     = 1500  # px² of foreground needed to trigger release
HANDOVER_CONSEC      = 2     # consecutive trigger frames required (debounce)
MIN_DEPTH_M          = 0.10
MAX_DEPTH_M          = 3.00


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
    half  = math.radians(angle_deg) / 2.0
    q_rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base = (0.0, 1.0, 0.0, 0.0)
    w, x, y, z = quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ─────────────────────────────────────────────────────────────────────────────
class HandoverPickNode(Node):
    """
    Pick object → carry to handover pose → wait for human to enter camera frame
    (shoes / legs / any foreground blob > HANDOVER_TRIG_PX) → open gripper → home.

    State machine:
        IDLE → HOVER → PLUNGE → LIFT → TO_HANDOVER
             → HANDOVER_SETTLE (build bg while arm is still)
             → HANDOVER_WAIT   (monitor depth for person)
             → RELEASE → HOME → IDLE

    Runs alongside any_pose_detector_rs  (subscribes to /object_pose + /object_angle).
    Does NOT share the arm with pick_node_any_pose — run one OR the other.
    """

    def __init__(self):
        super().__init__('handover_pick_node')

        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        self.current_joint_map = {}

        self.is_moving      = True
        self.motion_stage   = 'IDLE'
        self.sequence_timer = None
        self._pending_tight = False

        self.goal_x = self.goal_y = self.goal_z = None
        self.goal_angle = 0.0
        self.hover_z = self.plunge_z = None

        self.last_x = self.last_y = None
        self.pending_x = self.pending_y = self.pending_z = self.pending_start = None
        self._latest_x = self._latest_y = self._latest_z = None
        self._latest_angle = 0.0

        # ── Handover depth detection ──────────────────────────────────────────
        self.bridge             = CvBridge()
        self._depth_img         = None
        self._handover_bg       = None
        self._handover_bg_buf   = []
        self._handover_settle_t = None
        self._consec_detect     = 0
        self._detect_timer      = None
        self._depth_sub         = None   # created only at handover pose, destroyed after release

        # ── ROS clients / publishers / subscribers ────────────────────────────
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.create_subscription(PoseStamped, '/object_pose',  self._pose_cb,  10)
        self.create_subscription(Float64,     '/object_angle', self._angle_cb, 10)
        self.create_subscription(JointState,  '/joint_states', self._joint_cb, 20)
        # depth NOT subscribed here — subscribed lazily when arm reaches handover pose

        self.gripper_pub     = self.create_publisher(Int32MultiArray, '/gripper_control', 10)
        self.pick_active_pub = self.create_publisher(Bool, '/pick_active', 10)

        self._startup_timer = self.create_timer(3.0, self._startup_home_cb)

        self.get_logger().info(
            'Handover Pick Node ready — moving to home, then detection starts. '
            'After pick: arm goes to handover pose and waits for you to enter frame.')

    # ─────────────────────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def _angle_cb(self, msg):
        self._latest_angle = msg.data

    def _depth_cb(self, msg):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self._depth_img = raw.astype(np.float32) / 1000.0
        except Exception:
            pass

    def _have_joints(self):
        return all(n in self.current_joint_map for n in self.ur_joint_names)

    def _gripper(self, pos, speed=150, force=100):
        m = Int32MultiArray()
        m.data = [pos, speed, force]
        self.gripper_pub.publish(m)

    def _set_pick_active(self, active: bool):
        m = Bool()
        m.data = active
        self.pick_active_pub.publish(m)

    def _clear_timer(self):
        if self.sequence_timer is not None:
            self.destroy_timer(self.sequence_timer)
            self.sequence_timer = None

    def _clear_detect_timer(self):
        if self._detect_timer is not None:
            self.destroy_timer(self._detect_timer)
            self._detect_timer = None

    def _start_depth(self):
        if self._depth_sub is None:
            self._depth_sub = self.create_subscription(
                Image, '/camera/camera/aligned_depth_to_color/image_raw',
                self._depth_cb, 10)
            self._depth_img = None

    def _stop_depth(self):
        if self._depth_sub is not None:
            self.destroy_subscription(self._depth_sub)
            self._depth_sub = None
            self._depth_img = None

    # ─────────────────────────────────────────────────────────────────────────
    def _startup_home_cb(self):
        self.destroy_timer(self._startup_timer)
        if not self._have_joints():
            self._startup_timer = self.create_timer(1.0, self._startup_home_cb)
            self.get_logger().info('Waiting for joint states...')
            return
        self._set_pick_active(True)
        self.get_logger().info('Moving to home...')
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

    @staticmethod
    def _dist2d(x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _commit_pick(self, x, y, z):
        self.goal_x     = x
        self.goal_y     = y
        self.goal_z     = z
        self.goal_angle = (self._latest_angle + GRIPPER_ANGLE_OFFSET) % 180.0
        self.hover_z    = z + GRIPPER_LENGTH + HOVER_GAP
        self.plunge_z   = z + GRIPPER_LENGTH - GRASP_DEPTH
        self.pending_x = self.pending_y = self.pending_z = self.pending_start = None
        self._set_pick_active(True)
        self.get_logger().info(
            f'Pick committed  obj=({x:.3f},{y:.3f},{z:.3f})  '
            f'hover_z={self.hover_z:.3f}  plunge_z={self.plunge_z:.3f}  '
            f'angle={self.goal_angle:.1f}°')
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
            self.get_logger().error(f'IK failed at {self.motion_stage}. Unlocking.')
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

    def _plan_joints(self, goal_positions, tight=False, vel=0.08, acc=0.08):
        tol = TIGHT_TOLERANCE if tight else LOOSE_TOLERANCE
        req = GetMotionPlan.Request()
        mp  = req.motion_plan_request
        mp.group_name                      = 'ur_manipulator'
        mp.num_planning_attempts           = 5 if tight else 3
        mp.allowed_planning_time           = 3.0 if tight else 2.5
        mp.max_velocity_scaling_factor     = vel
        mp.max_acceleration_scaling_factor = acc
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
        self._clear_detect_timer()
        self._stop_depth()
        self.is_moving    = False
        self.motion_stage = 'IDLE'
        self._set_pick_active(False)

    # ─────────────────────────────────────────────────────────────────────────
    def _result_cb(self, future):
        stage = self.motion_stage
        if stage == 'HOME':
            self.get_logger().info('At home. Detection active.')
            self._unlock()
        elif stage == 'HOVER':
            self.get_logger().info('Hover reached. Opening gripper...')
            self.sequence_timer = self.create_timer(0.3, self._open_then_plunge)
        elif stage == 'PLUNGE':
            self.get_logger().info('Plunge done. Closing gripper...')
            self.sequence_timer = self.create_timer(0.5, self._close_then_lift)
        elif stage == 'LIFT':
            self.get_logger().info('Lift done. Moving to handover pose...')
            self.sequence_timer = self.create_timer(0.3, self._go_handover)
        elif stage == 'TO_HANDOVER':
            self.get_logger().info(
                f'At handover pose. Settling {HANDOVER_SETTLE_SEC:.0f} s '
                'then building background...')
            self.motion_stage         = 'HANDOVER_SETTLE'
            self._handover_bg_buf     = []
            self._handover_bg         = None
            self._consec_detect       = 0
            self._handover_settle_t   = time.time()
            self._start_depth()   # subscribe to depth only now — no overhead during pick
            self._detect_timer = self.create_timer(0.1, self._handover_monitor_cb)

    # ── Pick steps ────────────────────────────────────────────────────────────
    def _go_hover(self):
        self.get_logger().info(f'Moving to hover z={self.hover_z:.3f} m...')
        self.motion_stage = 'HOVER'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _open_then_plunge(self):
        self._clear_timer()
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(0.5, self._do_plunge)

    def _do_plunge(self):
        self._clear_timer()
        self.get_logger().info(f'Plunging to z={self.plunge_z:.3f} m (tight)...')
        self.motion_stage = 'PLUNGE'
        self._ik_move(self.goal_x, self.goal_y, self.plunge_z, tight=True)

    def _close_then_lift(self):
        self._clear_timer()
        self._gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self._do_lift)

    def _do_lift(self):
        self._clear_timer()
        self.get_logger().info('Lifting...')
        self.motion_stage = 'LIFT'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _go_handover(self):
        self._clear_timer()
        self.get_logger().info('Carrying to handover pose...')
        self.motion_stage = 'TO_HANDOVER'
        self._plan_joints(HANDOVER_JOINTS, tight=False, vel=0.12, acc=0.10)

    # ── Handover depth monitor ────────────────────────────────────────────────
    def _handover_monitor_cb(self):
        """10 Hz timer: settle → capture BG → wait for person → release."""
        if self._depth_img is None:
            return

        # ── Phase 1: settle ───────────────────────────────────────────────────
        if self.motion_stage == 'HANDOVER_SETTLE':
            elapsed = time.time() - self._handover_settle_t
            if elapsed < HANDOVER_SETTLE_SEC:
                self._show_handover_frame('Settling...', elapsed / HANDOVER_SETTLE_SEC)
                return
            # accumulate background frames
            self._handover_bg_buf.append(self._depth_img.copy())
            pct = len(self._handover_bg_buf) / HANDOVER_BG_FRAMES
            self._show_handover_frame(
                f'Building BG {len(self._handover_bg_buf)}/{HANDOVER_BG_FRAMES}', pct)
            if len(self._handover_bg_buf) >= HANDOVER_BG_FRAMES:
                stack = np.stack(self._handover_bg_buf, axis=0)
                self._handover_bg = np.median(stack, axis=0).astype(np.float32)
                self._handover_bg_buf = []
                self.motion_stage = 'HANDOVER_WAIT'
                self.get_logger().info(
                    'Handover BG ready. '
                    f'Bring your shoes/hand into camera view to release (> {HANDOVER_TRIG_PX} px).')
            return

        # ── Phase 2: wait for person ──────────────────────────────────────────
        if self.motion_stage == 'HANDOVER_WAIT':
            valid = (self._handover_bg > MIN_DEPTH_M) & (self._depth_img > MIN_DEPTH_M)
            diff  = self._handover_bg - self._depth_img
            fg    = (
                valid &
                (diff > HANDOVER_BG_THRESH_M) &
                (self._depth_img < MAX_DEPTH_M)
            ).astype(np.uint8) * 255
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            area = int(cv2.countNonZero(fg))

            if area >= HANDOVER_TRIG_PX:
                self._consec_detect += 1
            else:
                self._consec_detect = 0

            self._show_handover_frame(
                f'Waiting for hand/shoes  fg={area} px  '
                f'({self._consec_detect}/{HANDOVER_CONSEC})',
                min(area / HANDOVER_TRIG_PX, 1.0),
                fg_mask=fg)

            if self._consec_detect >= HANDOVER_CONSEC:
                self._trigger_release()

    def _trigger_release(self):
        self._clear_detect_timer()
        self._stop_depth()   # unsubscribe — no depth overhead during home transit
        self.get_logger().info('Person detected! Opening gripper to release object.')
        self.motion_stage = 'RELEASE'
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._go_home_after_release)

    def _go_home_after_release(self):
        self._clear_timer()
        self.last_x = None   # clear last pick — arm is free to pick again
        self.last_y = None
        self.get_logger().info('Object released. Returning to home...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False, vel=0.12, acc=0.10)

    # ── Visualisation ─────────────────────────────────────────────────────────
    def _show_handover_frame(self, label, progress=0.0, fg_mask=None):
        h, w = self._depth_img.shape
        if fg_mask is not None:
            vis = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        else:
            # normalise depth to 8-bit for display
            disp = np.clip(self._depth_img, 0, MAX_DEPTH_M)
            disp = (disp / MAX_DEPTH_M * 255).astype(np.uint8)
            vis  = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        # progress bar
        bar_w = int(w * min(progress, 1.0))
        cv2.rectangle(vis, (0, h - 8), (bar_w, h), (0, 220, 100), -1)

        # status banner
        cv2.rectangle(vis, (0, 0), (w, 30), (30, 30, 30), -1)
        cv2.putText(vis, label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.imshow('Handover Monitor', vis)
        cv2.waitKey(1)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = HandoverPickNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
