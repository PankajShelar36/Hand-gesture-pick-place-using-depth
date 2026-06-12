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
# Tolerance constants
# ============================================================
LOOSE_TOLERANCE = 0.02    # joint-space transit moves
TIGHT_TOLERANCE = 0.003   # straight Z plunge/lift — repeatable depth


# ============================================================
# Fixed joint positions  (degrees → radians at startup)
# ============================================================
def _deg_to_rad(deg_list):
    return [math.radians(d) for d in deg_list]


# Approach position above first pick spot
POS_A = _deg_to_rad([ 26.92,  -93.96,  -30.99, -143.27,  88.68, -60.14])

# Approach position above drop / place spot
POS_B = _deg_to_rad([-49.64,  -89.08,  -47.57, -131.81,  88.38, -47.27])

# Final drop position for red object
POS_C = _deg_to_rad([-56.19, -102.59,  -20.02, -145.68,  88.51, -53.82])


# ============================================================
# Straight-line Z distances  (meters, positive = up)
# ============================================================
DOWN_A  = 0.125   # drop straight down 12.5 cm at A to pick
UP_A    = 0.130   # lift straight up   13.0 cm after gripping at A
DOWN_B  = 0.112   # drop straight down 11.2 cm at B to place
UP_B    = 0.150   # lift straight up   15.0 cm after releasing at B


# ============================================================
# Camera / table geometry  (for red-object detection)
# ============================================================
TABLE_TOP_Z    = 0.64
HOVER_GAP      = 0.15
GRIPPER_LENGTH = 0.27
HOVER_Z        = TABLE_TOP_Z + HOVER_GAP + GRIPPER_LENGTH   # flange hover height
PLUNGE_DEPTH   = 0.08                                        # how far below hover to descend


# ============================================================
# Quaternion helpers
# ============================================================
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
    """Return (qx, qy, qz, qw) for gripper pointing straight down,
    rotated angle_deg around its approach axis."""
    half  = math.radians(angle_deg) / 2.0
    q_rot = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base = (0.0, 1.0, 0.0, 0.0)   # 180° around world X = pointing down
    w, x, y, z = quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ============================================================
class SequenceNode(Node):
    """
    Fixed pick-and-place sequence:

      1. GOTO_A       — move to joint position A
      2. DOWN_A       — straight down 12.5 cm
      3. (close grip) — grasp object
      4. UP_A         — straight up 13 cm
      5. GOTO_B       — move to joint position B (shortest path)
      6. DOWN_B       — straight down 11.2 cm
      7. (open grip)  — release object
      8. UP_B         — straight up 15 cm  → transit height stored
      9. FIND_RED     — wait for red object via camera
     10. RED_HOVER    — move above detected red object at transit height
     11. RED_PLUNGE   — straight down to pick height
     12. (close grip) — grasp red object
     13. RED_LIFT     — straight up back to transit height
     14. GOTO_C       — move to joint position C (shortest path)
     15. (open grip)  — release red object
     16. DONE         — stop
    """

    # Stage constants
    ST_GOTO_A     = 'GOTO_A'
    ST_DOWN_A     = 'DOWN_A'
    ST_UP_A       = 'UP_A'
    ST_GOTO_B     = 'GOTO_B'
    ST_DOWN_B     = 'DOWN_B'
    ST_UP_B       = 'UP_B'
    ST_RED_HOVER  = 'RED_HOVER'
    ST_RED_PLUNGE = 'RED_PLUNGE'
    ST_RED_LIFT   = 'RED_LIFT'
    ST_GOTO_C     = 'GOTO_C'
    ST_DONE       = 'DONE'

    def __init__(self):
        super().__init__('sequence_node')

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.motion_stage   = 'IDLE'
        self.sequence_timer = None
        self._pending_tight = False

        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.current_joint_map = {}

        # Stored Cartesian positions for straight Z moves
        # Set by _read_tool_pose() just before each plunge/lift
        self._tool_x = None
        self._tool_y = None
        self._tool_z = None

        # Transit height after UP_B — used as lift target after red pick
        self.transit_z = None

        # Red object state
        self.red_angle       = 0.0
        self.searching_red   = False   # True only while waiting for red
        self._red_stable_x   = None
        self._red_stable_y   = None
        self._red_pending_x  = None
        self._red_pending_y  = None
        self._red_pending_t  = None
        self.RED_STABLE_SEC  = 2.0     # seconds of consistency before committing

        # ── ROS services / actions ────────────────────────────────────
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        # ── Subscriptions ─────────────────────────────────────────────
        self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 20)
        self.create_subscription(
            Float64, '/red_box_angle', self._angle_cb, 10)
        self.create_subscription(
            Point, '/red_box_ray', self._ray_cb, 10)

        # ── Gripper publisher ─────────────────────────────────────────
        self.gripper_pub = self.create_publisher(
            Int32MultiArray, '/gripper_control', 10)

        self.get_logger().info(
            "Sequence node ready. Starting in 2 s...")
        self.sequence_timer = self.create_timer(2.0, self._start_sequence)

    # ─────────────────────────────────────────────────────────────────
    # Basic helpers
    # ─────────────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def _angle_cb(self, msg):
        self.red_angle = msg.data

    def _have_joints(self):
        return all(n in self.current_joint_map for n in self.ur_joint_names)

    def _gripper(self, pos, speed=150, force=100):
        msg = Int32MultiArray()
        msg.data = [pos, speed, force]
        self.gripper_pub.publish(msg)

    def _clear_timer(self):
        if self.sequence_timer is not None:
            self.destroy_timer(self.sequence_timer)
            self.sequence_timer = None

    # ─────────────────────────────────────────────────────────────────
    # Read current TCP pose via TF (tool0 → base)
    # ─────────────────────────────────────────────────────────────────
    def _read_tool_pose(self):
        """
        Look up the current tool0 position in the base frame and store it
        in self._tool_x/y/z.  Called just before every straight Z move so
        the IK target is exactly relative to where the arm actually stopped.
        Returns True on success, False on TF error.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                'base', 'tool0',
                rclpy.time.Time(),
                timeout=Duration(seconds=2.0))
            self._tool_x = t.transform.translation.x
            self._tool_y = t.transform.translation.y
            self._tool_z = t.transform.translation.z
            self.get_logger().info(
                f"Tool pose read: x={self._tool_x:.4f}  "
                f"y={self._tool_y:.4f}  z={self._tool_z:.4f}")
            return True
        except TransformException as ex:
            self.get_logger().error(f"TF lookup failed: {ex}")
            return False

    # ─────────────────────────────────────────────────────────────────
    # Red-object ray callback  (active only while searching_red=True)
    # ─────────────────────────────────────────────────────────────────
    def _ray_cb(self, msg):
        if not self.searching_red or not self._have_joints():
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

            origin_b = tf2_geometry_msgs.do_transform_point(origin_cam, t)
            ray_b    = tf2_geometry_msgs.do_transform_point(ray_cam, t)

            dx = ray_b.point.x - origin_b.point.x
            dy = ray_b.point.y - origin_b.point.y
            dz = ray_b.point.z - origin_b.point.z
            if abs(dz) < 1e-6:
                return

            scale   = (TABLE_TOP_Z - origin_b.point.z) / dz
            final_x = origin_b.point.x + scale * dx
            final_y = origin_b.point.y + scale * dy

            now = time.time()

            if self._red_pending_x is None:
                self._red_pending_x = final_x
                self._red_pending_y = final_y
                self._red_pending_t = now
                return

            dist = math.sqrt((final_x - self._red_pending_x)**2 +
                             (final_y - self._red_pending_y)**2)

            if dist > 0.04:   # moved >4 cm — reset timer
                self._red_pending_x = final_x
                self._red_pending_y = final_y
                self._red_pending_t = now
                return

            if (now - self._red_pending_t) < self.RED_STABLE_SEC:
                return

            # ── Stable red object confirmed ───────────────────────────
            self.get_logger().info(
                f"[RED] Stable at ({final_x:.3f}, {final_y:.3f}) "
                f"angle={self.red_angle:.0f}°  — committing pick!")
            self._red_stable_x = self._red_pending_x
            self._red_stable_y = self._red_pending_y
            self._red_pending_x = self._red_pending_y = self._red_pending_t = None
            self.searching_red  = False

            # Start red pick sequence
            self._trigger_red_hover()

        except TransformException as ex:
            self.get_logger().info(f"TF Error in ray_cb: {ex}")

    # ─────────────────────────────────────────────────────────────────
    # IK motion
    # ─────────────────────────────────────────────────────────────────
    def _ik_move(self, x, y, z, angle_deg=0.0, tight=False):
        """Request IK then plan to the resulting joint goal."""
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass

        qx, qy, qz, qw = gripper_quat(angle_deg)

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
        req.ik_request.group_name       = "ur_manipulator"
        req.ik_request.ik_link_name     = "tool0"
        req.ik_request.pose_stamped     = target_pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout          = BuiltinDuration(sec=1, nanosec=0)
        req.ik_request.robot_state.joint_state.name     = self.ur_joint_names
        req.ik_request.robot_state.joint_state.position = [
            self.current_joint_map[n] for n in self.ur_joint_names]

        self._pending_tight = tight
        future = self.ik_client.call_async(req)
        future.add_done_callback(self._ik_cb)

    def _ik_cb(self, future):
        response = future.result()
        if response is None or response.error_code.val != 1:
            self.get_logger().error(
                f"IK failed at stage {self.motion_stage}. Stopping.")
            self.motion_stage = self.ST_DONE
            return
        ik_map = dict(zip(response.solution.joint_state.name,
                          response.solution.joint_state.position))
        try:
            joints = [ik_map[n] for n in self.ur_joint_names]
            self._plan_joints(joints, tight=self._pending_tight)
        except KeyError:
            self.get_logger().error("IK solution missing joint. Stopping.")
            self.motion_stage = self.ST_DONE

    # ─────────────────────────────────────────────────────────────────
    # Joint-space planner
    # ─────────────────────────────────────────────────────────────────
    def _plan_joints(self, goal_positions, tight=False):
        tolerance = TIGHT_TOLERANCE if tight else LOOSE_TOLERANCE

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
                    joint_name=jn, position=jp,
                    tolerance_above=tolerance,
                    tolerance_below=tolerance,
                    weight=1.0))
        req.motion_plan_request.goal_constraints.append(c)

        future = self.plan_client.call_async(req)
        future.add_done_callback(self._plan_cb)

    def _plan_cb(self, future):
        response = future.result()
        if response is None or response.motion_plan_response.error_code.val != 1:
            self.get_logger().error(
                f"Planning failed at stage {self.motion_stage}. Stopping.")
            self.motion_stage = self.ST_DONE
            return
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = \
            response.motion_plan_response.trajectory.joint_trajectory
        self.traj_client.wait_for_server()
        f = self.traj_client.send_goal_async(goal_msg)
        f.add_done_callback(self._goal_resp_cb)

    def _goal_resp_cb(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().error(
                f"Trajectory rejected at stage {self.motion_stage}. Stopping.")
            self.motion_stage = self.ST_DONE
            return
        gh.get_result_async().add_done_callback(self._result_cb)

    # ─────────────────────────────────────────────────────────────────
    # State-machine dispatcher — called when each move completes
    # ─────────────────────────────────────────────────────────────────
    def _result_cb(self, future):

        if self.motion_stage == self.ST_GOTO_A:
            self.get_logger().info("At position A. Reading tool pose...")
            self.sequence_timer = self.create_timer(0.5, self._after_goto_a)

        elif self.motion_stage == self.ST_DOWN_A:
            self.get_logger().info("Down A complete. Closing gripper...")
            self.sequence_timer = self.create_timer(0.3, self._close_and_up_a)

        elif self.motion_stage == self.ST_UP_A:
            self.get_logger().info("Up A complete. Moving to position B...")
            self.sequence_timer = self.create_timer(0.3, self._trigger_goto_b)

        elif self.motion_stage == self.ST_GOTO_B:
            self.get_logger().info("At position B. Reading tool pose...")
            self.sequence_timer = self.create_timer(0.5, self._after_goto_b)

        elif self.motion_stage == self.ST_DOWN_B:
            self.get_logger().info("Down B complete. Opening gripper...")
            self.sequence_timer = self.create_timer(0.3, self._open_and_up_b)

        elif self.motion_stage == self.ST_UP_B:
            self.get_logger().info(
                "Up B complete. Now searching for red object...")
            self._start_red_search()

        elif self.motion_stage == self.ST_RED_HOVER:
            self.get_logger().info("Red hover reached. Plunging to pick...")
            self.sequence_timer = self.create_timer(0.5, self._trigger_red_plunge)

        elif self.motion_stage == self.ST_RED_PLUNGE:
            self.get_logger().info("Red plunge complete. Closing gripper...")
            self.sequence_timer = self.create_timer(0.3, self._close_and_red_lift)

        elif self.motion_stage == self.ST_RED_LIFT:
            self.get_logger().info(
                "Red lift complete. Moving to final position C...")
            self.sequence_timer = self.create_timer(0.3, self._trigger_goto_c)

        elif self.motion_stage == self.ST_GOTO_C:
            self.get_logger().info(
                "At position C. Opening gripper — sequence complete!")
            self.sequence_timer = self.create_timer(0.5, self._finish)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 1: GOTO_A ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _start_sequence(self):
        self._clear_timer()
        if not self._have_joints():
            self.get_logger().warn("Waiting for joint states...")
            self.sequence_timer = self.create_timer(1.0, self._start_sequence)
            return
        self.get_logger().info("Moving to position A...")
        self.motion_stage = self.ST_GOTO_A
        self._plan_joints(POS_A, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 2: DOWN_A ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _after_goto_a(self):
        self._clear_timer()
        if not self._read_tool_pose():
            self.sequence_timer = self.create_timer(0.5, self._after_goto_a)
            return
        target_z = self._tool_z - DOWN_A
        self.get_logger().info(
            f"Dropping straight down {DOWN_A*100:.1f} cm → z={target_z:.4f} m")
        self.motion_stage = self.ST_DOWN_A
        self._ik_move(self._tool_x, self._tool_y, target_z,
                      angle_deg=0.0, tight=True)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 3: close gripper → Step 4: UP_A ────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _close_and_up_a(self):
        self._clear_timer()
        self.get_logger().info("Closing gripper...")
        self._gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self._trigger_up_a)

    def _trigger_up_a(self):
        self._clear_timer()
        # Read fresh pose after gripping (arm may have micro-settled)
        if not self._read_tool_pose():
            self.sequence_timer = self.create_timer(0.5, self._trigger_up_a)
            return
        target_z = self._tool_z + UP_A
        self.get_logger().info(
            f"Lifting {UP_A*100:.1f} cm → z={target_z:.4f} m")
        self.motion_stage = self.ST_UP_A
        self._ik_move(self._tool_x, self._tool_y, target_z,
                      angle_deg=0.0, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 5: GOTO_B ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _trigger_goto_b(self):
        self._clear_timer()
        self.get_logger().info("Moving to position B (shortest path)...")
        self.motion_stage = self.ST_GOTO_B
        self._plan_joints(POS_B, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 6: DOWN_B ───────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _after_goto_b(self):
        self._clear_timer()
        if not self._read_tool_pose():
            self.sequence_timer = self.create_timer(0.5, self._after_goto_b)
            return
        target_z = self._tool_z - DOWN_B
        self.get_logger().info(
            f"Dropping straight down {DOWN_B*100:.1f} cm → z={target_z:.4f} m")
        self.motion_stage = self.ST_DOWN_B
        self._ik_move(self._tool_x, self._tool_y, target_z,
                      angle_deg=0.0, tight=True)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 7: open gripper → Step 8: UP_B ─────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _open_and_up_b(self):
        self._clear_timer()
        self.get_logger().info("Opening gripper — releasing object...")
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._trigger_up_b)

    def _trigger_up_b(self):
        self._clear_timer()
        if not self._read_tool_pose():
            self.sequence_timer = self.create_timer(0.5, self._trigger_up_b)
            return
        target_z = self._tool_z + UP_B
        # Store this as transit height — used after picking red
        self.transit_z = target_z
        self.get_logger().info(
            f"Lifting {UP_B*100:.1f} cm → z={target_z:.4f} m  "
            f"(transit_z stored for red pick lift)")
        self.motion_stage = self.ST_UP_B
        self._ik_move(self._tool_x, self._tool_y, target_z,
                      angle_deg=0.0, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 9: FIND_RED — wait for camera detection ─────────────────
    # ─────────────────────────────────────────────────────────────────
    def _start_red_search(self):
        """
        Enable the ray callback.  The node sits idle (no timers) until
        the red object is seen consistently for RED_STABLE_SEC seconds.
        The ray callback calls _trigger_red_hover() when ready.
        """
        self._red_pending_x = None
        self._red_pending_y = None
        self._red_pending_t = None
        self.searching_red  = True
        self.get_logger().info(
            "Searching for red object... place it on the table.")

    # ─────────────────────────────────────────────────────────────────
    # ── Step 10: RED_HOVER ───────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _trigger_red_hover(self):
        """Move above the detected red object at HOVER_Z height."""
        self._clear_timer()
        x = self._red_stable_x
        y = self._red_stable_y
        self.get_logger().info(
            f"[RED] Hovering above ({x:.3f}, {y:.3f})  z={HOVER_Z:.4f} m")
        self.motion_stage = self.ST_RED_HOVER
        # Open gripper before descending
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(0.8, lambda: (
            self._clear_timer() or
            self._ik_move(x, y, HOVER_Z, angle_deg=self.red_angle, tight=False)
        ))

    # ─────────────────────────────────────────────────────────────────
    # ── Step 11: RED_PLUNGE ──────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _trigger_red_plunge(self):
        self._clear_timer()
        plunge_z = HOVER_Z - PLUNGE_DEPTH
        x = self._red_stable_x
        y = self._red_stable_y
        self.get_logger().info(
            f"[RED] Plunging to z={plunge_z:.4f} m (tight)...")
        self.motion_stage = self.ST_RED_PLUNGE
        self._ik_move(x, y, plunge_z, angle_deg=self.red_angle, tight=True)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 12: close gripper → Step 13: RED_LIFT ──────────────────
    # ─────────────────────────────────────────────────────────────────
    def _close_and_red_lift(self):
        self._clear_timer()
        self.get_logger().info("[RED] Closing gripper...")
        self._gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self._trigger_red_lift)

    def _trigger_red_lift(self):
        self._clear_timer()
        x = self._red_stable_x
        y = self._red_stable_y
        z = self.transit_z   # lift back to the 15-cm-above-B height
        self.get_logger().info(
            f"[RED] Lifting to transit height z={z:.4f} m...")
        self.motion_stage = self.ST_RED_LIFT
        self._ik_move(x, y, z, angle_deg=self.red_angle, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 14: GOTO_C ──────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _trigger_goto_c(self):
        self._clear_timer()
        self.get_logger().info("Moving to final position C (shortest path)...")
        self.motion_stage = self.ST_GOTO_C
        self._plan_joints(POS_C, tight=False)

    # ─────────────────────────────────────────────────────────────────
    # ── Step 15: open gripper → DONE ─────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    def _finish(self):
        self._clear_timer()
        self.get_logger().info("[RED] Opening gripper — releasing at C.")
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._done)

    def _done(self):
        self._clear_timer()
        self.motion_stage = self.ST_DONE
        self.get_logger().info(
            "══════════════════════════════════\n"
            "  Sequence complete. Node idle.   \n"
            "══════════════════════════════════")


def main(args=None):
    rclpy.init(args=args)
    node = SequenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
