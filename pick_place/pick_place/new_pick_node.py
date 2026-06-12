import math
import time

import numpy as np
from scipy.interpolate import RBFInterpolator

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
LOOSE_TOLERANCE = 0.02    # hover / lift — fast planning
TIGHT_TOLERANCE = 0.003   # plunge only  — repeatable Z depth


# ============================================================
# Color priority order
# ============================================================
COLOR_PRIORITY = ['RED', 'BLUE']


# ============================================================
# 2-D PROJECTION-ERROR CORRECTION
# ============================================================
#
# WHY THIS EXISTS
# ───────────────
# A small camera TF offset / tilt causes the ray-plane intersection to
# drift laterally as the box moves away from the camera's principal axis.
# The error grows with distance from the zero-error point and also changes
# with table depth (how far the box is from the robot).
# An RBF correction map (thin-plate spline) is exact at calibration points
# and smoothly interpolated / extrapolated everywhere else.
#
# ─────────────────────────────────────────────────────────────
# HOW TO CALIBRATE  (one-time, ~10 minutes)
# ─────────────────────────────────────────────────────────────
#
#  Step 1 ── Set CALIBRATION_MODE = True below and run the node.
#
#  Step 2 ── Place the box at the CENTER of each of your 12 positions
#             (6 columns × bottom row + 10 cm front row).
#             Every detection prints a line:
#               [CAL] raw_x=0.3421  raw_y=-0.2715
#             Write down raw_x and raw_y for each position.
#             (The arm does NOT move in CAL mode.)
#
#  Step 3 ── Fill raw_x / raw_y into CALIBRATION_POINTS below.
#             The correction values are pre-filled from your measurements:
#               correction_x = +0.04  → arm went 4 cm LEFT  (shift RIGHT)
#               correction_x = -0.04  → arm went 4 cm RIGHT (shift LEFT)
#               correction_y = 0      → no forward/back error observed
#
#  Step 4 ── Set CALIBRATION_MODE = False and run.  Done.
#
# ─────────────────────────────────────────────────────────────
# YOUR MEASURED ERRORS (arm position vs. actual box position)
# ─────────────────────────────────────────────────────────────
#
#   Position │ bottom-row error │ front-row (+10 cm) error
#   ─────────┼──────────────────┼─────────────────────────
#   Part 1   │   0   cm         │  −3.5 cm (left)
#   Part 2   │  −4   cm (left)  │  −3.0 cm (left)
#   Part 3   │  −3   cm (left)  │  −1.0 cm (left)
#   Part 4   │   0   cm ✓       │   0.0 cm ✓
#   Part 5   │  +3   cm (right) │  +2.8 cm (right)
#   Part 6   │  +4   cm (right) │  +3.8 cm (right)
#
#   correction = −error  (opposite sign)
#
# ─────────────────────────────────────────────────────────────

CALIBRATION_MODE = False   # ← set True for the calibration logging run

# Replace every ??? with the raw_x / raw_y printed during the CAL run.
# Correction values are already filled from your measured errors above.
CALIBRATION_POINTS = [
    # (raw_x_m,  raw_y_m,   corr_x_m,  corr_y_m)

    # ── bottom row (far edge of table) ───────────────────────────────
    # (???,      ???,        0.000,     0.000),   # part 1 — correct
    # (???,      ???,        0.040,     0.000),   # part 2 — arm 4 cm left
    # (???,      ???,        0.030,     0.000),   # part 3 — arm 3 cm left
    # (???,      ???,        0.000,     0.000),   # part 4 — correct
    # (???,      ???,       -0.030,     0.000),   # part 5 — arm 3 cm right
    # (???,      ???,       -0.040,     0.000),   # part 6 — arm 4 cm right

    # ── front row (+0.10 m closer to robot) ─────────────────────────
    # (???,      ???,        0.035,     0.000),   # part 1
    # (???,      ???,        0.030,     0.000),   # part 2
    # (???,      ???,        0.010,     0.000),   # part 3
    # (???,      ???,        0.000,     0.000),   # part 4 — correct
    # (???,      ???,       -0.028,     0.000),   # part 5
    # (???,      ???,       -0.038,     0.000),   # part 6
]


# ============================================================
class CorrectionMap:
    """
    Smooth 2-D (x, y) projection-error correction via thin-plate-spline RBF.

      • Exact at every calibration point.
      • C2-smooth and well-behaved everywhere else.
      • Falls back to identity (no correction) when fewer than 4
        calibration points are provided.
    """

    def __init__(self, calib_points):
        self._active = False
        if len(calib_points) < 4:
            return
        xy = np.array([[p[0], p[1]] for p in calib_points], dtype=float)
        cx = np.array([p[2] for p in calib_points], dtype=float)
        cy = np.array([p[3] for p in calib_points], dtype=float)
        self._rbf_x = RBFInterpolator(xy, cx, kernel='thin_plate_spline')
        self._rbf_y = RBFInterpolator(xy, cy, kernel='thin_plate_spline')
        self._active = True

    @property
    def active(self):
        return self._active

    def correct(self, raw_x: float, raw_y: float):
        """Return (corrected_x, corrected_y). Identity when inactive."""
        if not self._active:
            return raw_x, raw_y
        pt = np.array([[raw_x, raw_y]])
        return (raw_x + float(self._rbf_x(pt)[0]),
                raw_y + float(self._rbf_y(pt)[0]))


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


def gripper_orientation_for_angle(angle_deg):
    angle_rad = math.radians(angle_deg)
    half      = angle_rad / 2.0
    q_rot     = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base    = (0.0, 1.0, 0.0, 0.0)
    w, x, y, z = quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ============================================================
class PickNode(Node):
    def __init__(self):
        super().__init__('pick_node')

        # ── Projection-error correction ───────────────────────────────
        self.correction_map = CorrectionMap(CALIBRATION_POINTS)
        if self.correction_map.active:
            self.get_logger().info(
                f"Correction map ACTIVE with "
                f"{len(CALIBRATION_POINTS)} calibration points.")
        elif CALIBRATION_MODE:
            self.get_logger().warn(
                "CALIBRATION MODE ON — raw projections will be logged. "
                "Robot will NOT move.")
        else:
            self.get_logger().warn(
                "No calibration points loaded — correction DISABLED. "
                "Run with CALIBRATION_MODE=True to calibrate.")

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.is_moving    = False
        self.motion_stage = 'IDLE'
        self.sequence_timer = None

        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        self.current_joint_map = {}

        # ── Per-color state ──────────────────────────────────────────
        self.color_state = {
            'RED':  {'latest_x': None, 'latest_y': None,
                     'latest_angle': 0.0,
                     'last_x': None,   'last_y': None,
                     'last_angle': None,
                     'pending_x': None, 'pending_y': None,
                     'pending_start': None},
            'BLUE': {'latest_x': None, 'latest_y': None,
                     'latest_angle': 0.0,
                     'last_x': None,   'last_y': None,
                     'last_angle': None,
                     'pending_x': None, 'pending_y': None,
                     'pending_start': None},
        }

        self.active_color   = None
        self.current_goal_x = None
        self.current_goal_y = None
        self.goal_box_angle = 0.0

        self.retarget_threshold = 0.05
        self.stable_wait_sec    = 3.0

        self._pending_tight = False

        # ── ROS infrastructure ───────────────────────────────────────
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.create_subscription(
            Point,   '/red_box_ray',    self._make_ray_cb('RED'),    10)
        self.create_subscription(
            Float64, '/red_box_angle',  self._make_angle_cb('RED'),  10)
        self.create_subscription(
            Point,   '/blue_box_ray',   self._make_ray_cb('BLUE'),   10)
        self.create_subscription(
            Float64, '/blue_box_angle', self._make_angle_cb('BLUE'), 10)

        self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 20)

        self.gripper_pub = self.create_publisher(
            Int32MultiArray, '/gripper_control', 10)

        # ── Z geometry ───────────────────────────────────────────────
        self.table_top_z    = 0.64
        self.hover_gap      = 0.15
        self.gripper_length = 0.27
        self.target_flange_z = (self.table_top_z
                                + self.hover_gap
                                + self.gripper_length)
        self.plunge_depth = 0.08

        self.get_logger().info(
            "Pick Node ready (with orientation + correction) — priority: "
            + " → ".join(COLOR_PRIORITY))

    # ─────────────────────────────────────────────────────────────────
    def _make_angle_cb(self, color):
        def cb(msg):
            self.color_state[color]['latest_angle'] = msg.data
        return cb

    def _make_ray_cb(self, color):
        def cb(msg):
            self._ray_callback(msg, color)
        return cb

    # ─────────────────────────────────────────────────────────────────
    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def have_all_joint_states(self):
        return all(n in self.current_joint_map for n in self.ur_joint_names)

    def distance_2d(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def control_gripper(self, pos, speed, force):
        msg = Int32MultiArray()
        msg.data = [pos, speed, force]
        self.gripper_pub.publish(msg)

    # ─────────────────────────────────────────────────────────────────
    def _ray_callback(self, msg, color):
        if not self.have_all_joint_states():
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

            # ── Apply 2-D projection-error correction ─────────────────
            final_x, final_y = self.correction_map.correct(raw_x, raw_y)

            # ── Calibration mode: log raw coords and stop ─────────────
            if CALIBRATION_MODE:
                self.get_logger().info(
                    f"[CAL][{color}] raw_x={raw_x:.4f}  raw_y={raw_y:.4f}  "
                    f"→  corrected_x={final_x:.4f}  corrected_y={final_y:.4f}  "
                    f"(correction active: {self.correction_map.active})")
                return   # ← no motion during calibration logging

            cs = self.color_state[color]

            # Always update the latest known position for this color
            cs['latest_x'] = final_x
            cs['latest_y'] = final_y

            # ── Only check for a new pick trigger when robot is idle ──
            if self.is_moving:
                return

            highest_idle = self._highest_priority_ready_color()
            if highest_idle != color:
                return

            now_sec = time.time()

            # ── First ever pick for this color ────────────────────────
            if cs['last_x'] is None:
                self._commit_pick(color, final_x, final_y)
                return

            dist_from_last = self.distance_2d(
                final_x, final_y, cs['last_x'], cs['last_y'])

            if dist_from_last < self.retarget_threshold:
                angle_changed = (cs['last_angle'] is not None and
                                 cs['latest_angle'] != cs['last_angle'])
                if not angle_changed:
                    cs['pending_x'] = cs['pending_y'] = \
                        cs['pending_start'] = None
                    return
                final_x, final_y = cs['last_x'], cs['last_y']

            # ── Pending timer ─────────────────────────────────────────
            if cs['pending_x'] is None:
                cs['pending_x'] = final_x
                cs['pending_y'] = final_y
                cs['pending_start'] = now_sec
                return

            dist_from_pending = self.distance_2d(
                final_x, final_y, cs['pending_x'], cs['pending_y'])
            if dist_from_pending >= self.retarget_threshold:
                cs['pending_x'] = final_x
                cs['pending_y'] = final_y
                cs['pending_start'] = now_sec
                return

            if (now_sec - cs['pending_start']) < self.stable_wait_sec:
                return

            # ── Stable target confirmed ───────────────────────────────
            stable_x = cs['pending_x']
            stable_y = cs['pending_y']
            cs['pending_x'] = cs['pending_y'] = cs['pending_start'] = None
            self._commit_pick(color, stable_x, stable_y)

        except TransformException as ex:
            self.get_logger().info(f"TF Error: {ex}")

    # ─────────────────────────────────────────────────────────────────
    def _highest_priority_ready_color(self):
        for color in COLOR_PRIORITY:
            if self.color_state[color]['latest_x'] is not None:
                return color
        return None

    def _commit_pick(self, color, x, y):
        cs = self.color_state[color]
        self.active_color   = color
        self.current_goal_x = x
        self.current_goal_y = y
        self.goal_box_angle = cs['latest_angle']

        self.get_logger().info(
            f"[{color}] Pick committed at ({x:.3f}, {y:.3f})  "
            f"angle={self.goal_box_angle:.0f} deg")

        self.is_moving = True
        self.start_hover_motion(x, y)

    # ─────────────────────────────────────────────────────────────────
    def start_hover_motion(self, x, y):
        self.motion_stage = 'HOVER'
        self.execute_ik_motion(x, y, self.target_flange_z, tight=False)

    def execute_ik_motion(self, x, y, z, tight=False):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass

        qx, qy, qz, qw = gripper_orientation_for_angle(self.goal_box_angle)
        self.get_logger().info(
            f"[{self.active_color}] IK z={z:.3f}  "
            f"angle={self.goal_box_angle:.0f}°  tight={tight}")

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
            self.plan_to_joint_goal(goal_positions, tight=self._pending_tight)
        except KeyError:
            self.get_logger().error("IK solution missing joint. Unlocking.")
            self.is_moving    = False
            self.motion_stage = 'IDLE'

    def plan_to_joint_goal(self, goal_positions, tight=False):
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
        future.add_done_callback(self.plan_callback)

    def plan_callback(self, future):
        response = future.result()
        if response is None or response.motion_plan_response.error_code.val != 1:
            self.get_logger().error("MoveIt planning failed. Unlocking.")
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
            self.get_logger().error("Trajectory rejected. Unlocking.")
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
            self.get_logger().info("Lift complete. Placing in 1 s...")
            self.sequence_timer = self.create_timer(1.0, self.trigger_place_plunge)

        elif self.motion_stage == 'PLACE_PLUNGE':
            self.get_logger().info("At drop height. Opening gripper...")
            self.sequence_timer = self.create_timer(0.5, self.step_open_and_retract)

        elif self.motion_stage == 'PLACE_LIFT':
            self.get_logger().info("Retract done. Deciding next action...")
            self.sequence_timer = self.create_timer(1.0, self.decide_next_action)

    # ── Step methods ──────────────────────────────────────────────────
    def step_open_and_plunge(self):
        self._clear_timer()
        self.control_gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self.trigger_plunge_motion)

    def trigger_plunge_motion(self):
        self._clear_timer()
        plunge_z = self.target_flange_z - self.plunge_depth
        self.get_logger().info(
            f"[{self.active_color}] Plunging to z={plunge_z:.4f} m (tight)...")
        self.motion_stage = 'PLUNGE'
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y, plunge_z, tight=True)

    def step_close_and_lift(self):
        self._clear_timer()
        self.control_gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self.trigger_lift_motion)

    def trigger_lift_motion(self):
        self._clear_timer()
        self.motion_stage = 'LIFT'
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            self.target_flange_z, tight=False)

    def trigger_place_plunge(self):
        self._clear_timer()
        plunge_z = self.target_flange_z - self.plunge_depth
        self.get_logger().info(
            f"[{self.active_color}] Dropping at z={plunge_z:.4f} m (tight)...")
        self.motion_stage = 'PLACE_PLUNGE'
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y, plunge_z, tight=True)

    def step_open_and_retract(self):
        self._clear_timer()
        self.control_gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self.trigger_place_lift)

    def trigger_place_lift(self):
        self._clear_timer()
        self.motion_stage = 'PLACE_LIFT'
        self.execute_ik_motion(
            self.current_goal_x, self.current_goal_y,
            self.target_flange_z, tight=False)

    # ─────────────────────────────────────────────────────────────────
    def decide_next_action(self):
        self._clear_timer()

        # ── 1. Record completed pick ──────────────────────────────────
        done_color = self.active_color
        cs = self.color_state[done_color]
        cs['last_x']     = self.current_goal_x
        cs['last_y']     = self.current_goal_y
        cs['last_angle'] = self.goal_box_angle
        cs['pending_x']  = cs['pending_y'] = cs['pending_start'] = None

        self.get_logger().info(
            f"[{done_color}] Sequence complete. "
            f"Checking for next color to pick...")

        # ── 2. Find next color to pick ────────────────────────────────
        current_idx = COLOR_PRIORITY.index(done_color)

        for offset in range(1, len(COLOR_PRIORITY) + 1):
            idx        = (current_idx + offset) % len(COLOR_PRIORITY)
            next_color = COLOR_PRIORITY[idx]
            next_cs    = self.color_state[next_color]

            if next_cs['latest_x'] is None:
                continue

            if (next_cs['last_x'] is not None and
                    self.distance_2d(next_cs['latest_x'], next_cs['latest_y'],
                                     next_cs['last_x'],   next_cs['last_y'])
                    < self.retarget_threshold and
                    next_cs['latest_angle'] == next_cs['last_angle']):
                self.get_logger().info(
                    f"[{next_color}] visible but already picked at same "
                    f"spot/angle — skipping.")
                continue

            # ── 3. Commit pick for next_color ─────────────────────────
            self.get_logger().info(
                f"[{next_color}] detected at "
                f"({next_cs['latest_x']:.3f}, {next_cs['latest_y']:.3f}) "
                f"— picking now!")
            self._commit_pick(
                next_color, next_cs['latest_x'], next_cs['latest_y'])
            return

        # ── 4. Nothing to pick — go idle ─────────────────────────────
        self.get_logger().info(
            "No further objects to pick. Waiting for new targets...")
        self.active_color = None
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
