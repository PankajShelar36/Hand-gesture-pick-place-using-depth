"""
gesture_place_node.py

Pick → lift to hover (object in gripper) → show fingers to select destination:
  1 finger   → HANDOVER pose
  2 fingers  → PLACE_B pose
  3 fingers  → PLACE_C pose
Arm moves to chosen pose → open palm then CLOSE FIST to release → home.

Camera opens once at node startup (same latency design as gesture_handover_pick_node).
Gesture thread runs the whole lifetime; _gesture_mode flag switches its behaviour.
"""

import math
import time
import threading

import cv2
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

import depthai as dai
import mediapipe as mediapipe_module

# ── Joint poses (teach-pendant degrees → radians) ─────────────────────────────
HOME_JOINTS     = [math.radians(d) for d in [ 77.03,  -88.01,   63.03,  -63.82,  -88.73, 266.75]]
HANDOVER_JOINTS = [math.radians(d) for d in [ 78.85,  -89.08,   65.74,  -76.01,    2.16, 277.87]]
PLACE_B_JOINTS  = [math.radians(d) for d in [155.21,  -88.06,  109.61, -112.83,  -87.12, 246.68]]
PLACE_C_JOINTS  = [math.radians(d) for d in [  6.79,  -88.80,   66.67,  -65.21,  -91.98,  98.38]]

DEST_JOINTS = {1: HANDOVER_JOINTS, 2: PLACE_B_JOINTS, 3: PLACE_C_JOINTS}
DEST_NAMES  = {1: 'Handover',      2: 'Place B',      3: 'Place C'}
DEST_COLORS = {1: (0, 210, 60),    2: (0, 140, 255),  3: (200, 80, 255)}

# ── Geometry (metres) ──────────────────────────────────────────────────────────
GRIPPER_LENGTH       = 0.27
HOVER_GAP            = 0.15
GRASP_DEPTH          = 0.05
GRIPPER_ANGLE_OFFSET = -90.0

# ── MoveIt tolerances ──────────────────────────────────────────────────────────
LOOSE_TOLERANCE = 0.02
TIGHT_TOLERANCE = 0.003

# ── Object stability gate ──────────────────────────────────────────────────────
STABLE_SEC      = 3.0
RETARGET_DIST_M = 0.05

# ── Gesture recognition ────────────────────────────────────────────────────────
SELECT_HOLD_SEC = 1.5  # seconds to hold finger count before destination is confirmed
RELEASE_HOLD    = 3    # frames to hold open/closed state for gripper release (~0.10 s)
WAVE_WINDOW  = 6.0  # seconds: open seen but no close → reset release state
CAM_WIDTH    = 640
CAM_HEIGHT   = 480

_TIPS = [8, 12, 16, 20]
_PIPS = [6, 10, 14, 18]


def _count_fingers(landmarks):
    """Count extended non-thumb fingers (0-4)."""
    return sum(
        1 for t, p in zip(_TIPS, _PIPS)
        if landmarks.landmark[t].y < landmarks.landmark[p].y
    )


def _classify_release(landmarks):
    """Return 'open', 'closed', or None for the release gesture (fist = 0 fingers)."""
    n = _count_fingers(landmarks)
    if n >= 3:
        return 'open'
    if n == 0:
        return 'closed'
    return None   # 1-2 fingers = mid-transition, ignore


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    )


def _gripper_quat(angle_deg=0.0):
    half   = math.radians(angle_deg) / 2.0
    q_rot  = (math.cos(half), 0.0, 0.0, math.sin(half))
    q_base = (0.0, 1.0, 0.0, 0.0)
    w, x, y, z = _quat_multiply(q_base, q_rot)
    return float(x), float(y), float(z), float(w)


# ─────────────────────────────────────────────────────────────────────────────
class GesturePlaceNode(Node):
    """
    Pick object → at hover (object in hand), show 1/2/3 fingers to choose
    destination → arm moves there → open-then-close fist to release → home.

    _gesture_mode (set from ROS callbacks, read by camera thread):
        'idle'    — show landmarks only, no detection
        'select'  — detect finger count 1/2/3 to choose destination
        'release' — detect open→close fist to open gripper
    """

    def __init__(self):
        super().__init__('gesture_place_node')

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

        # ── Gesture state shared with camera thread ────────────────────────────
        self._gesture_mode    = 'idle'   # 'idle' | 'select' | 'release'
        self._gesture_trigger = threading.Event()
        self._gesture_stop    = threading.Event()
        self._selected_dest   = 0        # 1/2/3 written by thread, read by ROS callback
        self._detect_timer    = None

        # ── ROS infrastructure ─────────────────────────────────────────────────
        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.create_subscription(PoseStamped, '/object_pose',  self._pose_cb,  10)
        self.create_subscription(Float64,     '/object_angle', self._angle_cb, 10)
        self.create_subscription(JointState,  '/joint_states', self._joint_cb, 20)

        self.gripper_pub     = self.create_publisher(Int32MultiArray, '/gripper_control', 10)
        self.pick_active_pub = self.create_publisher(Bool, '/pick_active', 10)

        # Camera + MediaPipe start NOW — warm by the time the arm finishes picking
        self._cam_thread = threading.Thread(
            target=self._gesture_monitor_thread, daemon=True)
        self._cam_thread.start()

        self._startup_timer = self.create_timer(3.0, self._startup_home_cb)
        self.get_logger().info(
            'GesturePlaceNode ready — camera warming up. '
            'After lift: show 1/2/3 fingers to select destination, '
            'then close fist to release.')

    # ── Utility ───────────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def _angle_cb(self, msg):
        self._latest_angle = msg.data

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

    # ── Startup ───────────────────────────────────────────────────────────────
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

    # ── Object pose / detection ───────────────────────────────────────────────
    def _pose_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self._latest_x, self._latest_y, self._latest_z = x, y, z

        if self.is_moving or not self._have_joints():
            return

        now = time.time()

        if self.last_x is None:
            if self.pending_x is None:
                self.pending_x, self.pending_y, self.pending_z = x, y, z
                self.pending_start = now
                return
            if self._dist2d(x, y, self.pending_x, self.pending_y) >= RETARGET_DIST_M:
                self.pending_x, self.pending_y, self.pending_z = x, y, z
                self.pending_start = now
                return
            if (now - self.pending_start) >= STABLE_SEC:
                self._commit_pick(self.pending_x, self.pending_y, self.pending_z)
            return

        if self._dist2d(x, y, self.last_x, self.last_y) < RETARGET_DIST_M:
            self.pending_x = self.pending_y = self.pending_z = self.pending_start = None
            return
        if self.pending_x is None:
            self.pending_x, self.pending_y, self.pending_z = x, y, z
            self.pending_start = now
            return
        if self._dist2d(x, y, self.pending_x, self.pending_y) >= RETARGET_DIST_M:
            self.pending_x, self.pending_y, self.pending_z = x, y, z
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
        self.pending_x  = self.pending_y = self.pending_z = self.pending_start = None
        self._set_pick_active(True)
        self.get_logger().info(
            f'Pick committed  obj=({x:.3f},{y:.3f},{z:.3f})  '
            f'hover_z={self.hover_z:.3f}  plunge_z={self.plunge_z:.3f}  '
            f'angle={self.goal_angle:.1f}°')
        self.is_moving = True
        self._go_hover()

    # ── IK + planning ──────────────────────────────────────────────────────────
    def _ik_move(self, x, y, z, tight=False):
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            pass
        qx, qy, qz, qw = _gripper_quat(self.goal_angle)
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
        self._gesture_mode = 'idle'
        self.is_moving     = False
        self.motion_stage  = 'IDLE'
        self._set_pick_active(False)

    # ── Trajectory result → next step ─────────────────────────────────────────
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
            # Object in hand, arm at hover — ask user to select destination
            self.get_logger().info(
                'Object lifted. Show 1 / 2 / 3 fingers to choose destination.')
            self.motion_stage = 'SELECT_GESTURE'
            self._gesture_trigger.clear()
            self._gesture_mode = 'select'
            self._detect_timer = self.create_timer(0.05, self._check_gesture_trigger)

        elif stage == 'TO_DEST':
            dest = self._selected_dest
            self.get_logger().info(
                f'At {DEST_NAMES[dest]}. Open palm then CLOSE FIST to release.')
            self.motion_stage = 'RELEASE_WAIT'
            self._gesture_trigger.clear()
            self._gesture_mode = 'release'
            self._detect_timer = self.create_timer(0.05, self._check_gesture_trigger)

    # ── Pick steps ─────────────────────────────────────────────────────────────
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

    # ── Gesture trigger poll (20 Hz) ───────────────────────────────────────────
    def _check_gesture_trigger(self):
        if not self._gesture_trigger.is_set():
            return
        self._clear_detect_timer()
        self._gesture_trigger.clear()

        if self.motion_stage == 'SELECT_GESTURE':
            dest = self._selected_dest
            self._gesture_mode = 'idle'
            self.get_logger().info(
                f'Destination selected: {DEST_NAMES[dest]} — moving...')
            self.motion_stage = 'TO_DEST'
            self._plan_joints(DEST_JOINTS[dest], tight=False, vel=0.12, acc=0.10)

        elif self.motion_stage == 'RELEASE_WAIT':
            self._gesture_mode = 'idle'
            self.get_logger().info('Gesture confirmed — opening gripper.')
            self.motion_stage = 'RELEASE'
            self._gripper(0, 150, 100)
            self.sequence_timer = self.create_timer(1.0, self._go_home_after_release)

    def _go_home_after_release(self):
        self._clear_timer()
        self.last_x = None
        self.last_y = None
        self.get_logger().info('Object released. Returning home...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False, vel=0.12, acc=0.10)

    # ── Gesture monitor thread ─────────────────────────────────────────────────
    def _gesture_monitor_thread(self):
        """
        Runs for the entire node lifetime.
        Reads self._gesture_mode each frame:
          'idle'    — draw landmarks, no detection
          'select'  — detect 1/2/3 finger count → set _selected_dest + _gesture_trigger
          'release' — detect open→close fist → set _gesture_trigger
        """
        mp_hands = mediapipe_module.solutions.hands
        mp_draw  = mediapipe_module.solutions.drawing_utils

        pipeline = dai.Pipeline()
        cam = pipeline.createColorCamera()
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
        cam.setPreviewSize(CAM_WIDTH, CAM_HEIGHT)
        cam.setInterleaved(False)
        cam.setFps(30)
        xout = pipeline.createXLinkOut()
        xout.setStreamName('rgb')
        cam.preview.link(xout.input)

        # ── Per-mode tracking state (reset on each mode transition) ───────────
        prev_mode = 'idle'

        # select state
        sel_candidate = 0
        sel_start_t   = None   # time current candidate was first seen

        # release state
        rel_last_confirmed = None
        rel_candidate      = None
        rel_cnt            = 0
        rel_wave_start_t   = None

        try:
            with dai.Device(pipeline) as device, \
                 mp_hands.Hands(
                     static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.7,
                     min_tracking_confidence=0.6) as hands:

                q = device.getOutputQueue('rgb', maxSize=4, blocking=False)
                self.get_logger().info('OAK-1 Lite ready (idle — landmarks always visible).')

                while not self._gesture_stop.is_set():
                    mode = self._gesture_mode   # single read — GIL-safe

                    # Reset internal state on mode transition
                    if mode != prev_mode:
                        sel_candidate = 0;  sel_start_t = None
                        rel_last_confirmed = None;  rel_candidate = None
                        rel_cnt = 0;  rel_wave_start_t = None
                        prev_mode = mode

                    frame_data = q.tryGet()
                    if frame_data is None:
                        time.sleep(0.008)
                        continue

                    frame = frame_data.getCvFrame()

                    # ── MediaPipe — always run for live skeleton ───────────────
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = hands.process(rgb)
                    rgb.flags.writeable = True

                    lm = None
                    if results.multi_hand_landmarks:
                        lm = results.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                    # ── SELECT mode ────────────────────────────────────────────
                    if mode == 'select':
                        if lm is not None:
                            n = _count_fingers(lm)
                            if n in (1, 2, 3):
                                if n != sel_candidate:
                                    sel_candidate = n
                                    sel_start_t   = time.time()
                                elapsed = time.time() - sel_start_t
                                if elapsed >= SELECT_HOLD_SEC:
                                    self._selected_dest = sel_candidate
                                    self._gesture_trigger.set()
                                    self.get_logger().info(
                                        f'Destination {sel_candidate} confirmed.')
                                    sel_candidate = 0;  sel_start_t = None
                            else:
                                sel_candidate = 0;  sel_start_t = None
                        else:
                            sel_candidate = 0;  sel_start_t = None

                        elapsed = (time.time() - sel_start_t) if sel_start_t else 0.0
                        self._draw_select(frame, sel_candidate, elapsed)

                    # ── RELEASE mode ───────────────────────────────────────────
                    elif mode == 'release':
                        current = _classify_release(lm) if lm is not None else None

                        if current is not None:
                            if current == rel_candidate:
                                rel_cnt += 1
                            else:
                                rel_candidate = current
                                rel_cnt       = 1

                            if rel_cnt == RELEASE_HOLD and current != rel_last_confirmed:
                                prev_conf          = rel_last_confirmed
                                rel_last_confirmed = current

                                if prev_conf == 'open' and current == 'closed':
                                    self._gesture_trigger.set()
                                    self._draw_release(frame, triggered=True,
                                                       waiting=False)
                                    cv2.waitKey(1)
                                    self._gesture_mode = 'idle'
                                    prev_mode = 'idle'
                                    rel_last_confirmed = None
                                    rel_candidate = None;  rel_cnt = 0
                                    rel_wave_start_t = None
                                    continue

                                if rel_last_confirmed == 'open' and rel_wave_start_t is None:
                                    rel_wave_start_t = time.time()
                        else:
                            rel_cnt = max(0, rel_cnt - 2)

                        # Timeout: open seen but close didn't follow in time
                        if (rel_wave_start_t and rel_last_confirmed == 'open'
                                and (time.time() - rel_wave_start_t) > WAVE_WINDOW):
                            self.get_logger().info('Release gesture timeout — reset')
                            rel_last_confirmed = None
                            rel_wave_start_t   = None

                        self._draw_release(frame,
                                           triggered=False,
                                           waiting=(rel_last_confirmed == 'open'))

                    # ── IDLE mode ──────────────────────────────────────────────
                    else:
                        self._draw_idle(frame)

                    cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Gesture thread error: {exc}')

        try:
            cv2.destroyWindow('Gesture Monitor')
        except Exception:
            pass

    # ── Draw helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _banner(frame, text, col):
        """Draw a semi-transparent top banner with text."""
        h, w = frame.shape[:2]
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, 44), (15, 15, 15), -1)
        cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)
        cv2.putText(frame, text, (8, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_idle(frame):
        GesturePlaceNode._banner(frame, 'Gesture Place — waiting for pick...', (120, 120, 120))
        cv2.imshow('Gesture Monitor', frame)

    @staticmethod
    def _draw_select(frame, candidate, elapsed):
        """Show 3 destination options; highlight the one being held."""
        h, w = frame.shape[:2]

        GesturePlaceNode._banner(
            frame,
            'Show fingers to choose:  1=Handover  2=Place B  3=Place C',
            (0, 220, 255))

        # Three option boxes at the bottom
        labels = ['1\nHandover', '2\nPlace B', '3\nPlace C']
        colors = list(DEST_COLORS.values())
        box_w, box_h = 130, 64
        gap   = 20
        total = 3 * box_w + 2 * gap
        x0    = (w - total) // 2
        y0    = h - box_h - 16

        for i in range(3):
            dest  = i + 1
            bx    = x0 + i * (box_w + gap)
            active = (candidate == dest)
            fill  = colors[i] if active else (40, 40, 40)
            cv2.rectangle(frame, (bx, y0), (bx + box_w, y0 + box_h), fill, -1)
            cv2.rectangle(frame, (bx, y0), (bx + box_w, y0 + box_h), colors[i], 2)

            # Finger count number
            cv2.putText(frame, str(dest), (bx + 10, y0 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
            # Destination name
            name = DEST_NAMES[dest]
            cv2.putText(frame, name, (bx + 36, y0 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

            # Hold progress bar inside active box
            if active and elapsed > 0:
                bar_w = int((box_w - 4) * min(elapsed / SELECT_HOLD_SEC, 1.0))
                cv2.rectangle(frame,
                              (bx + 2, y0 + box_h - 10),
                              (bx + 2 + bar_w, y0 + box_h - 2),
                              (255, 255, 255), -1)

        cv2.imshow('Gesture Monitor', frame)

    @staticmethod
    def _draw_release(frame, triggered, waiting):
        if triggered:
            GesturePlaceNode._banner(frame, 'RELEASING!', (0, 255, 80))
        elif waiting:
            GesturePlaceNode._banner(frame, 'Now CLOSE your FIST!', (0, 80, 255))
        else:
            GesturePlaceNode._banner(
                frame, 'Open palm, then CLOSE FIST to release', (0, 220, 255))
        cv2.imshow('Gesture Monitor', frame)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = GesturePlaceNode()
    try:
        rclpy.spin(node)
    finally:
        node._gesture_stop.set()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
