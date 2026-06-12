"""
gesture_handover_pick_node.py

Same pick sequence as handover_pick_node, but triggers gripper release via
a single open→close fist gesture detected by OAK-1 Lite + MediaPipe Hands.

Latency design:
  - Camera and MediaPipe open ONCE at node startup (background thread stays alive).
  - Reaching handover pose flips _gesture_active — zero device-open overhead.
  - Trigger sets _gesture_trigger; gripper fires in the SAME callback, no thread join.
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
HOME_JOINTS     = [math.radians(d) for d in [77.03, -88.01, 63.03, -63.82, -88.73, 266.75]]
HANDOVER_JOINTS = [math.radians(d) for d in [78.85, -89.08, 65.74, -76.01,   2.16, 277.87]]

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
# Trigger: open palm then close fist once (open → closed transition).
# Each state needs WAVE_HOLD consecutive frames (~0.1 s) — no long hold required.
WAVE_HOLD  = 3      # frames to confirm a gesture state (~0.1 s at 30 fps)
WAVE_WINDOW = 6.0   # seconds; if open is seen but close doesn't follow, reset
CAM_WIDTH  = 640
CAM_HEIGHT = 480

_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky tip landmarks
_PIPS = [6, 10, 14, 18]   # corresponding PIP joints


def _classify_hand(landmarks):
    """Return 'open', 'closed', or None (ambiguous)."""
    extended = sum(
        1 for t, p in zip(_TIPS, _PIPS)
        if landmarks.landmark[t].y < landmarks.landmark[p].y
    )
    if extended >= 3:
        return 'open'
    if extended <= 1:
        return 'closed'
    return None


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
class GestureHandoverPickNode(Node):
    """
    Pick → handover pose → open fist once to release → home.

    The OAK-1 Lite camera and MediaPipe Hands open once at node startup in a
    background thread.  When the arm reaches the handover pose, _gesture_active
    is set and the thread starts showing the GUI and looking for the gesture —
    no depthai device open/close on the critical path.

    Gripper fires in the same timer callback that detects the trigger; no thread
    join blocks the path.
    """

    def __init__(self):
        super().__init__('gesture_handover_pick_node')

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

        # ── Gesture monitor events ─────────────────────────────────────────────
        self._gesture_active  = threading.Event()   # set when arm is at handover pose
        self._gesture_trigger = threading.Event()   # set by thread when gesture fires
        self._gesture_stop    = threading.Event()   # set on node shutdown
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

        # Start camera + MediaPipe NOW so they are warm by the time the arm picks
        self._cam_thread = threading.Thread(
            target=self._gesture_monitor_thread, daemon=True)
        self._cam_thread.start()

        self._startup_timer = self.create_timer(3.0, self._startup_home_cb)
        self.get_logger().info(
            'GestureHandoverPickNode ready — camera warming up in background. '
            'After pick: arm → handover pose; open palm then CLOSE FIST to release.')

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

    # ── IK + trajectory planning ──────────────────────────────────────────────
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
        self._gesture_active.clear()   # tell thread to go idle (non-blocking)
        self.is_moving    = False
        self.motion_stage = 'IDLE'
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
            self.get_logger().info('Lift done. Moving to handover pose...')
            self.sequence_timer = self.create_timer(0.3, self._go_handover)
        elif stage == 'TO_HANDOVER':
            self.get_logger().info(
                'At handover pose — open palm then CLOSE FIST to release.')
            self.motion_stage = 'HANDOVER_WAIT'
            # Activate gesture detection: thread is already running, just flip the flag
            self._gesture_trigger.clear()
            self._gesture_active.set()
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

    def _go_handover(self):
        self._clear_timer()
        self.get_logger().info('Carrying to handover pose...')
        self.motion_stage = 'TO_HANDOVER'
        self._plan_joints(HANDOVER_JOINTS, tight=False, vel=0.12, acc=0.10)

    # ── Gesture trigger poll (20 Hz, ROS spin thread) ─────────────────────────
    def _check_gesture_trigger(self):
        if self._gesture_trigger.is_set():
            self._clear_detect_timer()
            self._gesture_active.clear()   # tell thread to go idle instantly
            self._gesture_trigger.clear()
            # Open gripper IMMEDIATELY — no thread join in the way
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
        - Camera and MediaPipe open once here at startup.
        - IDLE when _gesture_active is clear: drains frames, keeps camera warm.
        - ACTIVE when _gesture_active is set: shows GUI, detects open→close.
        - Sets _gesture_trigger when a close-fist event is confirmed, then goes idle.
        """
        mp_hands = mediapipe_module.solutions.hands
        mp_draw  = mediapipe_module.solutions.drawing_utils

        # ── depthai pipeline ───────────────────────────────────────────────────
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

        try:
            with dai.Device(pipeline) as device, \
                 mp_hands.Hands(
                     static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.7,
                     min_tracking_confidence=0.6) as hands:

                q = device.getOutputQueue('rgb', maxSize=4, blocking=False)
                self.get_logger().info('OAK-1 Lite ready (camera warm, detection idle).')

                # Per-activation gesture state (reset each time we go active)
                last_confirmed = None
                candidate      = None
                candidate_cnt  = 0
                wave_start_t   = None
                was_active     = False

                while not self._gesture_stop.is_set():
                    active_now = self._gesture_active.is_set()

                    # Detect activation edge → reset gesture state
                    if active_now and not was_active:
                        last_confirmed = None
                        candidate      = None
                        candidate_cnt  = 0
                        wave_start_t   = None
                        self.get_logger().info('Gesture detection active.')
                    was_active = active_now

                    frame_data = q.tryGet()
                    if frame_data is None:
                        time.sleep(0.008)
                        continue

                    frame = frame_data.getCvFrame()

                    # ── MediaPipe inference — always run for live skeleton ──────
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = hands.process(rgb)
                    rgb.flags.writeable = True

                    current = None
                    if results.multi_hand_landmarks:
                        lm = results.multi_hand_landmarks[0]
                        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                        current = _classify_hand(lm)

                    if not active_now:
                        # IDLE: show skeleton but skip gesture logic
                        self._draw_frame(frame, triggered=False, waiting_for_close=False)
                        cv2.waitKey(1)
                        continue

                    # ── Debounce ───────────────────────────────────────────────
                    if current is not None:
                        if current == candidate:
                            candidate_cnt += 1
                        else:
                            candidate     = current
                            candidate_cnt = 1

                        if candidate_cnt == WAVE_HOLD and current != last_confirmed:
                            prev           = last_confirmed
                            last_confirmed = current

                            # Confirmed open → closed transition = trigger
                            if prev == 'open' and current == 'closed':
                                self._gesture_trigger.set()
                                self._draw_frame(frame, triggered=True)
                                cv2.waitKey(1)
                                # Go idle — _check_gesture_trigger fires gripper
                                self._gesture_active.clear()
                                was_active = False
                                continue

                            # Timeout: open was confirmed but close didn't come in time
                            if (last_confirmed == 'open' and wave_start_t is None):
                                wave_start_t = time.time()

                        if (wave_start_t and last_confirmed == 'open'
                                and (time.time() - wave_start_t) > WAVE_WINDOW):
                            self.get_logger().info('Gesture timeout — reset')
                            last_confirmed = None
                            wave_start_t   = None
                    else:
                        candidate_cnt = max(0, candidate_cnt - 2)
                        if (wave_start_t and last_confirmed == 'open'
                                and (time.time() - wave_start_t) > WAVE_WINDOW):
                            self.get_logger().info('Gesture timeout — reset')
                            last_confirmed = None
                            wave_start_t   = None

                    self._draw_frame(frame, triggered=False,
                                     waiting_for_close=(last_confirmed == 'open'))
                    cv2.waitKey(1)

        except Exception as exc:
            self.get_logger().error(f'Gesture monitor thread error: {exc}')

        try:
            cv2.destroyWindow('Gesture Monitor')
        except Exception:
            pass
        self.get_logger().info('Gesture monitor thread exited.')

    # ── Visualisation ──────────────────────────────────────────────────────────
    @staticmethod
    def _draw_frame(frame, triggered=False, waiting_for_close=False):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        if triggered:
            text = 'RELEASING!'
            col  = (0, 255, 80)
        elif waiting_for_close:
            text = 'Now CLOSE your FIST!'
            col  = (0, 80, 255)
        else:
            text = 'Open your palm, then close fist to release'
            col  = (0, 220, 255)

        cv2.putText(frame, text, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2, cv2.LINE_AA)
        cv2.imshow('Gesture Monitor', frame)


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = GestureHandoverPickNode()
    try:
        rclpy.spin(node)
    finally:
        node._gesture_stop.set()   # signal camera thread to exit
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
