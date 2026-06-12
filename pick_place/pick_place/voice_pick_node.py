import math
import json
import threading

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

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer

# ─────────────────────────────────────────────────────────────────────────────
# Home position  (teach-pendant degrees → radians) — same as pick_node_any_pose
# ─────────────────────────────────────────────────────────────────────────────
HOME_JOINTS = [math.radians(d) for d in [77.03, -88.01, 63.03, -63.82, -88.73, 266.75]]

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants  (metres)
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_LENGTH       = 0.27    # tool0 flange → fingertip
HOVER_GAP            = 0.15    # clearance above object top before descending
GRASP_DEPTH          = 0.05    # how far below object top the fingers close
GRIPPER_ANGLE_OFFSET = -90.0   # PCA → gripper angle offset

# ─────────────────────────────────────────────────────────────────────────────
# MoveIt tolerances
# ─────────────────────────────────────────────────────────────────────────────
LOOSE_TOLERANCE = 0.02
TIGHT_TOLERANCE = 0.003

# ─────────────────────────────────────────────────────────────────────────────
# Voice / mic settings
# ─────────────────────────────────────────────────────────────────────────────
VOSK_MODEL_PATH  = '/home/ubuntu/vosk-model-small-en-in-0.4'
MIC_DEVICE_INDEX = 24      # sounddevice index for H3-VR ZOOM mic (hw:2,0)
VOSK_RATE        = 16000   # Hz — required by Vosk
CAPTURE_RATE     = 44100   # Hz — only rate the ZOOM mic supports
BLOCK_SIZE       = 22050   # samples at CAPTURE_RATE per read  (0.5 s)

# Vosk grammar — model ONLY tries to match these phrases, nothing else.
# "[unk]" handles unrecognised audio so the decoder doesn't stall.
# "alpha" is the wake-word: every valid command must start with it.
GRAMMAR = json.dumps([
    "alpha pick red box",
    "alpha pick blue box",
    "[unk]",
])
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
class VoicePickNode(Node):
    """
    Voice-triggered pick-and-place for red and blue boxes.

    Subscribes to:
        /red_box_pose   (PoseStamped) — from voice_color_detector_rs
        /red_box_angle  (Float64)
        /blue_box_pose  (PoseStamped)
        /blue_box_angle (Float64)
        /joint_states   (JointState)

    Publishes:
        /gripper_control (Int32MultiArray) → Robotiq driver
        /pick_active     (Bool)            → detector pause signal

    Voice commands (ZOOM mic, Vosk offline STT):
        "pick red box"  → picks the red box, places it back at same position
        "pick blue box" → picks the blue box, places it back at same position

    Does NOT auto-pick on detection — only the voice trigger initiates a pick.
    If a voice command arrives while the arm is moving, it is held and executed
    once the arm returns to IDLE.

    State machine: IDLE → HOVER → PLUNGE → LIFT → PLACE_PLUNGE → PLACE_LIFT → HOME
    """

    def __init__(self):
        super().__init__('voice_pick_node')

        self.ur_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
        ]
        self.current_joint_map = {}

        self.is_moving      = True    # blocked until startup home move completes
        self.motion_stage   = 'IDLE'
        self.sequence_timer = None
        self._pending_tight = False

        self.goal_x = self.goal_y = self.goal_z = None
        self.goal_angle = 0.0
        self.hover_z = self.plunge_z = None

        # Latest known poses from detector — only updated while arm is idle
        self._red_pose   = None
        self._red_angle  = 0.0
        self._blue_pose  = None
        self._blue_angle = 0.0

        # Voice command state — thread-safe via lock
        self._voice_command = None   # 'red' | 'blue' | None
        self._cmd_lock      = threading.Lock()

        # Active trajectory goal handle — stored so result callback can clear it
        self._active_gh      = None
        # Set True by _obstacle_cb; checked before each dangerous step (plunge/place)
        self._obstacle_flag  = False

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.ik_client   = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        self.create_subscription(PoseStamped, '/red_box_pose',      self._red_pose_cb,   10)
        self.create_subscription(Float64,     '/red_box_angle',     self._red_angle_cb,  10)
        self.create_subscription(PoseStamped, '/blue_box_pose',     self._blue_pose_cb,  10)
        self.create_subscription(Float64,     '/blue_box_angle',    self._blue_angle_cb, 10)
        self.create_subscription(JointState,  '/joint_states',      self._joint_cb,      20)
        self.create_subscription(Bool,        '/obstacle_near_tcp', self._obstacle_cb,   10)

        self.gripper_pub     = self.create_publisher(Int32MultiArray, '/gripper_control', 10)
        self.pick_active_pub = self.create_publisher(Bool, '/pick_active', 10)

        # 3 s delay before home move — lets MoveIt finish loading
        self._startup_timer    = self.create_timer(3.0, self._startup_home_cb)
        # 10 Hz poll to act on queued voice commands
        self._voice_check_timer = self.create_timer(0.1, self._check_voice_cb)

        # Vosk voice listener in a daemon thread
        t = threading.Thread(target=self._run_voice_listener, daemon=True)
        t.start()

        self.get_logger().info(
            'Voice Pick Node ready. '
            f'Mic: device {MIC_DEVICE_INDEX} (H3-VR ZOOM). '
            'Moving to home first — then say "pick red box" or "pick blue box".')

    # ─────────────────────────────────────────────────────────────────────────
    # Subscriptions
    # ─────────────────────────────────────────────────────────────────────────
    def _joint_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_map[name] = pos

    def _red_pose_cb(self, msg):
        if not self.is_moving:
            self._red_pose = msg

    def _red_angle_cb(self, msg):
        if not self.is_moving:
            self._red_angle = msg.data

    def _blue_pose_cb(self, msg):
        if not self.is_moving:
            self._blue_pose = msg

    def _blue_angle_cb(self, msg):
        if not self.is_moving:
            self._blue_angle = msg.data

    def _have_joints(self):
        return all(n in self.current_joint_map for n in self.ur_joint_names)

    # ─────────────────────────────────────────────────────────────────────────
    # Gripper / utility
    # ─────────────────────────────────────────────────────────────────────────
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

    def _unlock(self):
        self.is_moving    = False
        self.motion_stage = 'IDLE'
        self._set_pick_active(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Voice listener  (runs in a background daemon thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_voice_listener(self):
        try:
            model = Model(VOSK_MODEL_PATH)
            # Grammar-restricted recogniser — only decodes the listed phrases.
            # This eliminates almost all false positives from background noise/speech.
            rec = KaldiRecognizer(model, VOSK_RATE, GRAMMAR)
        except Exception as e:
            self.get_logger().error(f'Vosk model load failed: {e}')
            return

        # ZOOM mic only supports 44100/48000 Hz — capture at 44100, resample to 16000
        try:
            stream = sd.RawInputStream(
                samplerate=CAPTURE_RATE, blocksize=BLOCK_SIZE,
                device=MIC_DEVICE_INDEX, dtype='int16', channels=1)
        except Exception as e:
            self.get_logger().error(
                f'Cannot open mic (device {MIC_DEVICE_INDEX}): {e}')
            return

        self.get_logger().info(
            f'Mic open at {CAPTURE_RATE} Hz → resampled to {VOSK_RATE} Hz for Vosk. '
            'Grammar-restricted. Say "alpha pick red box" or "alpha pick blue box".')
        with stream:
            while rclpy.ok():
                try:
                    data, _ = stream.read(BLOCK_SIZE)
                    # Resample 44100 → 16000 Hz  (ratio 160/441)
                    pcm = np.frombuffer(bytes(data), dtype=np.int16).astype(np.float32)
                    pcm = resample_poly(pcm, 160, 441)
                    pcm16 = pcm.astype(np.int16).tobytes()

                    if not rec.AcceptWaveform(pcm16):
                        continue
                    text = json.loads(rec.Result()).get('text', '').lower().strip()
                    if not text or text == '[unk]':
                        continue

                    self.get_logger().info(f'[Voice] heard: "{text}"')

                    # Wake-word guard: "alpha" must be present (second defence layer)
                    if 'alpha' not in text:
                        self.get_logger().info('[Voice] ignored — no wake word "alpha"')
                        continue

                    with self._cmd_lock:
                        if self._voice_command is None:
                            if 'red' in text:
                                self._voice_command = 'red'
                                self.get_logger().info('[Voice] → PICK RED queued')
                            elif 'blue' in text:
                                self._voice_command = 'blue'
                                self.get_logger().info('[Voice] → PICK BLUE queued')
                except Exception as e:
                    self.get_logger().warn(f'Voice stream error: {e}')

    # ─────────────────────────────────────────────────────────────────────────
    # Voice command dispatch  (10 Hz timer on ROS2 executor thread)
    # ─────────────────────────────────────────────────────────────────────────
    def _check_voice_cb(self):
        with self._cmd_lock:
            cmd = self._voice_command

        if cmd is None or self.is_moving:
            return   # arm busy or no command — hold and retry

        if cmd == 'red':
            if self._red_pose is None:
                self.get_logger().warn(
                    '[Voice] RED command held — no red box visible yet, waiting...')
                return
            with self._cmd_lock:
                self._voice_command = None
            p = self._red_pose.pose.position
            angle = (self._red_angle + GRIPPER_ANGLE_OFFSET) % 180.0
            self._commit_pick(p.x, p.y, p.z, angle, 'RED')

        elif cmd == 'blue':
            if self._blue_pose is None:
                self.get_logger().warn(
                    '[Voice] BLUE command held — no blue box visible yet, waiting...')
                return
            with self._cmd_lock:
                self._voice_command = None
            p = self._blue_pose.pose.position
            angle = (self._blue_angle + GRIPPER_ANGLE_OFFSET) % 180.0
            self._commit_pick(p.x, p.y, p.z, angle, 'BLUE')

    # ─────────────────────────────────────────────────────────────────────────
    # Startup home
    # ─────────────────────────────────────────────────────────────────────────
    def _startup_home_cb(self):
        self.destroy_timer(self._startup_timer)
        if not self._have_joints():
            self._startup_timer = self.create_timer(1.0, self._startup_home_cb)
            self.get_logger().info('Waiting for joint states...')
            return
        self._set_pick_active(True)
        self.get_logger().info('Moving to home position...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Commit pick
    # ─────────────────────────────────────────────────────────────────────────
    def _commit_pick(self, x, y, z, angle_deg, color):
        self.goal_x     = x
        self.goal_y     = y
        self.goal_z     = z
        self.goal_angle = angle_deg
        self.hover_z    = z + GRIPPER_LENGTH + HOVER_GAP
        self.plunge_z   = z + GRIPPER_LENGTH - GRASP_DEPTH

        self.get_logger().info(
            f'[{color}] Pick committed  obj=({x:.3f},{y:.3f},{z:.3f})  '
            f'hover_z={self.hover_z:.3f}  plunge_z={self.plunge_z:.3f}  '
            f'angle={angle_deg:.1f}deg')

        self.is_moving = True
        self._set_pick_active(True)
        self._go_hover()

    # ─────────────────────────────────────────────────────────────────────────
    # MoveIt helpers  (identical pattern to pick_node_any_pose)
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
            self._plan_joints([ik_map[n] for n in self.ur_joint_names],
                              tight=self._pending_tight)
        except KeyError:
            self.get_logger().error('IK solution missing joint. Unlocking.')
            self._unlock()

    def _plan_joints(self, goal_positions, tight=False):
        tol = TIGHT_TOLERANCE if tight else LOOSE_TOLERANCE
        req = GetMotionPlan.Request()
        mp  = req.motion_plan_request
        mp.group_name                      = 'ur_manipulator'
        mp.num_planning_attempts           = 5 if tight else 3
        mp.allowed_planning_time           = 3.0 if tight else 2.5
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
            self.get_logger().error(
                f'Trajectory rejected at {self.motion_stage}. Unlocking.')
            self._active_gh = None
            self._unlock()
            return
        self._active_gh = gh
        gh.get_result_async().add_done_callback(self._result_cb)

    # ─────────────────────────────────────────────────────────────────────────
    # Obstacle flag — set during arm pauses when DBSCAN detects something near TCP
    # ─────────────────────────────────────────────────────────────────────────
    def _obstacle_cb(self, msg: Bool):
        if msg.data and self.is_moving:
            self._obstacle_flag = True
            self.get_logger().warn('Obstacle near TCP flagged — will abort before next step.')

    def _check_abort(self) -> bool:
        """Call at the start of each dangerous step. Returns True and aborts if obstacle flagged."""
        if not self._obstacle_flag:
            return False
        self._obstacle_flag = False
        with self._cmd_lock:
            self._voice_command = None
        self.get_logger().warn('OBSTACLE — aborting pick, returning home!')
        self._clear_timer()
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # State machine callbacks
    # ─────────────────────────────────────────────────────────────────────────
    def _result_cb(self, future):
        self._active_gh = None   # trajectory finished — clear handle
        stage = self.motion_stage
        if stage == 'HOME':
            self._obstacle_flag = False   # clear any stale obstacle from previous cycle
            self.get_logger().info('At home. Listening for voice commands.')
            self._unlock()
        elif stage == 'HOVER':
            self.get_logger().info('Hover reached. Opening gripper...')
            self.sequence_timer = self.create_timer(0.8, self._open_then_plunge)
        elif stage == 'PLUNGE':
            self.get_logger().info('Plunge done. Closing gripper...')
            self.sequence_timer = self.create_timer(0.5, self._close_then_lift)
        elif stage == 'LIFT':
            self.get_logger().info('Lift done. Descending to place...')
            self.sequence_timer = self.create_timer(0.8, self._go_place_plunge)
        elif stage == 'PLACE_PLUNGE':
            self.get_logger().info('At place depth. Opening gripper...')
            self.sequence_timer = self.create_timer(0.5, self._open_then_retract)
        elif stage == 'PLACE_LIFT':
            self.get_logger().info('Retracted. Returning home...')
            self.sequence_timer = self.create_timer(1.0, self._finish_cycle)

    # ── Steps ─────────────────────────────────────────────────────────────────
    def _go_hover(self):
        self.get_logger().info(f'Moving to hover  z={self.hover_z:.3f} m')
        self.motion_stage = 'HOVER'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _open_then_plunge(self):
        self._clear_timer()
        if self._check_abort():
            return
        self._gripper(0, 150, 100)
        self.sequence_timer = self.create_timer(1.0, self._do_plunge)

    def _do_plunge(self):
        self._clear_timer()
        if self._check_abort():
            return
        self.get_logger().info(f'Plunging  z={self.plunge_z:.3f} m  (tight)')
        self.motion_stage = 'PLUNGE'
        self._ik_move(self.goal_x, self.goal_y, self.plunge_z, tight=True)

    def _close_then_lift(self):
        self._clear_timer()
        self._gripper(255, 150, 50)
        self.sequence_timer = self.create_timer(1.5, self._do_lift)

    def _do_lift(self):
        self._clear_timer()
        self.get_logger().info(f'Lifting  z={self.hover_z:.3f} m')
        self.motion_stage = 'LIFT'
        self._ik_move(self.goal_x, self.goal_y, self.hover_z, tight=False)

    def _go_place_plunge(self):
        self._clear_timer()
        if self._check_abort():
            return
        self.get_logger().info(f'Descending to place  z={self.plunge_z:.3f} m  (tight)')
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
        self.get_logger().info('Cycle complete. Returning home...')
        self.motion_stage = 'HOME'
        self._plan_joints(HOME_JOINTS, tight=False)


def main(args=None):
    rclpy.init(args=args)
    node = VoicePickNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
