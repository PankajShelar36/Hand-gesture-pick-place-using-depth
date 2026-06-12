#!/usr/bin/env python3
"""
pointcloud_collision_monitor.py

Dynamic obstacle avoidance for UR10 using RealSense D435i.

4-state machine:
  INIT_BG   — collect BG_FRAMES depth frames at startup (arm at HOME, table empty)
  DETECTING — run obstacle detection, update MoveIt2 planning scene
  ARM_MOVING — arm in motion: detection paused, scene frozen
  REBASING  — arm just stopped: collect REBASE_FRAMES to re-baseline background
               (absorbs arm body at new position so it stops appearing as foreground)

Why rebasing matters:
  Background is captured with the arm at HOME. When the arm moves to HOVER or LIFT,
  its links appear closer than the background → detected as foreground → added as
  collision objects → MoveIt2 blocks further planning (PLACE_PLUNGE rejected, next
  HOVER rejected). Rebasing after each stop fixes this.

RViz displays to add:
  PointCloud2 → /collision_monitor/foreground_cloud  (Fixed Frame: base)
  MarkerArray → /collision_monitor/obstacle_markers
"""

import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import numpy as np

from std_msgs.msg import Bool, Int32
from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion, Vector3
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from sklearn.cluster import DBSCAN

# ─── Tuning ──────────────────────────────────────────────────────────────────
BG_FRAMES     = 30     # initial background frames (median)
BG_THRESH_MM  = 50     # foreground if bg − cur > 50 mm
REBASE_FRAMES = 4      # frames to re-baseline after arm stops (~0.5 s at 8 Hz)
SETTLE_SEC    = 0.3    # seconds after arm stops before rebasing starts
VOXEL_M       = 0.025  # 2.5 cm voxel grid
DBSCAN_EPS    = 0.06   # 6 cm cluster radius
DBSCAN_MIN    = 20     # min voxels per cluster
OBS_PADDING   = 0.06   # collision box half-size padding (m)
PICK_EXCL_R   = 0.25   # exclude clusters within 25 cm of pick target centroid
SAFETY_R      = 0.15   # TCP proximity warning radius (m) — DBSCAN cluster check
ARM_VEL_THR   = 0.008  # rad/s — above this = arm moving
PROCESS_HZ    = 8.0
MARKER_LIFE   = 0.4    # RViz marker lifetime (s)
IDLE_THRESHOLD = 2.0   # seconds arm must be idle before scene is updated.
                       # All pick-cycle inter-step pauses are <2 s, so the scene
                       # is never written mid-cycle. HOME wait is 3+ s, so obstacles
                       # are captured before the next pick is committed.

# Workspace filter in base frame (metres)
Z_MIN, Z_MAX = 0.05, 0.90
X_MIN, X_MAX = -0.70, 0.70
Y_MIN, Y_MAX = -1.15, 0.10

# ROS2 / MoveIt2
DEPTH_TOPIC = '/camera/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC  = '/camera/camera/color/camera_info'
SCENE_TOPIC = '/planning_scene'
CAM_FRAME   = 'camera_color_optical_frame'
BASE_FRAME  = 'base'
OBS_PREFIX  = 'dyn_obs_'
MAX_OBS     = 8

# State identifiers
ST_INIT    = 'INIT_BG'
ST_DETECT  = 'DETECTING'
ST_MOVING  = 'ARM_MOVING'
ST_REBASE  = 'REBASING'


class PointCloudCollisionMonitor(Node):

    def __init__(self):
        super().__init__('pointcloud_collision_monitor')

        self._bridge      = CvBridge()
        self._tf_buf      = tf2_ros.Buffer()
        tf2_ros.TransformListener(self._tf_buf, self)
        self._lock        = threading.Lock()

        # Depth / background
        self._depth_latest = None
        self._bg_model     = None      # uint16 H×W background
        self._accum_frames = []        # frame accumulator for INIT and REBASE states

        # State machine
        self._state       = ST_INIT
        self._arm_stop_ts = None       # timestamp when arm last stopped (for settle)
        self._startup_done = False

        # Sensor / TF
        self._cam2base   = None
        self._intrinsics = None        # (fx, fy, cx, cy)

        # External signals
        self._pick_targets = {}        # {'red': np.array, 'blue': np.array} — excluded from obstacles
        self._pick_active  = False     # True while voice_pick_node is executing a pick cycle
        self._arm_moving   = False

        # Planning scene tracking
        self._active_slots = {}        # slot_id → centroid np.array
        self._idle_since   = time.time()   # when arm last entered idle state

        # ── Subscribers ──────────────────────────────────────────────────────
        self.create_subscription(Image,       DEPTH_TOPIC,        self._depth_cb,        5)
        self.create_subscription(CameraInfo,  INFO_TOPIC,         self._caminfo_cb,      1)
        self.create_subscription(JointState,  '/joint_states',    self._joints_cb,      10)
        self.create_subscription(Bool,        '/pick_active',     self._pick_active_cb, 10)
        # Voice pipeline: track both boxes so neither is ever added as an obstacle
        self.create_subscription(PoseStamped, '/red_box_pose',    self._red_pose_cb,    10)
        self.create_subscription(PoseStamped, '/blue_box_pose',   self._blue_pose_cb,   10)

        # ── Publishers (all prefixed pub_) ───────────────────────────────────
        self.pub_detected  = self.create_publisher(Bool,          '/obstacle_detected',                   10)
        self.pub_count     = self.create_publisher(Int32,         '/obstacle_count',                      10)
        self.pub_near_tcp  = self.create_publisher(Bool,          '/obstacle_near_tcp',                   10)
        self.pub_scene     = self.create_publisher(PlanningScene, SCENE_TOPIC,                             5)
        self.pub_fgcloud   = self.create_publisher(PointCloud2,   '/collision_monitor/foreground_cloud',  5)
        self.pub_markers   = self.create_publisher(MarkerArray,   '/collision_monitor/obstacle_markers',  5)

        self.create_timer(1.0 / PROCESS_HZ, self._process)

        self.get_logger().info(
            f'PointCloudCollisionMonitor started  state={ST_INIT}\n'
            '  RViz: PointCloud2 → /collision_monitor/foreground_cloud  (Fixed Frame: base)\n'
            '        MarkerArray → /collision_monitor/obstacle_markers\n'
            '  Keep table CLEAR and arm at HOME until "Background model ready" log.')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _depth_cb(self, msg: Image):
        arr = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        with self._lock:
            self._depth_latest = np.array(arr, dtype=np.uint16)

    def _caminfo_cb(self, msg: CameraInfo):
        if self._intrinsics is None:
            self._intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info(
                f'Intrinsics cached: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} '
                f'cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}')

    def _joints_cb(self, msg: JointState):
        v = np.abs(msg.velocity) if msg.velocity else np.array([])
        self._arm_moving = bool(v.size > 0 and v.max() > ARM_VEL_THR)

    def _pick_active_cb(self, msg: Bool):
        self._pick_active = msg.data

    def _red_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._pick_targets['red'] = np.array([p.x, p.y, p.z])

    def _blue_pose_cb(self, msg: PoseStamped):
        p = msg.pose.position
        self._pick_targets['blue'] = np.array([p.x, p.y, p.z])

    # ── Main loop (PROCESS_HZ) ────────────────────────────────────────────────

    def _process(self):
        # One-time startup: purge any stale dyn_obs_* from a previous session
        if not self._startup_done:
            self._startup_done = True
            self._purge_all_slots()
            self.get_logger().info('Startup: cleared stale dyn_obs_* from MoveIt2 scene.')

        if self._intrinsics is None:
            return

        if self._cam2base is None:
            self._cam2base = self._lookup_tf(BASE_FRAME, CAM_FRAME)
            if self._cam2base is None:
                return

        with self._lock:
            if self._depth_latest is None:
                return
            depth = self._depth_latest.copy()

        # ── INIT_BG ───────────────────────────────────────────────────────────
        if self._state == ST_INIT:
            self._accum_frames.append(depth)
            n = len(self._accum_frames)
            if n % 10 == 0:
                self.get_logger().info(f'Background capture: {n}/{BG_FRAMES} frames')
            if n >= BG_FRAMES:
                self._bg_model = self._median_bg(self._accum_frames)
                self._accum_frames.clear()
                self._state = ST_DETECT
                self._idle_since = time.time()
                self.get_logger().info(
                    'Background model ready — place objects or test obstacle avoidance.')
            return

        # ── ARM_MOVING ────────────────────────────────────────────────────────
        if self._state == ST_MOVING:
            if self._arm_moving:
                self._arm_stop_ts = None        # still moving — reset settle timer
            else:
                if self._arm_stop_ts is None:
                    self._arm_stop_ts = time.time()
                if (time.time() - self._arm_stop_ts) >= SETTLE_SEC:
                    # Arm settled — start rebasing background
                    self._state = ST_REBASE
                    self._accum_frames.clear()
                    self._arm_stop_ts = None
                    self.get_logger().info(
                        f'Arm settled — rebasing background ({REBASE_FRAMES} frames)...')
            self._publish_idle()
            return

        # ── REBASING ──────────────────────────────────────────────────────────
        if self._state == ST_REBASE:
            if self._arm_moving:
                # Arm moved again mid-rebase — abort, go back to MOVING
                self._state = ST_MOVING
                self._accum_frames.clear()
                self._clear_scene()
                self._publish_idle()
                return
            self._accum_frames.append(depth)
            if len(self._accum_frames) >= REBASE_FRAMES:
                self._bg_model = self._median_bg(self._accum_frames)
                self._accum_frames.clear()
                self._state = ST_DETECT
                self._idle_since = time.time()   # start idle clock from here
                self.get_logger().info(
                    'Background rebased — obstacle detection active at new arm position.')
            self._publish_idle()
            return

        # ── DETECTING ─────────────────────────────────────────────────────────
        if self._arm_moving:
            # Arm just started moving — freeze the scene and switch state
            self._state = ST_MOVING
            self._arm_stop_ts = None
            self._clear_scene()
            self._publish_idle()
            return

        self._detect_and_update(depth)

    # ── Detection pipeline ────────────────────────────────────────────────────

    def _detect_and_update(self, depth: np.ndarray):
        bg      = self._bg_model.astype(np.int32)
        cur     = depth.astype(np.int32)
        fg_mask = (bg > 0) & (cur > 0) & ((bg - cur) > BG_THRESH_MM)

        if not fg_mask.any():
            self._clear_scene()
            self._emit_markers([], near_tcp=False)
            self.pub_fgcloud.publish(self._make_cloud(np.empty((0, 3))))
            return

        # Back-project to 3-D in base frame
        rows, cols = np.where(fg_mask)
        z_m = cur[rows, cols] / 1000.0
        fx, fy, cx, cy = self._intrinsics
        pts_cam = np.column_stack([
            (cols - cx) * z_m / fx,
            (rows - cy) * z_m / fy,
            z_m,
            np.ones(len(z_m)),
        ])
        pts_base = (self._cam2base @ pts_cam.T).T[:, :3]

        # Workspace filter
        m = (
            (pts_base[:, 2] >= Z_MIN) & (pts_base[:, 2] <= Z_MAX) &
            (pts_base[:, 0] >= X_MIN) & (pts_base[:, 0] <= X_MAX) &
            (pts_base[:, 1] >= Y_MIN) & (pts_base[:, 1] <= Y_MAX)
        )
        pts_ws = pts_base[m]
        self.pub_fgcloud.publish(self._make_cloud(pts_ws))

        if len(pts_ws) < DBSCAN_MIN:
            self._clear_scene()
            self._emit_markers([], near_tcp=False)
            return

        # Voxel downsample
        vox = np.round(pts_ws / VOXEL_M).astype(np.int32)
        _, uniq = np.unique(vox, axis=0, return_index=True)
        pts_ds = pts_ws[uniq]

        # DBSCAN cluster
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN, n_jobs=1).fit_predict(pts_ds)
        cluster_ids = set(labels) - {-1}
        if not cluster_ids:
            self._clear_scene()
            self._emit_markers([], near_tcp=False)
            return

        # Build cluster list — exclude clusters near either pick target (red or blue box).
        # Both boxes are always excluded: they are pick candidates, never obstacles.
        clusters = []
        for lbl in cluster_ids:
            pts_cl   = pts_ds[labels == lbl]
            centroid = pts_cl.mean(axis=0)
            if any(np.linalg.norm(centroid - t) < PICK_EXCL_R
                   for t in self._pick_targets.values()):
                continue
            lo = pts_cl.min(axis=0) - OBS_PADDING
            hi = pts_cl.max(axis=0) + OBS_PADDING
            clusters.append({'centroid': centroid,
                             'centre':   (lo + hi) / 2.0,
                             'size':     hi - lo})
            if len(clusters) >= MAX_OBS:
                break

        if not clusters:
            self._clear_scene()
            self._emit_markers([], near_tcp=False)
            return

        idle_sec = time.time() - self._idle_since
        # Gate scene updates: arm must be idle ≥ IDLE_THRESHOLD AND pick_active must be False.
        # pick_active=True means voice_pick_node is mid-cycle — never write scene then.
        if idle_sec >= IDLE_THRESHOLD and not self._pick_active:
            self._update_scene(clusters)
        # else: hold the current scene; obstacles committed on next full idle at HOME.

        near_tcp = self._check_tcp_proximity(clusters)
        self.pub_near_tcp.publish(Bool(data=near_tcp))
        if near_tcp:
            self.get_logger().warn(
                f'OBSTACLE within {SAFETY_R} m of tool0 TCP! Use E-stop if needed.')

        self._emit_markers(clusters, near_tcp)
        self.pub_detected.publish(Bool(data=True))
        self.pub_count.publish(Int32(data=len(clusters)))

    # ── RViz markers ──────────────────────────────────────────────────────────

    def _emit_markers(self, clusters, near_tcp: bool):
        ma    = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        life  = Duration(seconds=MARKER_LIFE).to_msg()

        # Delete all previous markers in each namespace
        for ns in ('obs_box', 'obs_centroid', 'tcp_safety'):
            d = Marker()
            d.header.frame_id = BASE_FRAME
            d.header.stamp    = stamp
            d.ns = ns; d.action = Marker.DELETEALL
            ma.markers.append(d)

        for i, cl in enumerate(clusters):
            c, sz = cl['centre'], cl['size']

            # Orange transparent box — matches the MoveIt2 CollisionObject
            box = Marker()
            box.header.frame_id = BASE_FRAME; box.header.stamp = stamp
            box.ns = 'obs_box'; box.id = i; box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position    = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            box.pose.orientation = Quaternion(w=1.0)
            box.scale = Vector3(x=float(max(sz[0], 0.05)),
                                y=float(max(sz[1], 0.05)),
                                z=float(max(sz[2], 0.05)))
            box.color.r = 1.0; box.color.g = 0.45; box.color.b = 0.0; box.color.a = 0.45
            box.lifetime = life
            ma.markers.append(box)

            # Red centroid sphere
            cen = Marker()
            cen.header.frame_id = BASE_FRAME; cen.header.stamp = stamp
            cen.ns = 'obs_centroid'; cen.id = i; cen.type = Marker.SPHERE
            cen.action = Marker.ADD
            cen.pose.position    = Point(x=float(cl['centroid'][0]),
                                         y=float(cl['centroid'][1]),
                                         z=float(cl['centroid'][2]))
            cen.pose.orientation = Quaternion(w=1.0)
            cen.scale = Vector3(x=0.05, y=0.05, z=0.05)
            cen.color.r = 1.0; cen.color.g = 0.0; cen.color.b = 0.0; cen.color.a = 1.0
            cen.lifetime = life
            ma.markers.append(cen)

        # TCP safety sphere — green normally, red when obstacle is near
        tcp = self._get_tcp_base()
        if tcp is not None:
            sph = Marker()
            sph.header.frame_id = BASE_FRAME; sph.header.stamp = stamp
            sph.ns = 'tcp_safety'; sph.id = 0; sph.type = Marker.SPHERE
            sph.action = Marker.ADD
            sph.pose.position    = Point(x=float(tcp[0]), y=float(tcp[1]), z=float(tcp[2]))
            sph.pose.orientation = Quaternion(w=1.0)
            r = SAFETY_R * 2
            sph.scale = Vector3(x=r, y=r, z=r)
            if near_tcp:
                sph.color.r = 1.0; sph.color.g = 0.0; sph.color.b = 0.0; sph.color.a = 0.40
            else:
                sph.color.r = 0.0; sph.color.g = 1.0; sph.color.b = 0.0; sph.color.a = 0.12
            sph.lifetime = life
            ma.markers.append(sph)

        self.pub_markers.publish(ma)

    def _publish_idle(self):
        """Publish empty cloud and markers (used in MOVING and REBASING states)."""
        self.pub_fgcloud.publish(self._make_cloud(np.empty((0, 3))))
        ma    = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for ns in ('obs_box', 'obs_centroid', 'tcp_safety'):
            d = Marker()
            d.header.frame_id = BASE_FRAME; d.header.stamp = stamp
            d.ns = ns; d.action = Marker.DELETEALL
            ma.markers.append(d)
        self.pub_markers.publish(ma)
        self.pub_detected.publish(Bool(data=False))
        self.pub_count.publish(Int32(data=0))
        self.pub_near_tcp.publish(Bool(data=False))

    # ── MoveIt2 planning scene ─────────────────────────────────────────────────

    def _update_scene(self, clusters):
        scene = PlanningScene()
        scene.is_diff = True
        new_slots = set()
        for i, cl in enumerate(clusters):
            new_slots.add(i)
            scene.world.collision_objects.append(
                self._make_col_obj(i, cl['centre'], cl['size']))
            self._active_slots[i] = cl['centroid']
        for slot in list(self._active_slots):
            if slot not in new_slots:
                scene.world.collision_objects.append(
                    self._make_col_obj(slot, None, None, CollisionObject.REMOVE))
                del self._active_slots[slot]
        self.pub_scene.publish(scene)

    def _clear_scene(self):
        if self._active_slots:
            scene = PlanningScene()
            scene.is_diff = True
            for slot in list(self._active_slots):
                scene.world.collision_objects.append(
                    self._make_col_obj(slot, None, None, CollisionObject.REMOVE))
            self._active_slots.clear()
            self.pub_scene.publish(scene)
            self.get_logger().info('Dynamic obstacles cleared from planning scene.')
        self.pub_detected.publish(Bool(data=False))
        self.pub_count.publish(Int32(data=0))
        self.pub_near_tcp.publish(Bool(data=False))

    def _purge_all_slots(self):
        """Remove all possible dyn_obs_* from MoveIt2 — call at startup and shutdown."""
        scene = PlanningScene()
        scene.is_diff = True
        for slot in range(MAX_OBS):
            rm = CollisionObject()
            rm.header.frame_id = BASE_FRAME
            rm.id = f'{OBS_PREFIX}{slot}'
            rm.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(rm)
        self._active_slots.clear()
        self.pub_scene.publish(scene)

    def _make_col_obj(self, slot, centre, size, op=CollisionObject.ADD):
        obj = CollisionObject()
        obj.header.frame_id = BASE_FRAME
        obj.id = f'{OBS_PREFIX}{slot}'
        obj.operation = op
        if op == CollisionObject.ADD:
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [
                float(max(size[0], 0.05)),
                float(max(size[1], 0.05)),
                float(max(size[2], 0.05)),
            ]
            pose = Pose()
            pose.position    = Point(x=float(centre[0]),
                                     y=float(centre[1]),
                                     z=float(centre[2]))
            pose.orientation = Quaternion(w=1.0)
            obj.primitives      = [prim]
            obj.primitive_poses = [pose]
        return obj

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _median_bg(frames):
        return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint16)

    def _lookup_tf(self, target: str, source: str):
        try:
            tr = self._tf_buf.lookup_transform(
                target, source, rclpy.time.Time(), Duration(seconds=4.0))
            t = tr.transform.translation
            r = tr.transform.rotation
            R = Rotation.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = [t.x, t.y, t.z]
            return T
        except Exception as e:
            self.get_logger().warn(f'TF {source}→{target}: {e}')
            return None

    def _get_tcp_base(self):
        try:
            tr = self._tf_buf.lookup_transform(BASE_FRAME, 'tool0', rclpy.time.Time())
            t  = tr.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:
            return None

    def _check_tcp_proximity(self, clusters) -> bool:
        tcp = self._get_tcp_base()
        if tcp is None:
            return False
        return any(np.linalg.norm(cl['centroid'] - tcp) < SAFETY_R for cl in clusters)

    def _make_cloud(self, pts: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.height      = 1
        msg.width       = len(pts)
        msg.fields      = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step   = 12
        msg.row_step     = 12 * len(pts)
        msg.is_dense     = True
        msg.data         = pts.astype(np.float32).tobytes()
        return msg


def main():
    rclpy.init()
    node = PointCloudCollisionMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutdown: clearing all dynamic obstacles from MoveIt2...')
        node._purge_all_slots()
        rclpy.spin_once(node, timeout_sec=0.5)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
