import math
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64, Bool
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException
import tf2_geometry_msgs
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────
MIN_AREA        = 1000   # px² — ignore small blobs
DEPTH_PATCH     = 9      # NxN median depth patch at centroid
MIN_DEPTH_M     = 0.10   # discard readings closer than 10 cm
MAX_DEPTH_M     = 3.00   # discard readings farther than 3 m
BG_FRAMES       = 30     # depth frames to median-average for background model
BG_THRESH_M     = 0.05   # foreground if depth < background - 5 cm (was 3 cm)
ANGLE_SMOOTH_N  = 10     # circular-mean smoothing window for angle
# ─────────────────────────────────────────────────────────────────────────────


class AnyPoseDetectorRS(Node):
    """
    Detects any object placed on a static surface using RealSense D435i.

    Background model: median of first BG_FRAMES depth frames captured at startup.
    Start this node with the workspace CLEAR (no objects on the table) so the
    background captures only the empty table/floor.

    For each frame:
      1. Foreground mask: pixels where depth < background - BG_THRESH_M
      2. Morphological cleanup → largest contour above MIN_AREA
      3. PCA on contour points → major axis angle in [0°, 180°)
      4. Median depth at centroid → back-project to camera frame → TF to base
      5. Publish /object_pose (PoseStamped) and /object_angle (Float64, degrees)
    """

    def __init__(self):
        super().__init__('any_pose_detector_rs')
        self.bridge = CvBridge()

        self.fx = self.fy = self.ppx = self.ppy = None
        self.intrinsics_ok = False

        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._depth_img  = None   # latest depth image (float32, metres)
        self._bg_frames  = []     # accumulation buffer during background capture
        self._bg_model   = None   # background depth image (float32, metres)
        self._bg_ready   = False

        self._pick_active   = False                        # set True by pick node during motion
        self._angle_buf     = deque(maxlen=ANGLE_SMOOTH_N) # circular smoothing buffer

        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_cb, 10)
        self.create_subscription(
            Bool, '/pick_active', self._pick_active_cb, 10)

        self.pose_pub  = self.create_publisher(PoseStamped, '/object_pose',  10)
        self.angle_pub = self.create_publisher(Float64,     '/object_angle', 10)

        self.get_logger().info(
            f'Any-Pose Detector ready. '
            f'Keep workspace CLEAR — capturing background ({BG_FRAMES} frames)...')

    # ─────────────────────────────────────────────────────────────────────────
    def _info_cb(self, msg):
        if not self.intrinsics_ok:
            self.fx  = msg.k[0];  self.ppx = msg.k[2]
            self.fy  = msg.k[4];  self.ppy = msg.k[5]
            self.intrinsics_ok = True
            self.get_logger().info(
                f'Intrinsics locked: fx={self.fx:.2f} fy={self.fy:.2f} '
                f'ppx={self.ppx:.2f} ppy={self.ppy:.2f}')

    def _depth_cb(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self._depth_img = raw.astype(np.float32) / 1000.0   # mm → m

        if self._bg_ready:
            return

        self._bg_frames.append(self._depth_img.copy())
        n = len(self._bg_frames)
        if n >= BG_FRAMES:
            stack = np.stack(self._bg_frames, axis=0)
            self._bg_model = np.median(stack, axis=0).astype(np.float32)
            self._bg_ready = True
            self._bg_frames = []
            self.get_logger().info(
                f'Background model ready ({BG_FRAMES} frames). '
                'You can now place objects.')
        elif n % 10 == 0:
            self.get_logger().info(f'Background: {n}/{BG_FRAMES} frames...')

    # ─────────────────────────────────────────────────────────────────────────
    def _pick_active_cb(self, msg):
        self._pick_active = msg.data

    def _median_depth(self, cx, cy):
        h, w = self._depth_img.shape
        half = DEPTH_PATCH // 2
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        patch = self._depth_img[y0:y1, x0:x1]
        valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
        return float(np.median(valid)) if valid.size > 0 else None

    def _pca_angle(self, contour):
        """PCA on convex hull points → major-axis angle in [0°, 180°).
        Using convex hull makes the shape noise-resistant."""
        hull = cv2.convexHull(contour)
        pts  = hull.reshape(-1, 2).astype(np.float32)
        _, eigvecs = cv2.PCACompute(pts, mean=None)
        angle_rad = np.arctan2(float(eigvecs[0, 1]), float(eigvecs[0, 0]))
        return float(np.degrees(angle_rad) % 180.0)

    def _smooth_angle(self, raw_deg):
        """Circular mean over ANGLE_SMOOTH_N frames (handles 0°/180° wrap)."""
        self._angle_buf.append(raw_deg)
        rads = [math.radians(a * 2) for a in self._angle_buf]  # double for 180° periodicity
        s = math.atan2(sum(math.sin(r) for r in rads),
                       sum(math.cos(r) for r in rads))
        return math.degrees(s / 2) % 180.0

    def _draw_pca_axis(self, frame, contour, cx, cy, angle_deg):
        hull = cv2.convexHull(contour)
        pts  = hull.reshape(-1, 2).astype(np.float32)
        _, eigvecs = cv2.PCACompute(pts, mean=None)
        length = max(cv2.minAreaRect(hull)[1]) / 2 + 20
        dx = float(eigvecs[0, 0]) * length
        dy = float(eigvecs[0, 1]) * length
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.arrowedLine(frame, p1, p2, (0, 165, 255), 2, tipLength=0.2)
        cv2.putText(frame, f'{angle_deg:.1f}deg', (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        if not self.intrinsics_ok or self._depth_img is None or not self._bg_ready:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        # ── Freeze during pick motion — gripper/wire would be detected ───────
        if self._pick_active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
            cv2.putText(frame, 'PICK IN PROGRESS — detection paused', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow('Any-Pose Detector', frame)
            cv2.waitKey(1)
            return

        # ── Foreground mask ──────────────────────────────────────────────────
        # Only compare where BOTH background and current readings are valid
        valid_pixels = (self._bg_model > MIN_DEPTH_M) & (self._depth_img > MIN_DEPTH_M)
        diff    = self._bg_model - self._depth_img
        fg_mask = (
            valid_pixels &
            (diff > BG_THRESH_M) &
            (self._depth_img < MAX_DEPTH_M)
        ).astype(np.uint8) * 255

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  np.ones((5,  5),  np.uint8))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        status_txt = 'No object'
        status_col = (120, 120, 120)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_AREA:
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    angle_deg = self._smooth_angle(self._pca_angle(largest))
                    depth_m   = self._median_depth(cx, cy)

                    if depth_m is not None:
                        cam_x = (cx - self.ppx) * depth_m / self.fx
                        cam_y = (cy - self.ppy) * depth_m / self.fy
                        cam_z = depth_m

                        try:
                            tf = self.tf_buffer.lookup_transform(
                                'base', 'camera_color_optical_frame',
                                rclpy.time.Time(),
                                timeout=rclpy.duration.Duration(seconds=0.1))

                            pt = tf2_geometry_msgs.PointStamped()
                            pt.header.frame_id = 'camera_color_optical_frame'
                            pt.point.x = cam_x
                            pt.point.y = cam_y
                            pt.point.z = cam_z
                            pt_base = tf2_geometry_msgs.do_transform_point(pt, tf)

                            obj_x = pt_base.point.x
                            obj_y = pt_base.point.y
                            obj_z = pt_base.point.z

                            now = self.get_clock().now().to_msg()

                            pose_msg = PoseStamped()
                            pose_msg.header.stamp    = now
                            pose_msg.header.frame_id = 'base'
                            pose_msg.pose.position.x = obj_x
                            pose_msg.pose.position.y = obj_y
                            pose_msg.pose.position.z = obj_z
                            pose_msg.pose.orientation.w = 1.0
                            self.pose_pub.publish(pose_msg)

                            tf_msg = TransformStamped()
                            tf_msg.header.stamp    = now
                            tf_msg.header.frame_id = 'base'
                            tf_msg.child_frame_id  = 'detected_object'
                            tf_msg.transform.translation.x = obj_x
                            tf_msg.transform.translation.y = obj_y
                            tf_msg.transform.translation.z = obj_z
                            tf_msg.transform.rotation.w    = 1.0
                            self.tf_broadcaster.sendTransform(tf_msg)

                            angle_msg = Float64()
                            angle_msg.data = angle_deg
                            self.angle_pub.publish(angle_msg)

                            status_txt = (
                                f'X={obj_x:.3f} Y={obj_y:.3f} Z={obj_z:.3f}  '
                                f'depth={depth_m:.3f}m  angle={angle_deg:.1f}deg')
                            status_col = (0, 220, 0)

                            hull = cv2.convexHull(largest)
                            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
                            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
                            self._draw_pca_axis(frame, largest, cx, cy, angle_deg)

                        except TransformException as ex:
                            status_txt = f'TF missing: {ex}'
                            status_col = (0, 0, 200)
                    else:
                        status_txt = 'Bad depth at centroid'
                        status_col = (0, 100, 255)

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
        cv2.putText(frame, status_txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)
        cv2.imshow('Any-Pose Detector', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = AnyPoseDetectorRS()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
