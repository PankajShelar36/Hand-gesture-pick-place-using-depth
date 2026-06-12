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
MIN_AREA       = 1500    # px² — ignore small blobs / noise
DEPTH_PATCH    = 9       # NxN median depth patch at centroid
MIN_DEPTH_M    = 0.20    # discard readings closer than 20 cm
MAX_DEPTH_M    = 2.00    # discard readings farther than 2 m
ANGLE_SMOOTH_N = 10      # circular-mean smoothing window for angle

# HSV colour ranges  (OpenCV: H in [0, 180])
# Tune S/V lower bounds if boxes look washed-out or shadowed under your lighting
RED_LO1 = np.array([  0, 100,  60], dtype=np.uint8)
RED_HI1 = np.array([ 10, 255, 255], dtype=np.uint8)
RED_LO2 = np.array([165, 100,  60], dtype=np.uint8)   # red wraps around 180 in HSV
RED_HI2 = np.array([180, 255, 255], dtype=np.uint8)
BLUE_LO = np.array([ 95, 100,  50], dtype=np.uint8)
BLUE_HI = np.array([135, 255, 255], dtype=np.uint8)
# ─────────────────────────────────────────────────────────────────────────────


class VoiceColorDetectorRS(Node):
    """
    Detects red and blue boxes using pure HSV colour detection + depth back-projection.

    No background model required — can be started at any time, boxes on the
    table or not. Detection relies only on colour (HSV mask) and depth (for 3-D
    position). This avoids the background-capture startup ritual and the failure
    mode where a box placed before startup gets baked into the background.

    Publishes:
        /red_box_pose   (PoseStamped, frame=base)
        /red_box_angle  (Float64, degrees 0-180)
        /blue_box_pose  (PoseStamped, frame=base)
        /blue_box_angle (Float64, degrees 0-180)

    Detection is paused while /pick_active = True.
    """

    def __init__(self):
        super().__init__('voice_color_detector_rs')
        self.bridge = CvBridge()

        self.fx = self.fy = self.ppx = self.ppy = None
        self.intrinsics_ok = False

        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._depth_img   = None
        self._pick_active = False

        self._red_buf  = deque(maxlen=ANGLE_SMOOTH_N)
        self._blue_buf = deque(maxlen=ANGLE_SMOOTH_N)

        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_cb, 10)
        self.create_subscription(
            Bool, '/pick_active', self._pick_active_cb, 10)

        self.red_pose_pub   = self.create_publisher(PoseStamped, '/red_box_pose',   10)
        self.red_angle_pub  = self.create_publisher(Float64,     '/red_box_angle',  10)
        self.blue_pose_pub  = self.create_publisher(PoseStamped, '/blue_box_pose',  10)
        self.blue_angle_pub = self.create_publisher(Float64,     '/blue_box_angle', 10)

        self.get_logger().info(
            'Voice Colour Detector ready (HSV-only, no background model). '
            'Place red/blue boxes anytime — detection starts immediately.')

    # ─────────────────────────────────────────────────────────────────────────
    def _info_cb(self, msg):
        if not self.intrinsics_ok:
            self.fx  = msg.k[0];  self.ppx = msg.k[2]
            self.fy  = msg.k[4];  self.ppy = msg.k[5]
            self.intrinsics_ok = True
            self.get_logger().info(
                f'Intrinsics: fx={self.fx:.2f} fy={self.fy:.2f} '
                f'ppx={self.ppx:.2f} ppy={self.ppy:.2f}')

    def _depth_cb(self, msg):
        raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self._depth_img = raw.astype(np.float32) / 1000.0   # mm → m

    def _pick_active_cb(self, msg):
        self._pick_active = msg.data

    # ─────────────────────────────────────────────────────────────────────────
    def _median_depth(self, cx, cy):
        h, w = self._depth_img.shape
        half = DEPTH_PATCH // 2
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        patch = self._depth_img[y0:y1, x0:x1]
        valid = patch[(patch > MIN_DEPTH_M) & (patch < MAX_DEPTH_M)]
        return float(np.median(valid)) if valid.size > 0 else None

    def _pca_angle(self, contour):
        hull = cv2.convexHull(contour)
        pts  = hull.reshape(-1, 2).astype(np.float32)
        _, eigvecs = cv2.PCACompute(pts, mean=None)
        return float(np.degrees(
            np.arctan2(float(eigvecs[0, 1]), float(eigvecs[0, 0]))) % 180.0)

    def _smooth_angle(self, buf, raw_deg):
        buf.append(raw_deg)
        rads = [math.radians(a * 2) for a in buf]
        s = math.atan2(sum(math.sin(r) for r in rads),
                       sum(math.cos(r) for r in rads))
        return math.degrees(s / 2) % 180.0

    # ─────────────────────────────────────────────────────────────────────────
    def _process_colour(self, frame, colour_mask, angle_buf,
                        pose_pub, angle_pub, tf_child, draw_color, label):
        """Find the largest colour blob, get its 3-D pose via depth, publish."""
        mask = cv2.morphologyEx(colour_mask, cv2.MORPH_OPEN,  np.ones((5,  5), np.uint8))
        mask = cv2.morphologyEx(mask,        cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) <= MIN_AREA:
            return
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        angle_deg = self._smooth_angle(angle_buf, self._pca_angle(largest))
        depth_m   = self._median_depth(cx, cy)
        if depth_m is None:
            return

        cam_x = (cx - self.ppx) * depth_m / self.fx
        cam_y = (cy - self.ppy) * depth_m / self.fy

        try:
            tf = self.tf_buffer.lookup_transform(
                'base', 'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))

            pt = tf2_geometry_msgs.PointStamped()
            pt.header.frame_id = 'camera_color_optical_frame'
            pt.point.x = cam_x
            pt.point.y = cam_y
            pt.point.z = depth_m
            pt_base = tf2_geometry_msgs.do_transform_point(pt, tf)

            now = self.get_clock().now().to_msg()

            pose_msg = PoseStamped()
            pose_msg.header.stamp    = now
            pose_msg.header.frame_id = 'base'
            pose_msg.pose.position.x = pt_base.point.x
            pose_msg.pose.position.y = pt_base.point.y
            pose_msg.pose.position.z = pt_base.point.z
            pose_msg.pose.orientation.w = 1.0
            pose_pub.publish(pose_msg)

            tf_msg = TransformStamped()
            tf_msg.header.stamp    = now
            tf_msg.header.frame_id = 'base'
            tf_msg.child_frame_id  = tf_child
            tf_msg.transform.translation.x = pt_base.point.x
            tf_msg.transform.translation.y = pt_base.point.y
            tf_msg.transform.translation.z = pt_base.point.z
            tf_msg.transform.rotation.w    = 1.0
            self.tf_broadcaster.sendTransform(tf_msg)

            angle_msg = Float64()
            angle_msg.data = angle_deg
            angle_pub.publish(angle_msg)

            # ── Visual overlay ────────────────────────────────────────────────
            hull = cv2.convexHull(largest)
            cv2.drawContours(frame, [hull], 0, draw_color, 2)
            cv2.circle(frame, (cx, cy), 6, draw_color, -1)

            pts2 = hull.reshape(-1, 2).astype(np.float32)
            _, evecs = cv2.PCACompute(pts2, mean=None)
            L  = max(cv2.minAreaRect(hull)[1]) / 2 + 20
            dx = float(evecs[0, 0]) * L
            dy = float(evecs[0, 1]) * L
            cv2.arrowedLine(frame,
                            (int(cx - dx), int(cy - dy)),
                            (int(cx + dx), int(cy + dy)),
                            draw_color, 2, tipLength=0.2)
            cv2.putText(frame, f'{label} {angle_deg:.1f}deg',
                        (cx + 10, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f'({pt_base.point.x:.2f},{pt_base.point.y:.2f},{pt_base.point.z:.2f})',
                        (cx + 10, cy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, draw_color, 1, cv2.LINE_AA)

        except TransformException:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        if not self.intrinsics_ok or self._depth_img is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        if self._pick_active:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
            cv2.putText(frame, 'PICK IN PROGRESS — detection paused', (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.imshow('Voice Colour Detector', frame)
            cv2.waitKey(1)
            return

        hsv       = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        red_mask  = cv2.bitwise_or(cv2.inRange(hsv, RED_LO1, RED_HI1),
                                   cv2.inRange(hsv, RED_LO2, RED_HI2))
        blue_mask = cv2.inRange(hsv, BLUE_LO, BLUE_HI)

        self._process_colour(
            frame, red_mask,
            self._red_buf, self.red_pose_pub, self.red_angle_pub,
            'detected_red_box', (0, 0, 220), 'RED')
        self._process_colour(
            frame, blue_mask,
            self._blue_buf, self.blue_pose_pub, self.blue_angle_pub,
            'detected_blue_box', (220, 0, 0), 'BLUE')

        cv2.imshow('Voice Colour Detector', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VoiceColorDetectorRS()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
