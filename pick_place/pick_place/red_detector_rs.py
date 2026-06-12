import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, TransformException
import tf2_geometry_msgs
from collections import deque
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Tuning
# ─────────────────────────────────────────────────────────────────────────────
MIN_AREA          = 500    # px² — ignore small blobs
DEPTH_PATCH       = 9      # sample NxN patch around centroid for median depth
MIN_DEPTH_M       = 0.10   # discard readings closer than 10 cm (sensor noise)
MAX_DEPTH_M       = 3.00   # discard readings farther than 3 m

RATIO_NORMAL_LOW   = 1.15  # w/h > this → box is NORMAL (0 deg)
RATIO_ROTATED_HIGH = 0.87  # w/h < this → box is ROTATED (90 deg)
VOTE_WINDOW    = 15
VOTE_THRESHOLD = 11
# ─────────────────────────────────────────────────────────────────────────────


class RedDetectorRS(Node):
    """
    Detects a red object using RealSense D435i colour + aligned depth.

    For each frame:
      1. HSV threshold → largest red contour
      2. Median depth over a DEPTH_PATCH×DEPTH_PATCH window at the centroid
      3. Back-project to 3-D in camera_color_optical_frame
      4. tf2 lookup → base frame coordinates
      5. Publish /red_object_pose (PoseStamped, frame=base)
             and /red_box_angle  (Float64, degrees)
             and TF  base → red_object

    Camera topics (RealSense ROS2 defaults):
        '/camera/camera/color/image_raw'
        '/camera/camera/color/camera_info'
        '/camera/camera/aligned_depth_to_color/image_raw'

    Static TF required (set after measuring camera position):
        ros2 run tf2_ros static_transform_publisher \
          --x ??? --y ??? --z ??? \
          --roll ??? --pitch ??? --yaw ??? \
          --frame-id base --child-frame-id camera_link
    """

    def __init__(self):
        super().__init__('red_detector_rs')
        self.bridge = CvBridge()

        # Camera intrinsics (populated once from /camera_info)
        self.fx = self.fy = self.ppx = self.ppy = None
        self.intrinsics_ok = False

        # TF
        self.tf_buffer      = Buffer()
        self.tf_listener    = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Latest aligned depth image (uint16, millimetres)
        self._depth_img = None

        # Orientation vote
        self.vote_window  = deque(maxlen=VOTE_WINDOW)
        self.stable_angle = 0.0

        # Subscribers
        self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self._depth_cb, 10)
        self.create_subscription(
            Image, '/camera/camera/color/image_raw', self._color_cb, 10)

        # Publishers
        self.pose_pub  = self.create_publisher(PoseStamped, '/red_object_pose', 10)
        self.angle_pub = self.create_publisher(Float64,      '/red_box_angle',   10)

        self.get_logger().info(
            "RealSense Red Detector ready. "
            "Waiting for camera info + static TF (base → camera_link)...")

    # ─────────────────────────────────────────────────────────────────────────
    def _info_cb(self, msg):
        if not self.intrinsics_ok:
            self.fx  = msg.k[0]
            self.ppx = msg.k[2]
            self.fy  = msg.k[4]
            self.ppy = msg.k[5]
            self.intrinsics_ok = True
            self.get_logger().info(
                f"Intrinsics locked: fx={self.fx:.2f} fy={self.fy:.2f} "
                f"ppx={self.ppx:.2f} ppy={self.ppy:.2f}")

    def _depth_cb(self, msg):
        self._depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    # ─────────────────────────────────────────────────────────────────────────
    def _median_depth(self, cx, cy):
        """
        Robust depth at pixel (cx, cy): median over a DEPTH_PATCH×DEPTH_PATCH
        neighbourhood, ignoring zero / invalid readings.
        Returns depth in metres, or None if no valid pixels found.
        """
        h, w = self._depth_img.shape
        half  = DEPTH_PATCH // 2
        y0, y1 = max(0, cy - half), min(h, cy + half + 1)
        x0, x1 = max(0, cx - half), min(w, cx + half + 1)
        patch = self._depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0   # mm → m

    # ─────────────────────────────────────────────────────────────────────────
    def _update_orientation(self, bw, bh):
        if bh < 1:
            return
        ratio = bw / bh
        if   ratio > RATIO_NORMAL_LOW:   vote = 90.0
        elif ratio < RATIO_ROTATED_HIGH: vote = 0.0
        else:                            return   # ambiguous — skip

        self.vote_window.append(vote)
        if len(self.vote_window) < VOTE_WINDOW:
            return
        v90 = sum(1 for v in self.vote_window if v == 90.0)
        v0  = VOTE_WINDOW - v90
        if   v90 >= VOTE_THRESHOLD and self.stable_angle != 90.0:
            self.stable_angle = 90.0
            self.get_logger().info("Orientation → 90 deg (ROTATED)")
        elif v0  >= VOTE_THRESHOLD and self.stable_angle != 0.0:
            self.stable_angle = 0.0
            self.get_logger().info("Orientation → 0 deg (NORMAL)")

    # ─────────────────────────────────────────────────────────────────────────
    def _color_cb(self, msg):
        if not self.intrinsics_ok or self._depth_img is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        # ── HSV red mask ─────────────────────────────────────────────────────
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0,   120,  70]), np.array([10,  255, 255])) |
                cv2.inRange(hsv, np.array([170, 120,  70]), np.array([180, 255, 255])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected   = False
        status_txt = "No red object"
        status_col = (120, 120, 120)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > MIN_AREA:
                bx, by, bw, bh = cv2.boundingRect(largest)
                cx = bx + bw // 2
                cy = by + bh // 2

                self._update_orientation(bw, bh)

                depth_m = self._median_depth(cx, cy)

                if depth_m is not None and MIN_DEPTH_M < depth_m < MAX_DEPTH_M:
                    # Back-project → camera_color_optical_frame
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

                        # Publish PoseStamped
                        now = self.get_clock().now().to_msg()
                        pose_msg = PoseStamped()
                        pose_msg.header.stamp    = now
                        pose_msg.header.frame_id = 'base'
                        pose_msg.pose.position.x = obj_x
                        pose_msg.pose.position.y = obj_y
                        pose_msg.pose.position.z = obj_z
                        pose_msg.pose.orientation.w = 1.0
                        self.pose_pub.publish(pose_msg)

                        # Publish TF: base → red_object
                        tf_msg = TransformStamped()
                        tf_msg.header.stamp    = now
                        tf_msg.header.frame_id = 'base'
                        tf_msg.child_frame_id  = 'red_object'
                        tf_msg.transform.translation.x = obj_x
                        tf_msg.transform.translation.y = obj_y
                        tf_msg.transform.translation.z = obj_z
                        tf_msg.transform.rotation.w    = 1.0
                        self.tf_broadcaster.sendTransform(tf_msg)

                        # Publish angle
                        angle_msg = Float64()
                        angle_msg.data = self.stable_angle
                        self.angle_pub.publish(angle_msg)

                        detected   = True
                        status_txt = (
                            f"X={obj_x:.3f} Y={obj_y:.3f} Z={obj_z:.3f} m  "
                            f"| depth={depth_m:.3f} m  | {int(self.stable_angle)} deg")
                        status_col = (0, 220, 0)

                        # Draw bounding box
                        rect    = cv2.minAreaRect(largest)
                        box_pts = np.int0(cv2.boxPoints(rect))
                        box_col = (0, 165, 255) if self.stable_angle == 90.0 else (0, 255, 0)
                        cv2.drawContours(frame, [box_pts], 0, box_col, 3)
                        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                    except TransformException as ex:
                        status_txt = f"TF missing: {ex}"
                        status_col = (0, 0, 200)

                else:
                    d_str = f"{depth_m:.3f}" if depth_m else "NaN"
                    status_txt = f"Bad depth: {d_str} m"
                    status_col = (0, 100, 255)

        # HUD
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (30, 30, 30), -1)
        cv2.putText(frame, status_txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)
        cv2.imshow("RealSense Red Detector", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = RedDetectorRS()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
