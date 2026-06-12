import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from collections import deque
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Tuning knobs
# ─────────────────────────────────────────────────────────────────────
RATIO_NORMAL_LOW   = 1.15
RATIO_ROTATED_HIGH = 0.87

VOTE_WINDOW    = 15
VOTE_THRESHOLD = 11

# Minimum contour area to be considered a valid detection
MIN_AREA = 500
# ─────────────────────────────────────────────────────────────────────


class BoxTracker:
    """
    Tracks orientation votes for a single colored box.
    Identical logic to the original RedBoxDetector vote system,
    just extracted so it can be reused for any color.
    """
    def __init__(self):
        self.vote_window  = deque(maxlen=VOTE_WINDOW)
        self.stable_angle = 0.0   # only 0.0 or 90.0

    def raw_vote(self, w, h):
        if h < 1:
            return None
        ratio = w / h
        if ratio > RATIO_NORMAL_LOW:
            return 90.0
        elif ratio < RATIO_ROTATED_HIGH:
            return 0.0
        return None

    def update(self, vote):
        """Push a vote and return (stable_angle, did_change)."""
        if vote is None:
            return self.stable_angle, False

        self.vote_window.append(vote)
        if len(self.vote_window) < VOTE_WINDOW:
            return self.stable_angle, False

        votes_90 = sum(1 for v in self.vote_window if v == 90.0)
        votes_0  = VOTE_WINDOW - votes_90

        if votes_90 >= VOTE_THRESHOLD and self.stable_angle != 90.0:
            self.stable_angle = 90.0
            return 90.0, True
        elif votes_0 >= VOTE_THRESHOLD and self.stable_angle != 0.0:
            self.stable_angle = 0.0
            return 0.0, True

        return self.stable_angle, False

    def confidence(self):
        """Return 0-100 confidence for the current stable angle."""
        if not self.vote_window:
            return 0.0
        votes_90 = sum(1 for v in self.vote_window if v == 90.0)
        return (votes_90 / len(self.vote_window) * 100
                if self.stable_angle == 90.0
                else (1 - votes_90 / len(self.vote_window)) * 100)


class MultiBoxDetector(Node):
    def __init__(self):
        super().__init__('multi_box_detector')
        self.bridge = CvBridge()

        self.fx = self.fy = self.cx_int = self.cy_int = None
        self.intrinsics_received = False

        # One tracker per color
        self.red_tracker  = BoxTracker()
        self.blue_tracker = BoxTracker()

        # Camera subscribers
        self.info_sub  = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/oak/rgb/image_raw', self.image_callback, 10)

        # ── Publishers ────────────────────────────────────────────────
        # Red box  (same topic names as before — pick_node unchanged)
        self.red_ray_pub   = self.create_publisher(Point,   '/red_box_ray',   10)
        self.red_angle_pub = self.create_publisher(Float64, '/red_box_angle', 10)

        # Blue box  (new topics consumed by pick_node)
        self.blue_ray_pub   = self.create_publisher(Point,   '/blue_box_ray',   10)
        self.blue_angle_pub = self.create_publisher(Float64, '/blue_box_angle', 10)

        self.get_logger().info(
            "Multi-Box Detector ready (RED + BLUE). "
            "Waiting for camera intrinsics...")

    # ─────────────────────────────────────────────────────────────────
    def info_callback(self, msg):
        if not self.intrinsics_received:
            self.fx     = msg.k[0]
            self.cx_int = msg.k[2]
            self.fy     = msg.k[4]
            self.cy_int = msg.k[5]
            self.intrinsics_received = True
            self.get_logger().info(
                f"Intrinsics locked — "
                f"fx={self.fx:.2f} fy={self.fy:.2f} "
                f"cx={self.cx_int:.2f} cy={self.cy_int:.2f}")

    # ─────────────────────────────────────────────────────────────────
    def _pixel_to_ray(self, px, py):
        """Convert pixel (px, py) to a normalised 3-D ray Point."""
        p = Point()
        p.x = float((px - self.cx_int) / self.fx)
        p.y = float((py - self.cy_int) / self.fy)
        p.z = 1.0
        return p

    def _build_red_mask(self, hsv):
        """Red wraps around 0/180 in HSV — needs two ranges."""
        lo1 = cv2.inRange(hsv, np.array([0,   120,  70]), np.array([10,  255, 255]))
        lo2 = cv2.inRange(hsv, np.array([170, 120,  70]), np.array([180, 255, 255]))
        return lo1 + lo2

    def _build_blue_mask(self, hsv):
        """Blue sits around H=100-130 in OpenCV HSV."""
        return cv2.inRange(hsv,
                           np.array([100, 120,  70]),
                           np.array([130, 255, 255]))

    def _detect_box(self, mask):
        """
        Return (cx, cy, bw, bh, largest_contour) for the biggest blob
        in mask, or None if nothing qualifies.
        """
        kernel   = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_AREA:
            return None

        bx, by, bw, bh = cv2.boundingRect(largest)
        cx = bx + bw // 2
        cy = by + bh // 2
        return cx, cy, bw, bh, largest

    # ─────────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        if not self.intrinsics_received:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        red_result  = self._detect_box(self._build_red_mask(hsv))
        blue_result = self._detect_box(self._build_blue_mask(hsv))

        # ── Draw HUD ─────────────────────────────────────────────────
        self._draw_header(frame, red_result, blue_result)

        # ── Handle RED ───────────────────────────────────────────────
        if red_result:
            cx, cy, bw, bh, contour = red_result

            vote = self.red_tracker.raw_vote(bw, bh)
            self.red_tracker.update(vote)
            angle = self.red_tracker.stable_angle

            # Draw
            rect     = cv2.minAreaRect(contour)
            box_pts  = np.int0(cv2.boxPoints(rect))
            color    = (0, 165, 255) if angle == 90.0 else (0, 255, 0)
            cv2.drawContours(frame, [box_pts], 0, color, 3)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            ratio = bw / bh if bh > 0 else 0.0
            cv2.putText(frame, f"RED  w/h={ratio:.2f}  {int(angle)}deg",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            conf_y = 82
            self._draw_conf_bar(frame, self.red_tracker.confidence(),
                                conf_y, (0, 80, 200), label="RED conf")

            # Publish
            self.red_ray_pub.publish(self._pixel_to_ray(cx, cy))
            msg_a = Float64(); msg_a.data = angle
            self.red_angle_pub.publish(msg_a)

        # ── Handle BLUE ──────────────────────────────────────────────
        if blue_result:
            cx, cy, bw, bh, contour = blue_result

            vote = self.blue_tracker.raw_vote(bw, bh)
            self.blue_tracker.update(vote)
            angle = self.blue_tracker.stable_angle

            # Draw
            rect     = cv2.minAreaRect(contour)
            box_pts  = np.int0(cv2.boxPoints(rect))
            color    = (255, 165, 0) if angle == 90.0 else (255, 200, 0)
            cv2.drawContours(frame, [box_pts], 0, color, 3)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)

            ratio = bw / bh if bh > 0 else 0.0
            cv2.putText(frame, f"BLUE w/h={ratio:.2f}  {int(angle)}deg",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

            conf_y = 98
            self._draw_conf_bar(frame, self.blue_tracker.confidence(),
                                conf_y, (200, 140, 0), label="BLU conf")

            # Publish
            self.blue_ray_pub.publish(self._pixel_to_ray(cx, cy))
            msg_a = Float64(); msg_a.data = angle
            self.blue_angle_pub.publish(msg_a)

        cv2.imshow("Multi-Box Detector", frame)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────
    def _draw_header(self, frame, red_result, blue_result):
        w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (w, 78), (30, 30, 30), -1)

        # Red status
        r_text  = f"RED:  {int(self.red_tracker.stable_angle)}deg" if red_result else "RED:  not seen"
        r_color = (0, 100, 255) if red_result else (100, 100, 100)
        cv2.putText(frame, r_text, (15, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, r_color, 2, cv2.LINE_AA)

        # Blue status
        b_text  = f"BLUE: {int(self.blue_tracker.stable_angle)}deg" if blue_result else "BLUE: not seen"
        b_color = (255, 180, 0) if blue_result else (100, 100, 100)
        cv2.putText(frame, b_text, (w // 2 + 10, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, b_color, 2, cv2.LINE_AA)

        cv2.line(frame, (0, 78), (w, 78), (80, 80, 80), 1)

    def _draw_conf_bar(self, frame, pct, y, color, label=""):
        w      = frame.shape[1]
        filled = int(w * pct / 100.0)
        cv2.rectangle(frame, (0, y), (w, y + 12), (50, 50, 50), -1)
        cv2.rectangle(frame, (0, y), (filled, y + 12), color, -1)
        cv2.putText(frame, f"{label} {pct:.0f}%",
                    (w - 160, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)


def main(args=None):
    rclpy.init(args=args)
    node = MultiBoxDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
