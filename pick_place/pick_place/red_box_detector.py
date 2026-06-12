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
# Tuning knobs  (read the comments before changing)
# ─────────────────────────────────────────────────────────────────────
RATIO_NORMAL_LOW   = 1.15   # w/h must be THIS much > 1 to vote "0 deg"
RATIO_ROTATED_HIGH = 0.87   # w/h must be THIS much < 1 to vote "90 deg"

# Rolling vote window
VOTE_WINDOW    = 15   # frames kept
VOTE_THRESHOLD = 11   # frames that must agree to flip the stable state
# ─────────────────────────────────────────────────────────────────────


class RedBoxDetector(Node):
    def __init__(self):
        super().__init__('red_box_detector')
        self.bridge = CvBridge()

        # Intrinsics will be populated dynamically
        self.fx     = None
        self.fy     = None
        self.cx_int = None
        self.cy_int = None
        self.intrinsics_received = False

        self.info_sub   = self.create_subscription(
            CameraInfo, '/oak/rgb/camera_info', self.info_callback, 10)
        self.image_sub  = self.create_subscription(
            Image, '/oak/rgb/image_raw', self.image_callback, 10)
        
        self.target_pub = self.create_publisher(Point,   '/red_box_ray',   10)
        self.angle_pub  = self.create_publisher(Float64, '/red_box_angle', 10)

        # Orientation state
        self.vote_window  = deque(maxlen=VOTE_WINDOW)
        self.stable_angle = 0.0   # only 0.0 or 90.0

        self.get_logger().info(
            f"Red Box Detector ready  "
            f"(vote window={VOTE_WINDOW}, threshold={VOTE_THRESHOLD}). "
            f"Waiting for camera intrinsics..."
        )

    # ─────────────────────────────────────────────────────────────────
    def info_callback(self, msg):
        """Extract intrinsics from the K matrix once, then stop logging."""
        if not self.intrinsics_received:
            # K matrix is a 1D array of 9 elements: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self.fx = msg.k[0]
            self.cx_int = msg.k[2]
            self.fy = msg.k[4]
            self.cy_int = msg.k[5]
            self.intrinsics_received = True
            
            self.get_logger().info(
                f"Camera Intrinsics Locked! "
                f"fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx_int:.2f}, cy={self.cy_int:.2f}"
            )

    # ─────────────────────────────────────────────────────────────────
    def _raw_vote(self, w, h):
        if h < 1:
            return None
        ratio = w / h

        if ratio > RATIO_NORMAL_LOW:
            return 90.0     
        elif ratio < RATIO_ROTATED_HIGH:
            return 0.0      
        else:
            return None     

    def _update_stable_angle(self, vote):
        if vote is None:
            return   

        self.vote_window.append(vote)

        if len(self.vote_window) < VOTE_WINDOW:
            return   

        votes_90 = sum(1 for v in self.vote_window if v == 90.0)
        votes_0  = VOTE_WINDOW - votes_90

        if votes_90 >= VOTE_THRESHOLD and self.stable_angle != 90.0:
            self.stable_angle = 90.0
            self.get_logger().info("Orientation LOCKED → 90 deg (ROTATED)")

        elif votes_0 >= VOTE_THRESHOLD and self.stable_angle != 0.0:
            self.stable_angle = 0.0
            self.get_logger().info("Orientation LOCKED → 0 deg (NORMAL)")

    # ─────────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        # Do not process math until we have the true lens parameters
        if not self.intrinsics_received:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            return

        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0,   120, 70]), np.array([10,  255, 255])) +
                cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255])))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._draw_status_bar(cv_image, detected=False)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:

                bx, by, bw, bh = cv2.boundingRect(largest)
                cx = bx + bw // 2
                cy = by + bh // 2

                vote = self._raw_vote(bw, bh)
                self._update_stable_angle(vote)

                rect     = cv2.minAreaRect(largest)
                box_pts  = np.int0(cv2.boxPoints(rect))
                box_color = (0, 165, 255) if self.stable_angle == 90.0 else (0, 255, 0)
                cv2.drawContours(cv_image, [box_pts], 0, box_color, 3)
                cv2.circle(cv_image, (cx, cy), 6, (255, 0, 0), -1)

                ratio = bw / bh if bh > 0 else 0.0
                cv2.putText(cv_image, f"w/h={ratio:.2f}",
                            (cx + 10, cy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (200, 200, 0), 2, cv2.LINE_AA)

                if len(self.vote_window) > 0:
                    votes_90   = sum(1 for v in self.vote_window if v == 90.0)
                    confidence = (votes_90 / len(self.vote_window) * 100
                                  if self.stable_angle == 90.0
                                  else (1 - votes_90 / len(self.vote_window)) * 100)
                    self._draw_confidence_bar(cv_image, confidence)

                self._draw_status_bar(cv_image, detected=True)

                # ── Math: Convert Pixel to 3D Ray using Dynamic Intrinsics ──
                ray_msg   = Point()
                ray_msg.x = float((cx - self.cx_int) / self.fx)
                ray_msg.y = float((cy - self.cy_int) / self.fy)
                ray_msg.z = 1.0
                self.target_pub.publish(ray_msg)

                angle_msg      = Float64()
                angle_msg.data = self.stable_angle
                self.angle_pub.publish(angle_msg)

        cv2.imshow("Red Box Vision", cv_image)
        cv2.waitKey(1)

    # ─────────────────────────────────────────────────────────────────
    def _draw_status_bar(self, frame, detected):
        w     = frame.shape[1]
        bar_h = 70

        if not detected:
            cv2.rectangle(frame, (0, 0), (w, bar_h), (40, 40, 40), -1)
            cv2.putText(frame, "No red box detected",
                        (20, 48), cv2.FONT_HERSHEY_DUPLEX, 1.2,
                        (160, 160, 160), 2, cv2.LINE_AA)
            return

        if self.stable_angle == 0.0:
            bar_color  = (20, 100, 20)
            text_color = (0, 255, 80)
            label      = "NORMAL"
        else:
            bar_color  = (10, 60, 140)
            text_color = (0, 200, 255)
            label      = "ROTATED"

        cv2.rectangle(frame, (0, 0), (w, bar_h), bar_color, -1)
        cv2.putText(frame, f"{int(self.stable_angle):3d} deg",
                    (20, 54), cv2.FONT_HERSHEY_DUPLEX, 1.8,
                    text_color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"Box: {label}",
                    (240, 54), cv2.FONT_HERSHEY_DUPLEX, 1.4,
                    text_color, 2, cv2.LINE_AA)
        cv2.line(frame, (0, bar_h), (w, bar_h), text_color, 2)

    def _draw_confidence_bar(self, frame, pct):
        w      = frame.shape[1]
        bar_y  = 72
        bar_h  = 10
        filled = int(w * pct / 100.0)
        cv2.rectangle(frame, (0, bar_y), (w, bar_y + bar_h), (60, 60, 60), -1)
        color = (0, 255, 80) if self.stable_angle == 0.0 else (0, 200, 255)
        cv2.rectangle(frame, (0, bar_y), (filled, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{pct:.0f}% conf",
                    (w - 140, bar_y + bar_h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)


def main(args=None):
    rclpy.init(args=args)
    node = RedBoxDetector()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
