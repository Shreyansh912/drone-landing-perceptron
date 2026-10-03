import cv2
import numpy as np


class ArUcoTargetTracker:
    def __init__(self, marker_id=0, marker_size_m=0.20):
        self.marker_id = marker_id
        self.marker_size = marker_size_m

        # Standard camera intrinsic matrix (pinhole model, fx=fy=360, cx=320, cy=240)
        self.camera_matrix = np.array([
            [360.0,   0.0, 320.0],
            [  0.0, 360.0, 240.0],
            [  0.0,   0.0,   1.0]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # OpenCV ArUco detector initialization (supports OpenCV 4.7+)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None

    def draw_marker_on_pad(self, image, center_px, pad_radius_px):
        """Renders a synthetic ArUco marker inside the landing pad center."""
        marker_px_size = max(8, int(pad_radius_px * 0.45))
        top_left_x = int(center_px[0] - marker_px_size // 2)
        top_left_y = int(center_px[1] - marker_px_size // 2)

        if marker_px_size >= 12 and 0 <= top_left_x < 640 - marker_px_size and 0 <= top_left_y < 480 - marker_px_size:
            marker_img = cv2.aruco.generateImageMarker(self.dictionary, self.marker_id, marker_px_size)
            marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
            image[top_left_y:top_left_y + marker_px_size, top_left_x:top_left_x + marker_px_size] = marker_bgr
        return image

    def detect_pose(self, frame):
        """
        Detects ArUco marker and returns physical translation error (dx_m, dy_m, dz_m).
        Returns: (detected: bool, tvec: np.ndarray, corners: np.ndarray)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)

        if ids is not None and self.marker_id in ids.flatten():
            idx = int(np.where(ids.flatten() == self.marker_id)[0][0])
            c = corners[idx][0]

            # Solve PnP for metric 3D position
            half_s = self.marker_size / 2.0
            obj_points = np.array([
                [-half_s,  half_s, 0.0],
                [ half_s,  half_s, 0.0],
                [ half_s, -half_s, 0.0],
                [-half_s, -half_s, 0.0]
            ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                obj_points, c, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if success:
                return True, tvec.flatten(), corners[idx]

        return False, None, None