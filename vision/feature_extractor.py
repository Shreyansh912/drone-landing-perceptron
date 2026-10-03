import cv2
import numpy as np


class FeatureExtractor:
    """Computes the 6D feature vector from image contours and edge maps."""

    def __init__(self, image_width: int = 640, image_height: int = 480):
        self.img_w = image_width
        self.img_h = image_height
        self.total_area = float(image_width * image_height)
        self.cx_frame = image_width / 2.0
        self.cy_frame = image_height / 2.0
        self.max_offset = np.sqrt(self.cx_frame**2 + self.cy_frame**2)

    def extract_features(self, contour: np.ndarray, gray_frame: np.ndarray, edge_map: np.ndarray):
        """
        Extracts 6D features and spatial alignment errors for a candidate contour.
        
        Returns:
        --------
        feature_vector: np.ndarray of shape (6,)
        spatial_info: dict with 'centroid', 'error_x', 'error_y', 'bounding_box'
        """
        # --- 1. Area & Centroid via Moments ---
        M = cv2.moments(contour)
        if M["m00"] <= 1e-4:
            return None, None

        area = float(M["m00"])
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Feature 1: Calibrated Normalized Area
        # Matches training distribution where standard pads score between 0.60 and 0.90
        f1_area = float(np.clip(area / 32000.0, 0.05, 0.95))

        # --- 2. Contour Mask & Obstacle Density ---
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        # Count obstacle/edge pixels strictly inside this candidate patch
        candidate_edges = cv2.bitwise_and(edge_map, edge_map, mask=mask)
        edge_pixel_count = np.count_nonzero(candidate_edges)

        # Feature 2: Obstacle Density
        f2_obstacle_density = min(1.0, edge_pixel_count / max(area, 1.0))

        # --- 3. Minimum Obstacle Distance (Clearance) ---
        # Compute Euclidean Distance Transform from non-edge pixels to nearest edge
        inv_edges = cv2.bitwise_not(edge_map)
        dist_transform = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
        
        # Minimum clearance near candidate centroid
        cx_int = int(np.clip(cx, 0, self.img_w - 1))
        cy_int = int(np.clip(cy, 0, self.img_h - 1))
        f3_min_dist = float(dist_transform[cy_int, cx_int]) / 100.0
        f3_min_dist = float(np.clip(f3_min_dist, 0.05, 0.95))

        # --- 4. Surface Flatness (Inverse of Texture Variance) ---
        mean_val, std_val = cv2.meanStdDev(gray_frame, mask=mask)
        f4_flatness = 1.0 - min(1.0, float(std_val[0][0]) / 64.0)

        # --- 5. Centroid Offset from Principal Image Center ---
        pixel_error_x = cx - self.cx_frame
        pixel_error_y = cy - self.cy_frame
        dist_from_center = np.sqrt(pixel_error_x**2 + pixel_error_y**2)
        f5_offset = min(1.0, dist_from_center / self.max_offset)

        # --- 6. Circularity / Isoperimetric Quotient ---
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter > 0:
            f6_circularity = (4.0 * np.pi * area) / (perimeter**2)
            f6_circularity = min(1.0, f6_circularity)
        else:
            f6_circularity = 0.0

        # Pack into 6D feature vector
        features = np.array([
            f1_area,
            f2_obstacle_density,
            f3_min_dist,
            f4_flatness,
            f5_offset,
            f6_circularity
        ], dtype=np.float64)

        x, y, w, h = cv2.boundingRect(contour)
        spatial_info = {
            "centroid": (cx, cy),
            "error_x": pixel_error_x,
            "error_y": pixel_error_y,
            "bbox": (x, y, w, h),
            "area": area
        }

        return features, spatial_info