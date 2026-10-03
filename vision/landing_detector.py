import cv2
import numpy as np


class LandingZoneDetector:
    """Segments potential landing regions from drone imagery using classical morphological filters."""

    def __init__(self, min_area: float = 3000.0, max_area: float = 200000.0):
        self.min_area = min_area
        self.max_area = max_area

    def detect_candidates(self, gray_frame: np.ndarray):
        """
        Segments candidate landing regions using adaptive morphological filtering.
        
        Returns:
        --------
        valid_contours: list of np.ndarray
        threshold_img: np.ndarray (binary debug mask)
        """
        # 1. Otsu thresholding after morphological gradient
        _, thresh = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Morphological closing to fill holes in candidate pads
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 3. Find external contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                valid_contours.append(c)

        return valid_contours, closed