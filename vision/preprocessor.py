import cv2
import numpy as np


class ImagePreprocessor:
    """Handles image cleaning and multi-channel representations for drone perception."""

    def __init__(self, blur_kernel: tuple = (5, 5), canny_low: int = 50, canny_high: int = 150):
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high

    def process(self, frame_bgr: np.ndarray):
        """
        Processes a raw BGR frame.
        
        Returns:
        --------
        blurred : np.ndarray (Smoothed BGR)
        gray : np.ndarray (Grayscale)
        edges : np.ndarray (Binary edge map indicating obstacles/textures)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame received by preprocessor.")

        # 1. Gaussian blur to suppress camera sensor noise
        blurred = cv2.GaussianBlur(frame_bgr, self.blur_kernel, 0)

        # 2. Grayscale conversion for luminance and texture calculations
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 3. Canny edge detector for hazard identification
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        return blurred, gray, edges