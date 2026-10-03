import os
import cv2
import numpy as np
from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from feature_extractor import FeatureExtractor


def generate_synthetic_camera_frame(width=640, height=480):
    """
    Creates a synthetic aerial camera frame simulating a drone looking down:
    - Candidate 1 (Right): A clear, bright, flat landing pad.
    - Candidate 2 (Left): A rocky, heavily cluttered unsafe zone.
    """
    frame = np.full((height, width, 3), 60, dtype=np.uint8)  # dark ground/soil

    # Candidate 1: Clear, round landing pad (SAFE)
    cv2.circle(frame, (460, 240), 90, (180, 180, 180), -1)  # Smooth concrete
    cv2.circle(frame, (460, 240), 75, (220, 220, 220), 4)   # Outer ring
    cv2.putText(frame, "H", (435, 265), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (240, 240, 240), 4)

    # Candidate 2: Irregular patch filled with high-frequency obstacles (UNSAFE)
    pts = np.array([[100, 120], [280, 140], [250, 340], [80, 300]], np.int32)
    cv2.fillPoly(frame, [pts], (140, 140, 140))
    
    # Add random debris / obstacle rocks
    np.random.seed(7)
    for _ in range(45):
        rx = np.random.randint(90, 260)
        ry = np.random.randint(130, 320)
        cv2.circle(frame, (rx, ry), np.random.randint(3, 8), (20, 20, 20), -1)

    return frame


def run():
    print("=" * 60)
    print("STAGE 4: CLASSICAL COMPUTER VISION PIPELINE TEST")
    print("=" * 60)

    # 1. Synthesize camera input
    frame = generate_synthetic_camera_frame(640, 480)
    
    # 2. Pipeline components
    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=4000.0)
    extractor = FeatureExtractor(640, 480)

    # 3. Execute
    _, gray, edges = preprocessor.process(frame)
    candidates, binary_mask = detector.detect_candidates(gray)

    print(f"[*] Candidates detected by OpenCV: {len(candidates)}")

    annotated = frame.copy()
    # Draw image center crosshair
    cv2.drawMarker(annotated, (320, 240), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

    for i, c in enumerate(candidates):
        features, spatial = extractor.extract_features(c, gray, edges)
        if features is None:
            continue

        cx, cy = int(spatial["centroid"][0]), int(spatial["centroid"][1])
        ex, ey = spatial["error_x"], spatial["error_y"]
        x, y, w, h = spatial["bbox"]

        # Draw contour and centroid
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        cv2.line(annotated, (320, 240), (cx, cy), (255, 255, 0), 1)

        print(f"\n--- Candidate #{i + 1} ---")
        print(f"  Centroid: ({cx}, {cy}) | Pixel Errors: ex={ex:+.1f} px, ey={ey:+.1f} px")
        print(f"  Area Ratio:        {features[0]:.4f}")
        print(f"  Obstacle Density:  {features[1]:.4f}")
        print(f"  Min Clearance:     {features[2]:.4f}")
        print(f"  Surface Flatness:  {features[3]:.4f}")
        print(f"  Centroid Offset:   {features[4]:.4f}")
        print(f"  Circularity:       {features[5]:.4f}")

    # Save output visualization
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_img = os.path.join(out_dir, "stage4_vision_output.png")
    cv2.imwrite(out_img, annotated)
    print(f"\n[SAVED] Annotated vision output saved to: {os.path.abspath(out_img)}")

    # Non-blocking display (shows window for 2.5 seconds, then closes automatically)
    cv2.imshow("Stage 4 - Landing Zone Detection & Error Vectors", annotated)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()
    print("[SUCCESS] Stage 4 pipeline complete.")


if __name__ == "__main__":
    run()