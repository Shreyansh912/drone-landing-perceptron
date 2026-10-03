import os
import cv2
import numpy as np
from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from feature_extractor import FeatureExtractor
from landing_evaluator import LandingZoneEvaluator


def build_aerial_frame(scenario: str = "mixed", width: int = 640, height: int = 480) -> np.ndarray:
    """
    Generates synthetic downward-facing drone camera frames.
    - 'mixed': One safe concrete landing pad and one rocky unsafe zone.
    - 'all_unsafe': Hazardous debris fields only.
    """
    frame = np.full((height, width, 3), 50, dtype=np.uint8)  # Terrain ground

    if scenario == "mixed":
        # Candidate 1: Safe Landing Zone (Right)
        cv2.circle(frame, (480, 240), 95, (175, 175, 175), -1)
        cv2.circle(frame, (480, 240), 80, (220, 220, 220), 4)
        cv2.putText(frame, "H", (455, 265), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (240, 240, 240), 4)

        # Candidate 2: Hazardous Rocky Zone (Left)
        pts = np.array([[80, 130], [270, 150], [240, 360], [70, 310]], np.int32)
        cv2.fillPoly(frame, [pts], (125, 125, 125))
        np.random.seed(11)
        for _ in range(55):
            rx = np.random.randint(85, 255)
            ry = np.random.randint(140, 340)
            cv2.circle(frame, (rx, ry), np.random.randint(3, 7), (15, 15, 15), -1)

    elif scenario == "all_unsafe":
        # Two hazardous debris zones
        pts1 = np.array([[80, 100], [260, 110], [240, 280], [70, 260]], np.int32)
        cv2.fillPoly(frame, [pts1], (120, 120, 120))
        pts2 = np.array([[380, 200], [580, 190], [560, 410], [370, 390]], np.int32)
        cv2.fillPoly(frame, [pts2], (120, 120, 120))
        np.random.seed(15)
        for _ in range(90):
            cv2.circle(frame, (np.random.randint(80, 250), np.random.randint(105, 270)), 5, (20, 20, 20), -1)
            cv2.circle(frame, (np.random.randint(380, 570), np.random.randint(200, 400)), 5, (20, 20, 20), -1)

    return frame


def run_pipeline(scenario: str = "mixed"):
    print("=" * 65)
    print(f"STAGE 5: FULL PERCEPTION + PERCEPTRON DECISION PIPELINE [{scenario.upper()}]")
    print("=" * 65)

    # 1. Initialize Modules
    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=3500.0)
    extractor = FeatureExtractor(640, 480)
    evaluator = LandingZoneEvaluator()

    # 2. Preprocess Frame
    frame = build_aerial_frame(scenario=scenario)
    _, gray, edges = preprocessor.process(frame)
    candidates, _ = detector.detect_candidates(gray)

    print(f"[*] Candidate regions segmented: {len(candidates)}")

    annotated = frame.copy()
    # Draw optical center crosshair (Drone position)
    cv2.drawMarker(annotated, (320, 240), (255, 255, 0), cv2.MARKER_CROSS, 24, 2)
    cv2.putText(annotated, "DRONE CENTER", (330, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    safe_candidates = []

    # 3. Feature Extraction & Perceptron Inference
    for idx, c in enumerate(candidates):
        features, spatial = extractor.extract_features(c, gray, edges)
        if features is None:
            continue

        is_safe, z_score = evaluator.evaluate(features)
        status_str = "SAFE" if is_safe else "UNSAFE"
        box_color = (0, 255, 0) if is_safe else (0, 0, 255)

        x, y, w, h = spatial["bbox"]

        # Draw candidate box and status tag
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(annotated, f"#{idx+1} {status_str} (z={z_score:+.2f})", 
                    (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        print(f"\nCandidate #{idx+1}:")
        print(f"  Status:             {status_str} (Score z={z_score:+.4f})")
        print(f"  Normalized Area:    {features[0]:.3f}")
        print(f"  Obstacle Density:   {features[1]:.3f}")
        print(f"  Min Obstacle Dist:  {features[2]:.3f}")
        print(f"  Surface Flatness:   {features[3]:.3f}")
        print(f"  Offset:             {features[4]:.3f}")
        print(f"  Pixel Error Vector: ex={spatial['error_x']:+.1f} px, ey={spatial['error_y']:+.1f} px")

        if is_safe:
            utility_score = z_score - 0.5 * features[4]
            safe_candidates.append({
                "index": idx + 1,
                "spatial": spatial,
                "score": utility_score,
                "z": z_score
            })

    # 4. Target Selection / Abort Logic
    print("-" * 65)
    if len(safe_candidates) > 0:
        safe_candidates.sort(key=lambda item: item["score"], reverse=True)
        target = safe_candidates[0]
        sp = target["spatial"]

        # Lock visual crosshair onto target
        tcx, tcy = int(sp["centroid"][0]), int(sp["centroid"][1])
        cv2.circle(annotated, (tcx, tcy), 14, (0, 215, 255), 2)
        cv2.line(annotated, (320, 240), (tcx, tcy), (0, 215, 255), 2)

        # Telemetry HUD
        cv2.putText(annotated, f"TARGET LOCKED: #{target['index']} | STATUS: READY TO ALIGN", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(annotated, f"ERROR SETPOINTS: ex={sp['error_x']:+.1f} px | ey={sp['error_y']:+.1f} px", 
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)

        print(f"[ACTION: ALIGN & DESCEND] Locked Target #{target['index']}.")
        print(f"[*] Command to Flight Controller: Align with pixel offset ({sp['error_x']:+.1f}, {sp['error_y']:+.1f})")
    else:
        cv2.putText(annotated, "STATUS: NO SAFE LANDING ZONE DETECTED - ABORT / HOVER", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        print("[ACTION: ABORT DESCENT] No candidates satisfied safety margin (z < 0).")
        print("[*] Command to Flight Controller: Maintain hover at search altitude.")

    # 5. Save Output
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"stage5_{scenario}_decision.png")
    cv2.imwrite(out_path, annotated)
    print(f"\n[OUTPUT SAVED] Results image saved to: {os.path.abspath(out_path)}")

    # 6. Display briefly without blocking
    cv2.imshow(f"Stage 5 - {scenario}", annotated)
    cv2.waitKey(2500)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline(scenario="mixed")
    run_pipeline(scenario="all_unsafe")
    print("\n[SUCCESS] Stage 5 end-to-end perception pipeline complete.")