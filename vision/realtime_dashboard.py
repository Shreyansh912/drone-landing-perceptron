import os
import cv2
import numpy as np
from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController


def draw_hud_panel(frame, state, alt, pos, err, z_score):
    """Draws real-time flight telemetry overlay on drone video stream."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (320, 115), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    state_color = (0, 255, 0) if state in ["ALIGN", "DESCEND", "TOUCHDOWN"] else (0, 0, 255)
    cv2.putText(frame, f"FSM STATE: {state}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)
    cv2.putText(frame, f"ALTITUDE:  {alt:4.2f} m", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"POS (X,Y): ({pos[0]:+.2f}, {pos[1]:+.2f}) m", (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"RAD ERR:   {err * 100:4.1f} cm", (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"SCORE z:   {z_score:+.4f}", (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 1)
    return frame


def draw_feature_bars(width, height, features, feature_names):
    """Draws horizontal bar graph representing the normalized 6D feature vector."""
    panel = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "PERCEPTRON 6D FEATURE VECTOR", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if features is None:
        cv2.putText(panel, "No Candidate Target Selected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        return panel

    bar_max_w = width - 180
    start_y = 55
    dy = 32

    for i, (name, val) in enumerate(zip(feature_names, features)):
        y = start_y + i * dy
        cv2.putText(panel, f"{name}:", (15, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        bar_len = int(np.clip(val, 0.0, 1.0) * bar_max_w)
        cv2.rectangle(panel, (130, y), (130 + bar_max_w, y + 16), (55, 55, 55), -1)
        bar_color = (0, 200, 0) if val > 0.5 else (0, 165, 255)
        cv2.rectangle(panel, (130, y), (130 + bar_len, y + 16), bar_color, -1)
        cv2.putText(panel, f"{val:.2f}", (135 + bar_max_w, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    return panel


def draw_radar_map(width, height, history_x, history_y, current_x, current_y):
    """Draws top-down 2D trajectory tracking convergence toward the center pad."""
    radar = np.full((height, width, 3), 25, dtype=np.uint8)
    cx, cy = width // 2, height // 2

    # Concentric distance rings: 0.5m, 1.0m, 1.5m
    scale = 75.0  # pixels per meter
    for r_m in [0.5, 1.0, 1.5, 2.0]:
        cv2.circle(radar, (cx, cy), int(r_m * scale), (50, 50, 50), 1)

    # Pad boundary
    cv2.circle(radar, (cx, cy), int(0.65 * scale), (0, 180, 0), 2)
    cv2.putText(radar, "PAD (0,0)", (cx - 25, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # Historical trajectory trail
    if len(history_x) > 1:
        pts = []
        for hx, hy in zip(history_x, history_y):
            px = int(cx + hx * scale)
            py = int(cy + hy * scale)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(radar, pts[i - 1], pts[i], (0, 215, 255), 2)

    # Current drone marker
    curr_px = int(cx + current_x * scale)
    curr_py = int(cy + current_y * scale)
    cv2.circle(radar, (curr_px, curr_py), 6, (0, 0, 255), -1)
    cv2.putText(radar, "TOP-DOWN TRAJECTORY", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return radar


def run_dashboard():
    print("=" * 70)
    print("STAGE 10: REAL-TIME AUTONOMOUS LANDING DASHBOARD")
    print("=" * 70)

    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)
    evaluator = LandingZoneEvaluator()

    pid_x = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)
    pid_y = PIDController(kp=1.15, ki=0.04, kd=0.38, max_output=1.4)

    drone_pos = np.array([-1.2, 0.8, 4.5], dtype=np.float64)
    pad_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    feature_names = ["Area", "Clutter", "Min Dist", "Flatness", "Offset", "Circularity"]
    history_x, history_y = [], []

    time_elapsed = 0.0
    dt = 0.1
    state = "SEARCH"
    aligned_ticks = 0

    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_path = os.path.join(out_dir, "stage10_dashboard_snapshot.png")

    final_dashboard = None

    while time_elapsed <= 20.0:
        alt = max(drone_pos[2], 0.20)
        px_per_m = 360.0 / alt
        u0, v0 = 320.0, 240.0

        # 1. Render aerial view
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        pad_px_x = int(u0 + (pad_pos[0] - drone_pos[0]) * px_per_m)
        pad_px_y = int(v0 + (pad_pos[1] - drone_pos[1]) * px_per_m)
        pad_radius = int(0.65 * px_per_m)

        if pad_radius > 4:
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (185, 185, 185), -1)
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)
            cv2.putText(frame, "H", (pad_px_x - int(pad_radius * 0.2), pad_px_y + int(pad_radius * 0.2)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, pad_radius / 40.0), (250, 250, 250), 2)

        # 2. Perception & Classification
        _, gray, edges = preprocessor.process(frame)
        candidates, _ = detector.detect_candidates(gray)

        best_features = None
        best_target = None
        max_score = -1e9
        best_z = 0.0

        for c in candidates:
            M = cv2.moments(c)
            if M["m00"] <= 1e-4:
                continue
            area = float(M["m00"])
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            expected_area = np.pi * (0.65 * px_per_m)**2
            f1 = float(np.clip(0.80 * (area / max(expected_area, 1.0)), 0.10, 0.95))

            mask = np.zeros((480, 640), dtype=np.uint8)
            cv2.drawContours(mask, [c], -1, 255, -1)
            cand_edges = cv2.bitwise_and(edges, edges, mask=mask)
            f2 = float(np.count_nonzero(cand_edges)) / max(area, 1.0)

            inv_edges = cv2.bitwise_not(edges)
            dist_tf = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)
            f3 = float(np.clip(dist_tf[int(np.clip(cy, 0, 479)), int(np.clip(cx, 0, 639))] / (px_per_m * 0.8), 0.10, 0.95))

            _, std_val = cv2.meanStdDev(gray, mask=mask)
            f4 = 1.0 - min(1.0, float(std_val[0][0]) / 40.0)

            dx_err = cx - u0
            dy_err = cy - v0
            f5 = min(1.0, np.sqrt(dx_err**2 + dy_err**2) / np.sqrt(u0**2 + v0**2))

            perim = cv2.arcLength(c, True)
            f6 = (4.0 * np.pi * area) / (perim**2) if perim > 0 else 0.0

            feats = np.array([f1, f2, f3, f4, f5, f6])
            is_safe, z_score = evaluator.evaluate(feats)

            if is_safe and z_score >= 0.0:
                util = z_score - 0.20 * f5
                if util > max_score:
                    max_score = util
                    best_z = z_score
                    best_features = feats
                    best_target = {"error_x": dx_err, "error_y": dy_err}

        # 3. State & Control Updates
        ex_m, ey_m = 0.0, 0.0
        vz_cmd = 0.0

        if best_target is not None:
            ex_m = best_target["error_x"] / px_per_m
            ey_m = best_target["error_y"] / px_per_m
            h_err = np.sqrt(ex_m**2 + ey_m**2)

            if state in ["SEARCH", "ABORT"]:
                state = "ALIGN"

            if state == "ALIGN":
                vz_cmd = 0.0
                if h_err < 0.15:
                    aligned_ticks += 1
                    if aligned_ticks >= 3:
                        state = "DESCEND"
                else:
                    aligned_ticks = 0

            elif state == "DESCEND":
                vz_cmd = -0.32
                if h_err > 0.40:
                    state = "ALIGN"
                    vz_cmd = 0.0
                    aligned_ticks = 0
                elif drone_pos[2] <= 0.18:
                    state = "TOUCHDOWN"
                    vz_cmd = 0.0
        else:
            state = "ABORT"
            vz_cmd = 0.05
            aligned_ticks = 0

        vx_cmd = pid_x.compute(ex_m, dt) if state != "TOUCHDOWN" else 0.0
        vy_cmd = pid_y.compute(ey_m, dt) if state != "TOUCHDOWN" else 0.0

        if state != "TOUCHDOWN":
            drone_pos[0] += vx_cmd * dt
            drone_pos[1] += vy_cmd * dt
            drone_pos[2] += vz_cmd * dt
            drone_pos[2] = max(0.0, drone_pos[2])

        history_x.append(drone_pos[0])
        history_y.append(drone_pos[1])

        # 4. Assemble 4-Panel Dashboard
        curr_rad_err = np.sqrt(drone_pos[0]**2 + drone_pos[1]**2)

        # Panel 1: Aerial Camera Feed + Telemetry
        p1 = draw_hud_panel(frame, state, drone_pos[2], drone_pos, curr_rad_err, best_z)
        p1_resized = cv2.resize(p1, (480, 360))

        # Panel 2: Preprocessed Edge Map
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(edges_bgr, "CANNY EDGE FILTER & CONTOURS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        p2_resized = cv2.resize(edges_bgr, (480, 360))

        # Panel 3: Feature Vector Telemetry
        p3_resized = draw_feature_bars(480, 360, best_features, feature_names)

        # Panel 4: 2D Radar Trajectory
        p4_resized = draw_radar_map(480, 360, history_x, history_y, drone_pos[0], drone_pos[1])

        # Composite 2x2 Grid: 960 x 720
        top_row = np.hstack([p1_resized, p2_resized])
        bottom_row = np.hstack([p3_resized, p4_resized])
        final_dashboard = np.vstack([top_row, bottom_row])

        if state == "TOUCHDOWN":
            print(f"[DASHBOARD] Touchdown reached at T={time_elapsed:.1f}s. Saving snapshot...")
            cv2.imwrite(snapshot_path, final_dashboard)
            print(f"[SAVED] Dashboard snapshot saved to: {os.path.abspath(snapshot_path)}")
            break

        time_elapsed += dt

    # Display for 3.5 seconds
    cv2.imshow("Autonomous Drone Landing - Real-Time Dashboard", final_dashboard)
    cv2.waitKey(3500)
    cv2.destroyAllWindows()
    print("[SUCCESS] Real-Time Dashboard run completed.")


if __name__ == "__main__":
    run_dashboard()