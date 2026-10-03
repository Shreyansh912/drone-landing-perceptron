import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController
from aruco_tracker import ArUcoTargetTracker
from kalman_target_tracker import TargetKalmanFilter


def run_moving_target_landing():
    print("=" * 70)
    print("STAGE 12: DYNAMIC MOVING PLATFORM PRECISION LANDING + KALMAN FILTER")
    print("=" * 70)

    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)
    evaluator = LandingZoneEvaluator()
    aruco_tracker = ArUcoTargetTracker(marker_id=0, marker_size_m=0.22)
    kf = TargetKalmanFilter(dt=0.1)

    # PID controllers with feedforward velocity support
    pid_x = PIDController(kp=1.35, ki=0.06, kd=0.45, max_output=1.8)
    pid_y = PIDController(kp=1.35, ki=0.06, kd=0.45, max_output=1.8)

    # Initial states
    drone_pos = np.array([-1.5, 1.0, 4.5], dtype=np.float64)
    pad_pos = np.array([0.0, -0.4, 0.0], dtype=np.float64)
    pad_vel = np.array([0.22, 0.08, 0.0], dtype=np.float64)  # AGV moving at ~23 cm/s

    dt = 0.1
    time_elapsed = 0.0
    state = "SEARCH"
    aligned_ticks = 0

    history_t = []
    history_alt = []
    history_err = []
    history_pad_x, history_pad_y = [], []
    history_drone_x, history_drone_y = [], []

    print(f"[TARGET] Moving Landing Pad initialized with velocity: ({pad_vel[0]:+.2f}, {pad_vel[1]:+.2f}) m/s")

    while time_elapsed <= 30.0:
        # Update physical pad position (ground vehicle motion)
        pad_pos[0] += pad_vel[0] * dt
        pad_pos[1] += pad_vel[1] * dt

        alt = max(drone_pos[2], 0.15)
        px_per_m = 360.0 / alt
        u0, v0 = 320.0, 240.0

        # 1. Render aerial camera view
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        pad_px_x = int(u0 + (pad_pos[0] - drone_pos[0]) * px_per_m)
        pad_px_y = int(v0 + (pad_pos[1] - drone_pos[1]) * px_per_m)
        pad_radius = int(0.65 * px_per_m)

        if pad_radius > 4:
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (185, 185, 185), -1)
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)
            frame = aruco_tracker.draw_marker_on_pad(frame, (pad_px_x, pad_px_y), pad_radius)

        # 2. Perception & Sensor Update
        aruco_found, tvec, _ = aruco_tracker.detect_pose(frame)
        raw_meas_target_pos = None

        if alt <= 1.60 and aruco_found:
            # Measure world position from camera pose estimate
            meas_target_x = drone_pos[0] + float(tvec[0])
            meas_target_y = drone_pos[1] + float(tvec[1])
            raw_meas_target_pos = (meas_target_x, meas_target_y)
        else:
            _, gray, edges = preprocessor.process(frame)
            candidates, _ = detector.detect_candidates(gray)

            best_target = None
            max_score = -1e9

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
                inner_r = max(2, int(0.35 * px_per_m))
                cv2.circle(mask, (int(cx), int(cy)), inner_r, 0, -1)

                cand_edges = cv2.bitwise_and(edges, edges, mask=mask)
                eval_area = max(float(np.count_nonzero(mask)), 1.0)
                f2 = float(np.count_nonzero(cand_edges)) / eval_area

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
                        best_target = {"meas_x": drone_pos[0] + dx_err / px_per_m,
                                       "meas_y": drone_pos[1] + dy_err / px_per_m}

            if best_target is not None:
                raw_meas_target_pos = (best_target["meas_x"], best_target["meas_y"])

        # 3. Kalman Filter Predict & Update
        kf.predict()
        if raw_meas_target_pos is not None:
            kf_state = kf.update(raw_meas_target_pos[0], raw_meas_target_pos[1])
        else:
            kf_state = kf.get_state()

        est_target_x, est_target_y, est_target_vx, est_target_vy = kf_state

        # 4. Error and FSM State Management
        ex_m = est_target_x - drone_pos[0]
        ey_m = est_target_y - drone_pos[1]
        h_err = np.sqrt(ex_m**2 + ey_m**2)

        vz_cmd = 0.0

        if alt <= 0.35 and h_err < 0.12:
            state = "TERMINAL_FLARE"
            vz_cmd = -0.25
        elif alt <= 1.60 and aruco_found:
            state = "ARUCO_LOCK"
            vz_cmd = -0.28
        elif raw_meas_target_pos is not None:
            if state in ["SEARCH", "ABORT"]:
                state = "ALIGN"

            if state == "ALIGN":
                vz_cmd = 0.0
                if h_err < 0.20:
                    aligned_ticks += 1
                    if aligned_ticks >= 3:
                        state = "DESCEND"
                else:
                    aligned_ticks = 0
            elif state == "DESCEND":
                vz_cmd = -0.30
        else:
            state = "ABORT"
            vz_cmd = 0.05
            aligned_ticks = 0

        if drone_pos[2] <= 0.16:
            state = "TOUCHDOWN"
            vz_cmd = 0.0

        # 5. PID + Velocity Feedforward Actuation
        if state != "TOUCHDOWN":
            # Add estimated target velocity as feedforward to eliminate tracking lag
            vx_cmd = pid_x.compute(ex_m, dt) + est_target_vx
            vy_cmd = pid_y.compute(ey_m, dt) + est_target_vy
        else:
            vx_cmd, vy_cmd, vz_cmd = 0.0, 0.0, 0.0

        if state != "TOUCHDOWN":
            drone_pos[0] += vx_cmd * dt
            drone_pos[1] += vy_cmd * dt
            drone_pos[2] += vz_cmd * dt
            drone_pos[2] = max(0.0, drone_pos[2])

        radial_err = np.sqrt((drone_pos[0] - pad_pos[0])**2 + (drone_pos[1] - pad_pos[1])**2)

        history_t.append(time_elapsed)
        history_alt.append(drone_pos[2])
        history_err.append(radial_err * 100.0)
        history_drone_x.append(drone_pos[0])
        history_drone_y.append(drone_pos[1])
        history_pad_x.append(pad_pos[0])
        history_pad_y.append(pad_pos[1])

        if int(time_elapsed * 10) % 10 == 0:
            print(f"[MOVING-FLIGHT] T={time_elapsed:4.1f}s | State={state:<14} | Alt={drone_pos[2]:4.2f}m | "
                  f"EstVel=({est_target_vx:+.2f}, {est_target_vy:+.2f})m/s | RadialErr={radial_err * 100:5.2f}cm")

        if state == "TOUCHDOWN":
            print("\n" + "=" * 70)
            print(f"[TOUCHDOWN] Landed on moving platform at T = {time_elapsed:.1f}s!")
            print(f"Final Relative Accuracy on Moving Deck: {radial_err * 100:.2f} cm from center.")
            print(f"Target Pad Final Position: ({pad_pos[0]:.2f}, {pad_pos[1]:.2f}) m")
            print("=" * 70)
            break

        time_elapsed += dt

    # 6. Plot Moving Intercept Trajectory
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "stage12_moving_target_trajectory.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: 2D Spatial Intercept Path
    ax1.plot(history_pad_x, history_pad_y, 'g--', lw=2.5, label="Moving Pad Path")
    ax1.plot(history_drone_x, history_drone_y, 'b-', lw=2, label="Drone Flight Path")
    ax1.scatter([history_drone_x[-1]], [history_drone_y[-1]], color='red', s=90, zorder=5, label="Touchdown Point")
    ax1.set_title("2D Ground-Plane Dynamic Intercept")
    ax1.set_xlabel("X Position [m]")
    ax1.set_ylabel("Y Position [m]")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Right: Altitude & Relative Error Over Time
    ax2.plot(history_t, history_alt, color="blue", lw=2, label="Altitude (m)")
    ax2.plot(history_t, [e / 100.0 for e in history_err], color="crimson", lw=2, label="Relative Error (m)")
    ax2.axhline(0.35, color="green", linestyle=":", label="Terminal Flare Cutoff (0.35m)")
    ax2.set_title("Descent & Relative Tracking Convergence")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Meters [m]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT SAVED] Moving landing plot saved to: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    run_moving_target_landing()