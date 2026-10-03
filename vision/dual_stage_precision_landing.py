import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to avoid Tk asset/GUI errors
import matplotlib.pyplot as plt
import cv2
import numpy as np

from preprocessor import ImagePreprocessor
from landing_detector import LandingZoneDetector
from landing_evaluator import LandingZoneEvaluator
from pid_controller import PIDController
from aruco_tracker import ArUcoTargetTracker


def run_dual_stage_landing():
    print("=" * 70)
    print("STAGE 11: DUAL-STAGE PERCEPTRON + ARUCO PRECISION LANDING")
    print("=" * 70)

    preprocessor = ImagePreprocessor()
    detector = LandingZoneDetector(min_area=150.0, max_area=500000.0)
    evaluator = LandingZoneEvaluator()
    aruco_tracker = ArUcoTargetTracker(marker_id=0, marker_size_m=0.22)

    # Lateral PID controllers
    pid_x = PIDController(kp=1.25, ki=0.04, kd=0.38, max_output=1.4)
    pid_y = PIDController(kp=1.25, ki=0.04, kd=0.38, max_output=1.4)

    # Initial drone pose: offset (-1.2m, 0.8m) at 4.5m altitude
    drone_pos = np.array([-1.2, 0.8, 4.5], dtype=np.float64)
    pad_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    dt = 0.1
    time_elapsed = 0.0
    state = "SEARCH"
    aligned_ticks = 0

    history_t = []
    history_alt = []
    history_err = []
    history_state = []

    print("[SYSTEM] Perception initialized: Stage 1 = Perceptron, Stage 2 = ArUco PnP, Stage 3 = Terminal Flare.")

    while time_elapsed <= 25.0:
        alt = max(drone_pos[2], 0.15)
        px_per_m = 360.0 / alt
        u0, v0 = 320.0, 240.0

        # 1. Render aerial camera scene with pad and ArUco marker
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        pad_px_x = int(u0 + (pad_pos[0] - drone_pos[0]) * px_per_m)
        pad_px_y = int(v0 + (pad_pos[1] - drone_pos[1]) * px_per_m)
        pad_radius = int(0.65 * px_per_m)

        if pad_radius > 4:
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (185, 185, 185), -1)
            cv2.circle(frame, (pad_px_x, pad_px_y), pad_radius, (230, 230, 230), 2)
            frame = aruco_tracker.draw_marker_on_pad(frame, (pad_px_x, pad_px_y), pad_radius)

        # 2. Perception & Multi-Stage State Machine
        aruco_found, tvec, _ = aruco_tracker.detect_pose(frame)
        ex_m, ey_m = 0.0, 0.0
        vz_cmd = 0.0

        # Stage 3: Terminal Flare below 0.35m to prevent FOV clipping stall
        if (state in ["ARUCO_LOCK", "TERMINAL_FLARE"]) and alt <= 0.35:
            state = "TERMINAL_FLARE"
            ex_m = 0.0 - drone_pos[0]
            ey_m = 0.0 - drone_pos[1]
            vz_cmd = -0.25

        # Stage 2: ArUco PnP Metric Tracking (between 0.35m and 1.6m)
        elif alt <= 1.60 and aruco_found:
            state = "ARUCO_LOCK"
            ex_m = float(tvec[0])
            ey_m = float(tvec[1])
            vz_cmd = -0.28

        # Stage 1: Classical Vision + Perceptron Safety Evaluation (above 1.6m)
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

                # Mask out the inner fiducial area so marker edges don't spoil surface flatness
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
                        best_target = {"error_x": dx_err, "error_y": dy_err}

            if best_target is not None:
                ex_m = best_target["error_x"] / px_per_m
                ey_m = best_target["error_y"] / px_per_m
                target_h_err = np.sqrt(ex_m**2 + ey_m**2)

                if state in ["SEARCH", "ABORT"]:
                    state = "ALIGN"

                if state == "ALIGN":
                    vz_cmd = 0.0
                    if target_h_err < 0.15:
                        aligned_ticks += 1
                        if aligned_ticks >= 3:
                            state = "DESCEND"
                    else:
                        aligned_ticks = 0
                elif state == "DESCEND":
                    vz_cmd = -0.32
            else:
                state = "ABORT"
                vz_cmd = 0.05
                aligned_ticks = 0

        # Disarm motors at ground threshold
        if drone_pos[2] <= 0.16:
            state = "TOUCHDOWN"
            vz_cmd = 0.0

        # 3. PID Closed-Loop Flight Actuation
        vx_cmd = pid_x.compute(ex_m, dt) if state != "TOUCHDOWN" else 0.0
        vy_cmd = pid_y.compute(ey_m, dt) if state != "TOUCHDOWN" else 0.0

        if state != "TOUCHDOWN":
            drone_pos[0] += vx_cmd * dt
            drone_pos[1] += vy_cmd * dt
            drone_pos[2] += vz_cmd * dt
            drone_pos[2] = max(0.0, drone_pos[2])

        rad_error = np.sqrt(drone_pos[0]**2 + drone_pos[1]**2)

        history_t.append(time_elapsed)
        history_alt.append(drone_pos[2])
        history_err.append(rad_error * 100.0)
        history_state.append(state)

        if int(time_elapsed * 10) % 10 == 0:
            print(f"[FLIGHT] T={time_elapsed:4.1f}s | State={state:<14} | Alt={drone_pos[2]:4.2f}m | Error={rad_error * 100:5.2f}cm")

        if state == "TOUCHDOWN":
            print("\n" + "=" * 70)
            print(f"[TOUCHDOWN] Final Touchdown at T = {time_elapsed:.1f}s!")
            print(f"Final Radial Accuracy: {rad_error * 100:.2f} cm from center.")
            print("=" * 70)
            break

        time_elapsed += dt

    # 4. Generate & Save Trajectory Plot
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, "stage11_dual_stage_trajectory.png")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(history_t, history_alt, color="royalblue", lw=2, label="Altitude (m)")
    ax1.axhline(1.6, color="darkorange", linestyle="--", label="ArUco Transition Threshold (1.6m)")
    ax1.axhline(0.35, color="green", linestyle=":", label="Terminal Flare Cutoff (0.35m)")
    ax1.set_ylabel("Altitude [m]")
    ax1.set_title("Dual-Stage Precision Landing Convergence Profile")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(history_t, history_err, color="crimson", lw=2, label="Radial Error (cm)")
    ax2.set_ylabel("Error [cm]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT SAVED] Trajectory plot exported to: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    run_dual_stage_landing()