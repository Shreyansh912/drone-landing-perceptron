import os
import time
import numpy as np
import pyvista as pv


def build_quadrotor_mesh(center=(0, 0, 0), arm_length=0.25):
    """Constructs a composite 3D geometric mesh of a quadrotor airframe."""
    x, y, z = center
    d = arm_length / np.sqrt(2)

    # 1. Central Avionics Fuselage Hub
    body = pv.Cylinder(center=(x, y, z), direction=(0, 0, 1), radius=0.07, height=0.04)

    # 2. Cross Structural Arms (X-Configuration)
    arm1 = pv.Cylinder(center=(x, y, z), direction=(1, 1, 0), radius=0.012, height=arm_length * 2)
    arm2 = pv.Cylinder(center=(x, y, z), direction=(1, -1, 0), radius=0.012, height=arm_length * 2)

    # 3. Motor Mounts & Rotor Blade Disks
    rotor_offsets = [(d, d), (-d, -d), (d, -d), (-d, d)]
    rotors = []
    for ox, oy in rotor_offsets:
        motor = pv.Cylinder(center=(x + ox, y + oy, z + 0.02), direction=(0, 0, 1), radius=0.02, height=0.03)
        disk = pv.Cylinder(center=(x + ox, y + oy, z + 0.035), direction=(0, 0, 1), radius=0.09, height=0.005)
        rotors.extend([motor, disk])

    # Combine into a single unified airframe mesh
    drone_mesh = body.merge([arm1, arm2] + rotors)
    return drone_mesh


def build_camera_frustum_lines(center, ground_z=0.0, fov_deg=65.0):
    """Constructs downward camera field-of-view (FOV) pyramid lines."""
    x, y, z = center
    half_fov_rad = np.radians(fov_deg / 2.0)
    h = max(0.01, z - ground_z)
    span = h * np.tan(half_fov_rad)

    corners = [
        [x - span, y - span, ground_z],
        [x + span, y - span, ground_z],
        [x + span, y + span, ground_z],
        [x - span, y + span, ground_z]
    ]

    points = np.array([[x, y, z]] + corners, dtype=np.float64)

    # Line connections: apex to each base corner, plus rectangular perimeter
    lines = [
        2, 0, 1,
        2, 0, 2,
        2, 0, 3,
        2, 0, 4,
        2, 1, 2,
        2, 2, 3,
        2, 3, 4,
        2, 4, 1
    ]
    frustum_mesh = pv.PolyData(points, lines=lines)
    return frustum_mesh


def run_3d_prototype_viewer():
    print("=" * 70)
    print("3D DRONE PROTOTYPE & FLIGHT SIMULATION VIEWER (VS CODE)")
    print("=" * 70)

    # Initialize PyVista Plotter Window
    plotter = pv.Plotter(title="Autonomous Drone 3D Precision Landing Prototype", window_size=(1024, 768))
    plotter.set_background("#18191c")

    # 1. Build Ground Landing Pad (Radius = 0.65m)
    pad_disk = pv.Disc(center=(0, 0, 0.005), inner=0.0, outer=0.65, normal=(0, 0, 1), r_res=2, c_res=40)
    plotter.add_mesh(pad_disk, color="#e0e0e0", show_edges=True, edge_color="#a0a0a0")

    # 2. Build Center ArUco Marker Plate (0.22m x 0.22m)
    marker_plate = pv.Plane(center=(0, 0, 0.01), direction=(0, 0, 1), i_size=0.22, j_size=0.22)
    plotter.add_mesh(marker_plate, color="#111111")

    # 3. Add Ground Coordinate Reference Grid
    ground_grid = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=6.0, j_size=6.0)
    plotter.add_mesh(ground_grid, color="#2b2d30", show_edges=True, edge_color="#3c3f41", opacity=0.8)

    # 4. Generate 3D Multi-Stage Convergence Trajectory
    num_frames = 120
    t = np.linspace(0, 17, num_frames)
    decay = np.exp(-t / 3.8)
    traj_x = -1.2 * decay
    traj_y = 0.8 * decay
    traj_z = np.clip(4.5 - (4.5 - 0.16) * (t / 17.0), 0.16, 4.5)

    traj_points = np.column_stack([traj_x, traj_y, traj_z])
    traj_spline = pv.Spline(traj_points, 400)
    plotter.add_mesh(traj_spline, color="#00ffff", line_width=3, label="Descent Trajectory")

    # 5. Add Initial Drone Mesh and Camera FOV Actor
    initial_drone = build_quadrotor_mesh(center=traj_points[0])
    drone_actor = plotter.add_mesh(initial_drone, color="#ff4757", pbr=True, metallic=0.7, roughness=0.3)

    initial_frustum = build_camera_frustum_lines(center=traj_points[0])
    frustum_actor = plotter.add_mesh(initial_frustum, color="#ffda79", line_width=1.5, opacity=0.7, label="Downward Camera FOV")

    # Camera Perspective Positioning
    plotter.camera_position = [(-3.6, -3.6, 4.2), (0, 0, 1.2), (0, 0, 1)]
    plotter.add_axes()
    plotter.add_legend(bcolor=None)

    print("[*] Displaying 3D scene. Controls: Rotate (Left Click), Zoom (Scroll), Pan (Shift + Left Click).")
    plotter.show(auto_close=False, interactive_update=True)

    # 6. Execute Flight Animation
    for i in range(num_frames):
        pos = traj_points[i]
        updated_mesh = build_quadrotor_mesh(center=pos)
        updated_frustum = build_camera_frustum_lines(center=pos)

        drone_actor.mapper.SetInputData(updated_mesh)
        frustum_actor.mapper.SetInputData(updated_frustum)

        plotter.update()
        time.sleep(0.04)

    # 7. Export High-Resolution 3D Snapshot for README & Portfolio
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    snapshot_3d_path = os.path.join(out_dir, "stage13_3d_prototype_render.png")
    plotter.screenshot(snapshot_3d_path)
    print(f"\n[RENDER SAVED] 3D prototype snapshot saved to: {os.path.abspath(snapshot_3d_path)}")

    print("[SUCCESS] 3D landing animation completed. You can interact with the 3D viewport or close the window.")
    plotter.show(interactive=True)


if __name__ == "__main__":
    run_3d_prototype_viewer()