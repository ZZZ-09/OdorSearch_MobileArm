"""在 OdorSim/GADEN 中验证 OdorSearch_MobileArm 搜索算法。

用法（必须在已 source OdorSim/setup/activate.sh 的 WSL shell 中运行）:
    cd /mnt/c/.../OdorSearch_MobileArm
    python scripts/run_odorsim_verification.py --seed 42 --max-steps 3000

输出：
    outputs/odorsim/history_seed<N>.npz      搜索历史
    outputs/odorsim/summary_seed<N>.png      轨迹/浓度摘要图
    outputs/odorsim/video_seed<N>.mp4        三维搜索过程视频（单视图或三视角）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# OdorSim imports (available after sourcing setup/activate.sh)
from odor_sim.bridge.gaden_bridge import GadenBridge  # type: ignore
from odor_sim.runtime.gaden_server import GadenServerManager  # type: ignore

from src.odor_sim_adapter import GadenBackedWarehouseEnv
from src.simulation_odorsim import OdorSearchSessionOdorSim
from src.utils import load_yaml
from src.visualization import TrajectoryPlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the mobile-arm odor search algorithm in OdorSim/GADEN."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode.")
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(PROJECT_ROOT / "odorsim_scenarios" / "warehouse_8x8" / "environment_configurations" / "config1"),
        help="Path to GADEN scenario config directory.",
    )
    parser.add_argument("--scene-id", type=str, default="scene1", help="GADEN scene id.")
    parser.add_argument("--gaden-dt", type=float, default=0.05, help="GADEN timestep [s].")
    parser.add_argument(
        "--env-config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "warehouse_empty_8x8.yaml"),
        help="Warehouse env config (default: empty 8x8 room).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "odorsim"),
        help="Directory for history, plots and video.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Video frame rate.")
    parser.add_argument(
        "--video-stride",
        type=int,
        default=2,
        help="Render one video frame every N simulation steps.",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video generation.")
    parser.add_argument(
        "--start-x", type=float, default=-3.0, help="Initial robot base X."
    )
    parser.add_argument(
        "--start-y", type=float, default=-3.0, help="Initial robot base Y."
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for independent evaluation success [m].",
    )
    parser.add_argument(
        "--video-layout",
        type=str,
        choices=["3d", "triple"],
        default="triple",
        help="Video layout: '3d' for single 3D view, 'triple' for 3D + top + front views.",
    )
    return parser.parse_args()


class VideoRecorder:
    """Render matplotlib frames into an MP4 of the search process.

    Supports two layouts:
      - '3d': single 3D view (legacy).
      - 'triple': three synchronized subplots (3D, top-down XY, front XZ).
    """

    def __init__(self, env, robot, fps: int = 10, layout: str = "triple"):
        self.env = env
        self.robot = robot
        self.fps = fps
        self.layout = layout
        self.frames: list[np.ndarray] = []

    # ------------------------------------------------------------------ #
    # 3D helpers ( reused from the original implementation )
    # ------------------------------------------------------------------ #
    def _draw_static_no_cloud(self, ax) -> None:
        """Draw obstacles and source only (skip expensive concentration cloud)."""
        low, high = self.env.bounds_low, self.env.bounds_high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Warehouse Odor Source Search (GADEN-backed)")

        for obs in self.env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            if obs["type"] == "box":
                self._draw_box(ax, obs["center"], obs["size"], color="darkred", alpha=0.55, linewidth=1.5)
                self._draw_box_surface(ax, obs["center"], obs["size"], color="darkred", alpha=0.18)
            elif obs["type"] == "cylinder":
                self._draw_cylinder(
                    ax, obs["center"], obs["axis"], obs["radius"], obs["length"],
                    color="darkblue", alpha=0.55, linewidth=1.5,
                )
                self._draw_cylinder_surface(
                    ax, obs["center"], obs["axis"], obs["radius"], obs["length"],
                    color="darkblue", alpha=0.18,
                )

        ax.scatter(
            *self.env.source_pos,
            color="red",
            s=200,
            marker="*",
            label="odor source",
        )

    @staticmethod
    def _draw_box(ax, center, size, color="gray", alpha=0.4, linewidth=1.0) -> None:
        c = np.asarray(center, dtype=float)
        s = np.asarray(size, dtype=float)
        corners = c + 0.5 * s * np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            pts = corners[[i, j]]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def _draw_box_surface(ax, center, size, color="gray", alpha=0.15) -> None:
        c = np.asarray(center, dtype=float)
        s = np.asarray(size, dtype=float)
        faces = [
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
        corners = c + 0.5 * s * np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
        )
        for face in faces:
            pts = corners[face]
            ax.plot_surface(
                pts[:, 0].reshape(2, 2),
                pts[:, 1].reshape(2, 2),
                pts[:, 2].reshape(2, 2),
                color=color,
                alpha=alpha,
                shade=False,
            )

    @staticmethod
    def _draw_cylinder(ax, center, axis, radius, length, color="steelblue", alpha=0.4, linewidth=1.0) -> None:
        c = np.asarray(center, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(a, u)
        theta = np.linspace(0.0, 2.0 * np.pi, 30)
        circle = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        R = np.column_stack([u, v, a])
        for z in [-length / 2.0, length / 2.0]:
            local = circle.copy()
            local[2, :] = z
            world = (R @ local).T + c
            ax.plot(world[:, 0], world[:, 1], world[:, 2], color=color, alpha=alpha, linewidth=linewidth)
        for i in [0, 10, 20]:
            local = circle[:, i]
            p1 = c + R @ np.array([local[0], local[1], -length / 2.0])
            p2 = c + R @ np.array([local[0], local[1], length / 2.0])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=color, alpha=alpha, linewidth=linewidth)

    @staticmethod
    def _draw_cylinder_surface(ax, center, axis, radius, length, color="steelblue", alpha=0.15) -> None:
        c = np.asarray(center, dtype=float)
        a = np.asarray(axis, dtype=float)
        a = a / (np.linalg.norm(a) + 1e-12)
        tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, tmp)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(a, u)
        R = np.column_stack([u, v, a])
        theta = np.linspace(0.0, 2.0 * np.pi, 30)
        z_vals = np.linspace(-length / 2.0, length / 2.0, 8)
        theta_grid, z_grid = np.meshgrid(theta, z_vals)
        local = np.stack(
            [
                radius * np.cos(theta_grid),
                radius * np.sin(theta_grid),
                z_grid,
            ],
            axis=-1,
        )
        world = np.einsum("ij,mnj->mni", R, local) + c
        ax.plot_surface(
            world[..., 0],
            world[..., 1],
            world[..., 2],
            color=color,
            alpha=alpha,
            shade=False,
        )

    # ------------------------------------------------------------------ #
    # 2D projection helpers for triple-view layout
    # ------------------------------------------------------------------ #
    def _draw_obstacles_top(self, ax) -> None:
        """Draw obstacle footprints in the XY (top-down) plane."""
        for obs in self.env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            c = obs["center"]
            if obs["type"] == "box":
                s = obs["size"]
                rect = Rectangle(
                    (c[0] - s[0] / 2.0, c[1] - s[1] / 2.0),
                    s[0],
                    s[1],
                    color="darkred",
                    alpha=0.35,
                )
                ax.add_patch(rect)
            elif obs["type"] == "cylinder":
                patch = self._cylinder_xy_patch(obs, color="darkblue", alpha=0.35)
                if patch is not None:
                    ax.add_patch(patch)

    def _draw_obstacles_front(self, ax) -> None:
        """Draw obstacle projections in the XZ (front) plane, looking along -Y."""
        for obs in self.env.obstacles:
            if obs["name"].endswith("_wall"):
                continue
            c = obs["center"]
            if obs["type"] == "box":
                s = obs["size"]
                rect = Rectangle(
                    (c[0] - s[0] / 2.0, c[2] - s[2] / 2.0),
                    s[0],
                    s[2],
                    color="darkred",
                    alpha=0.35,
                )
                ax.add_patch(rect)
            elif obs["type"] == "cylinder":
                patch = self._cylinder_xz_patch(obs, color="darkblue", alpha=0.35)
                if patch is not None:
                    ax.add_patch(patch)

    @staticmethod
    def _cylinder_xy_patch(obs, color="steelblue", alpha=0.35):
        """Return a matplotlib patch for a cylinder projected onto XY."""
        c = obs["center"]
        r = obs["radius"]
        length = obs["length"]
        axis = np.asarray(obs["axis"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        if np.allclose(np.abs(axis), [0, 0, 1]):
            return Circle((c[0], c[1]), r, color=color, alpha=alpha)
        if np.allclose(np.abs(axis), [1, 0, 0]):
            return Rectangle((c[0] - length / 2.0, c[1] - r), length, 2 * r, color=color, alpha=alpha)
        if np.allclose(np.abs(axis), [0, 1, 0]):
            return Rectangle((c[0] - r, c[1] - length / 2.0), 2 * r, length, color=color, alpha=alpha)
        # Fallback: bounding circle in XY
        return Circle((c[0], c[1]), max(r, length / 2.0), color=color, alpha=alpha)

    @staticmethod
    def _cylinder_xz_patch(obs, color="steelblue", alpha=0.35):
        """Return a matplotlib patch for a cylinder projected onto XZ."""
        c = obs["center"]
        r = obs["radius"]
        length = obs["length"]
        axis = np.asarray(obs["axis"], dtype=float)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        if np.allclose(np.abs(axis), [0, 1, 0]):
            return Circle((c[0], c[2]), r, color=color, alpha=alpha)
        if np.allclose(np.abs(axis), [1, 0, 0]):
            return Rectangle((c[0] - length / 2.0, c[2] - r), length, 2 * r, color=color, alpha=alpha)
        if np.allclose(np.abs(axis), [0, 0, 1]):
            return Rectangle((c[0] - r, c[2] - length / 2.0), 2 * r, length, color=color, alpha=alpha)
        # Fallback: bounding circle in XZ
        return Circle((c[0], c[2]), max(r, length / 2.0), color=color, alpha=alpha)

    def _draw_3d_panel(self, ax, history: list[dict]) -> None:
        """Render the legacy 3D panel."""
        self._draw_static_no_cloud(ax)

        trajectory = np.array([rec["ee_pos"] for rec in history])
        if len(trajectory) > 0:
            self.robot.base_pose = history[-1]["base_pose"]
            self.robot.set_joint_angles(history[-1]["joint_angles_deg"])

            # robot base: wireframe + surface for correct occlusion
            center = self.robot.base_position.copy()
            center[2] = self.robot.height / 2.0
            base_size = np.array([self.robot.length, self.robot.width, self.robot.height])
            self._draw_box(
                ax, center, base_size,
                color="darkorange", alpha=0.85, linewidth=1.5,
            )
            self._draw_box_surface(
                ax, center, base_size,
                color="darkorange", alpha=0.45,
            )

            # arm
            if self.robot._fk_cache is None:
                self.robot._compute_fk()
            points = [self.robot._fk_cache["t_base"]]
            for T in self.robot._fk_cache["T_list"]:
                points.append(T[:3, 3].copy())
            points = np.array(points)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], "b-o", linewidth=2, markersize=3)

            # trajectory
            ax.plot(
                trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                "g-", alpha=0.5, linewidth=1.2, label="EE trajectory",
            )

        ax.legend(loc="upper right")

    def _draw_top_panel(self, ax, history: list[dict]) -> None:
        """Render the top-down (XY) panel."""
        low, high = self.env.bounds_low, self.env.bounds_high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Top view (XY)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        self._draw_obstacles_top(ax)
        ax.scatter(*self.env.source_pos[:2], color="red", s=200, marker="*", label="odor source", zorder=5)

        if not history:
            return

        base_positions = np.array([rec["base_pose"][:3] for rec in history])
        ee_positions = np.array([rec["ee_pos"] for rec in history])

        ax.plot(base_positions[:, 0], base_positions[:, 1], "b-", alpha=0.6, linewidth=1.5, label="base path")
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], "g-", alpha=0.4, linewidth=1.0, label="EE path")

        # current positions
        ax.scatter(base_positions[-1, 0], base_positions[-1, 1], color="darkorange", s=100, zorder=4, label="base")
        ax.scatter(ee_positions[-1, 0], ee_positions[-1, 1], color="purple", s=60, zorder=4, label="EE")
        ax.legend(loc="upper right", fontsize=7)

    def _draw_front_panel(self, ax, history: list[dict]) -> None:
        """Render the front (XZ) panel, looking along -Y, to show arm up/down."""
        low, high = self.env.bounds_low, self.env.bounds_high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[2], high[2])
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Front view (XZ)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        self._draw_obstacles_front(ax)
        ax.scatter(self.env.source_pos[0], self.env.source_pos[2], color="red", s=200, marker="*", label="odor source", zorder=5)

        if not history:
            return

        ee_positions = np.array([rec["ee_pos"] for rec in history])
        ax.plot(ee_positions[:, 0], ee_positions[:, 2], "g-", alpha=0.4, linewidth=1.0, label="EE path")

        # current robot arm projected to XZ
        self.robot.base_pose = history[-1]["base_pose"]
        self.robot.set_joint_angles(history[-1]["joint_angles_deg"])
        if self.robot._fk_cache is None:
            self.robot._compute_fk()
        points = [self.robot._fk_cache["t_base"]]
        for T in self.robot._fk_cache["T_list"]:
            points.append(T[:3, 3].copy())
        points = np.array(points)
        ax.plot(points[:, 0], points[:, 2], "b-o", linewidth=2, markersize=3, label="arm (XZ)")

        # base top/bottom indicator
        base_x = self.robot.base_position[0]
        base_z = self.robot.height / 2.0
        ax.scatter(base_x, base_z, color="darkorange", s=100, zorder=4, label="base")
        ax.legend(loc="upper right", fontsize=7)

    def capture(self, history: list[dict], force: bool = False) -> None:
        if self.layout == "3d":
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection="3d")
            self._draw_3d_panel(ax, history)
        else:
            fig = plt.figure(figsize=(18.08, 6.08))
            ax3d = fig.add_subplot(131, projection="3d")
            ax_top = fig.add_subplot(132)
            ax_front = fig.add_subplot(133)

            self._draw_3d_panel(ax3d, history)
            self._draw_top_panel(ax_top, history)
            self._draw_front_panel(ax_front, history)

            step = history[-1]["step"] if history else 0
            state = history[-1]["state"] if history else "INIT"
            fig.suptitle(f"Odor Search — step {step} | state {state}", fontsize=14, fontweight="bold")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(np.ascontiguousarray(frame))
        plt.close(fig)

    def save(self, path: Path) -> None:
        if not self.frames:
            print("[video] no frames captured")
            return
        writer = imageio.get_writer(path, fps=self.fps, codec="libx264")
        for f in self.frames:
            writer.append_data(f)
        writer.close()
        print(f"[video] saved {path} ({len(self.frames)} frames)")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OdorSim/GADEN Verification for OdorSearch_MobileArm")
    print("=" * 70)
    print(f"Scenario : {args.scenario}")
    print(f"Scene ID : {args.scene_id}")
    print(f"Seed     : {args.seed}")
    print(f"Max steps: {args.max_steps}")
    print(f"Layout   : {args.video_layout}")
    print("-" * 70)

    server: GadenServerManager | None = None
    bridge: GadenBridge | None = None

    try:
        # ------------------------------------------------------------------ #
        # 1. Start GADEN server
        # ------------------------------------------------------------------ #
        print("[1/5] Starting odor_gaden_rt server...")
        server = GadenServerManager(
            scenario_path=args.scenario,
            scene_id=args.scene_id,
            step_on_timer=False,
            publish_markers=False,
            log_dir=Path(args.out_dir) / "gaden_logs",
        )
        server.start(timeout=120.0, kill_stale=True)
        print(f"      Server ready (log: {server.log_path})")

        # ------------------------------------------------------------------ #
        # 2. Connect ROS bridge
        # ------------------------------------------------------------------ #
        print("[2/5] Connecting GadenBridge...")
        bridge = GadenBridge(node_name="odor_search_bridge")
        if not bridge.wait_for_server(timeout=60.0):
            raise RuntimeError("GADEN /odor_value service did not become available.")
        print("      Bridge connected")

        # ------------------------------------------------------------------ #
        # 3. Build GADEN-backed environment and session
        # ------------------------------------------------------------------ #
        print("[3/5] Building GADEN-backed warehouse environment...")
        env_cfg = load_yaml(args.env_config)
        env = GadenBackedWarehouseEnv(
            config=env_cfg,
            seed=args.seed,
            bridge=bridge,
            gaden_dt=args.gaden_dt,
        )
        print(f"      Source position: {env.source_pos}")
        print(f"      Wind: {env.wind}")

        session = OdorSearchSessionOdorSim(env=env, seed=args.seed)

        base_pose = np.array([args.start_x, args.start_y, 0.0, 0.0], dtype=float)
        obs = session.reset(base_pose=base_pose)
        print(f"      Initial base pose: {obs['base_pose']}")
        print(f"      Initial EE position: {obs['ee_pos']}")

        # ------------------------------------------------------------------ #
        # 4. Run search loop
        # ------------------------------------------------------------------ #
        print("[4/5] Running search algorithm against GADEN...")
        recorder = VideoRecorder(env, session.robot, fps=args.fps, layout=args.video_layout)
        t_start = time.time()

        while not session.done and session.step_count < args.max_steps:
            _, done, info = session.step()

            if session.step_count % args.video_stride == 0 and not args.no_video:
                recorder.capture(session.history)

            if session.step_count % 200 == 0 or done:
                rec = session.history[-1]
                ee_ppm = rec["sensor_readings"].get("ee", {}).get("ppm", 0.0)
                print(
                    f"  step {session.step_count:4d} | state {rec['state']:12s} | "
                    f"ee_ppm {ee_ppm:8.3f} | pos ({rec['base_pose'][0]:6.2f}, {rec['base_pose'][1]:6.2f})"
                )

            if done:
                break

        # Capture the final state even if it does not fall on a stride boundary.
        if not args.no_video and session.history:
            recorder.capture(session.history)

        elapsed = time.time() - t_start
        summary = session.get_summary()

        # 独立评估
        from src.evaluation import SourceSearchEvaluator

        evaluator = SourceSearchEvaluator(success_distance_threshold=args.success_threshold)
        eval_result = evaluator.evaluate_session(session)

        print("-" * 70)
        print("Episode finished.")
        print(f"  Success       : {summary['success']}")
        print(f"  Total steps   : {summary['steps']}")
        print(f"  Final state   : {summary['final_state']}")
        print(f"  Final distance: {summary['final_distance_to_source']:.3f} m")
        print(f"  Max EE ppm    : {summary['max_ee_ppm']:.3f}")
        print(f"  Collision cnt : {summary['collision_count']}")
        print(f"  Wall time     : {elapsed:.1f} s")
        print(f"  GADEN queries : {env.get_query_stats()['query_count']}")
        print("-" * 70)
        print("Independent evaluation:")
        print(f"  Declared source : {eval_result['declared_source_pos']}")
        print(f"  Actual source   : {eval_result['actual_source_pos']}")
        print(f"  Error distance  : {eval_result['error_distance_m']:.3f} m")
        print(f"  Threshold       : {eval_result['success_distance_threshold_m']:.3f} m")
        print(f"  Evaluated success: {eval_result['evaluated_success']}")

        # ------------------------------------------------------------------ #
        # 5. Save artifacts
        # ------------------------------------------------------------------ #
        print("[5/5] Saving artifacts...")
        seed_tag = f"seed{args.seed}"
        history_path = out_dir / f"history_{seed_tag}.npz"
        summary_path = out_dir / f"summary_{seed_tag}.png"
        video_path = out_dir / f"video_{seed_tag}.mp4"

        session.export_history(history_path)
        print(f"      history -> {history_path}")

        plotter = TrajectoryPlotter(session.history, session.env)
        plotter.plot_summary(save_path=summary_path)
        print(f"      summary -> {summary_path}")

        if not args.no_video:
            recorder.save(video_path)

        print("=" * 70)
        return 0 if eval_result["evaluated_success"] else 1

    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception:
                pass
        if server is not None:
            try:
                server.stop()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
