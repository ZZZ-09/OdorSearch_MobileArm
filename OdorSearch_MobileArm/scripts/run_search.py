"""气味源搜索仿真入口脚本。

用法示例：
    python scripts/run_search.py --max-steps 3000 --seed 42 --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import OdorSearchSession
from src.visualization import TrajectoryPlotter, WarehouseVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 3D odor source search simulation with mobile robot + robotic arm."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum simulation steps per episode (default: 3000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show 3D visualization after the simulation finishes.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save summary plot (e.g., outputs/summary.png).",
    )
    parser.add_argument(
        "--save-history",
        type=str,
        default=None,
        help="Path to save history npz (e.g., outputs/history.npz).",
    )
    parser.add_argument(
        "--start-x",
        type=float,
        default=-7.0,
        help="Initial robot base X position (default: -7.0).",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=-6.5,
        help="Initial robot base Y position (default: -6.5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("3D Odor Source Search Simulation")
    print("Robot: AgileX UMR + RealMan RM65-6F-V")
    print("=" * 60)

    session = OdorSearchSession(seed=args.seed)

    base_pose = np.array([args.start_x, args.start_y, 0.0, 0.0], dtype=float)
    obs = session.reset(base_pose=base_pose)

    print(f"Source position: {session.env.source_pos}")
    print(f"Initial base pose: {obs['base_pose']}")
    print(f"Initial EE position: {obs['ee_pos']}")
    print("-" * 60)

    summary = session.run(max_steps=args.max_steps)

    print("\nSimulation finished.")
    print(f"Success: {summary['success']}")
    print(f"Total steps: {summary['steps']}")
    print(f"Final state: {summary['final_state']}")
    print(f"Final distance to source: {summary['final_distance_to_source']:.3f} m")
    print(f"Max EE ppm: {summary['max_ee_ppm']:.2f}")
    print(f"State counts: {summary['state_counts']}")
    print(f"Collision count: {summary['collision_count']}")

    # 保存历史
    if args.save_history:
        session.export_history(args.save_history)
        print(f"History saved to {args.save_history}")

    # 可视化
    if args.visualize:
        print("\nGenerating 3D visualization...")
        vis = WarehouseVisualizer(session.env, session.robot)
        vis.draw_static_environment()
        trajectory = np.array([rec["ee_pos"] for rec in session.history])
        vis.draw_robot(trajectory)
        # 绘制最终视觉检测射线
        last_vision = session.history[-1].get("vision", {}) if session.history else {}
        vis.draw_vision(last_vision)
        vis.show()

    # 绘制摘要图
    if args.save_plot or args.visualize:
        plotter = TrajectoryPlotter(session.history, session.env)
        save_path = args.save_plot or PROJECT_ROOT / "outputs" / "summary.png"
        plotter.plot_summary(save_path=save_path)
        print(f"Summary plot saved to {save_path}")

    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
