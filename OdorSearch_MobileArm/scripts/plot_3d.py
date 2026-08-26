"""从保存的历史记录生成三维仓库静态图。

用法：
    python scripts/plot_3d.py --history outputs/final_demo.npz --output outputs/warehouse_3d.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import WarehouseEnv
from src.visualization import WarehouseVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 3D warehouse plot from history.")
    parser.add_argument(
        "--history",
        type=str,
        default="outputs/final_demo.npz",
        help="Path to saved history npz file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/warehouse_3d.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = WarehouseEnv()
    # dummy robot just for visualization structure
    from src.robot import MobileArmRobot
    robot = MobileArmRobot()

    hist = np.load(args.history, allow_pickle=True)
    base_poses = hist["base_poses"]
    ee_positions = hist["ee_positions"]

    # Set robot to final state
    robot.base_pose = base_poses[-1]
    joint_angles = hist["joint_angles"]
    robot.set_joint_angles(joint_angles[-1])

    vis = WarehouseVisualizer(env, robot)
    vis.draw_static_environment()
    vis.draw_robot(ee_positions)
    vis.save(args.output)
    print(f"3D plot saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
