"""批量运行多个种子并保存每轮的运行图。

用法：
    python scripts/run_seed_benchmark.py --seeds 1 2 3 4 5 6 7 8 9 10 --max-steps 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import OdorSearchSession
from src.visualization import TrajectoryPlotter, WarehouseVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple seeds.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="List of seeds to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/seed_runs",
        help="Directory to save per-seed artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}...")
        print(f"{'='*60}")

        session = OdorSearchSession(seed=seed)
        obs = session.reset()
        print(f"Source: {session.env.source_pos}")
        print(f"Start:  {obs['base_pose'][:3]}")

        summary = session.run(max_steps=args.max_steps)

        # Save 2D summary plot
        plotter = TrajectoryPlotter(session.history, session.env)
        summary_path = output_dir / f"summary_seed{seed}.png"
        plotter.plot_summary(save_path=summary_path)

        # Save 3D warehouse plot
        vis = WarehouseVisualizer(session.env, session.robot)
        vis.draw_static_environment()
        trajectory = np.array([rec["ee_pos"] for rec in session.history])
        vis.draw_robot(trajectory)
        last_vision = session.history[-1].get("vision", {}) if session.history else {}
        vis.draw_vision(last_vision)
        vis_3d_path = output_dir / f"warehouse_3d_seed{seed}.png"
        vis.save(vis_3d_path)

        # Save history
        history_path = output_dir / f"history_seed{seed}.npz"
        session.export_history(history_path)

        record = {
            "seed": seed,
            "source_pos": session.env.source_pos.tolist(),
            "success": summary["success"],
            "steps": summary["steps"],
            "final_distance": summary["final_distance_to_source"],
            "max_ee_ppm": summary["max_ee_ppm"],
            "collision_count": summary["collision_count"],
            "state_counts": summary["state_counts"],
            "summary_plot": str(summary_path),
            "warehouse_3d_plot": str(vis_3d_path),
            "history": str(history_path),
        }
        results.append(record)

        print(f"Success: {summary['success']}, Steps: {summary['steps']}, "
              f"Final dist: {summary['final_distance_to_source']:.3f} m, "
              f"Collisions: {summary['collision_count']}")

    # Save JSON summary
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print aggregate stats
    successes = sum(1 for r in results if r["success"])
    total = len(results)
    collision_free = sum(1 for r in results if r["collision_count"] == 0)
    avg_steps = np.mean([r["steps"] for r in results if r["success"]]) if successes > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Benchmark complete: {successes}/{total} succeeded "
          f"(success rate: {100.0*successes/total:.1f}%)")
    print(f"Collision-free episodes: {collision_free}/{total}")
    print(f"Average steps (success only): {avg_steps:.1f}")
    print(f"Results saved to {json_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
