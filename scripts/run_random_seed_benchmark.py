"""在解析环境中用随机种子批量测试搜索算法成功率。

用法：
    python scripts/run_random_seed_benchmark.py --num-runs 30 --max-steps 2000 --output-dir outputs/random_seed_runs
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
from src.visualization import TrajectoryPlotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-seed benchmark in analytical environment.")
    parser.add_argument("--num-runs", type=int, default=30, help="Number of random seeds to test.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/random_seed_runs",
        help="Directory to save per-seed artifacts.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=2026,
        help="Offset for the random seed generator.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for independent evaluation success [m].",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help="Path to warehouse env config (default: config/warehouse.yaml).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed_offset)
    seeds = [int(s) for s in rng.integers(100, 100000, size=args.num_runs)]

    print(f"Testing {args.num_runs} random seeds: {seeds}")

    results = []
    for idx, seed in enumerate(seeds, start=1):
        print(f"\n{'='*60}")
        print(f"Run {idx}/{args.num_runs} | seed {seed}")
        print(f"{'='*60}")

        session = OdorSearchSession(seed=seed, env_config=args.env_config)
        obs = session.reset()
        print(f"Source: {session.env.source_pos}")
        print(f"Start:  {obs['base_pose'][:3]}")

        summary = session.run(max_steps=args.max_steps)

        # Independent evaluation
        from src.evaluation import SourceSearchEvaluator

        evaluator = SourceSearchEvaluator(success_distance_threshold=args.success_threshold)
        eval_result = evaluator.evaluate_session(session)

        # Save summary plot
        plotter = TrajectoryPlotter(session.history, session.env)
        summary_path = output_dir / f"summary_seed{seed}.png"
        plotter.plot_summary(save_path=summary_path)

        # Save history
        history_path = output_dir / f"history_seed{seed}.npz"
        session.export_history(history_path)

        record = {
            "run": idx,
            "seed": seed,
            "source_pos": session.env.source_pos.tolist(),
            "success": summary["success"],
            "evaluated_success": eval_result["evaluated_success"],
            "steps": summary["steps"],
            "final_distance": summary["final_distance_to_source"],
            "error_distance_m": eval_result["error_distance_m"],
            "max_ee_ppm": summary["max_ee_ppm"],
            "collision_count": summary["collision_count"],
            "state_counts": summary["state_counts"],
            "summary_plot": str(summary_path),
            "history": str(history_path),
        }
        results.append(record)

        status = "SUCCESS" if eval_result["evaluated_success"] else "FAIL"
        print(
            f"[{status}] steps={summary['steps']}, final_dist={summary['final_distance_to_source']:.3f}m, "
            f"error={eval_result['error_distance_m']:.3f}m, max_ppm={summary['max_ee_ppm']:.2f}"
        )

    # Save JSON summary
    json_path = output_dir / "random_seed_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Aggregate stats
    successes = sum(1 for r in results if r["evaluated_success"])
    total = len(results)
    collision_free = sum(1 for r in results if r["collision_count"] == 0)
    success_steps = [r["steps"] for r in results if r["evaluated_success"]]
    avg_steps = float(np.mean(success_steps)) if success_steps else 0.0
    errors = [r["error_distance_m"] for r in results if r["evaluated_success"]]
    mean_error = float(np.mean(errors)) if errors else 0.0
    max_error = float(np.max(errors)) if errors else 0.0

    print(f"\n{'='*60}")
    print(f"Random-seed benchmark complete: {successes}/{total} succeeded ({100.0*successes/total:.1f}%)")
    print(f"Collision-free episodes: {collision_free}/{total}")
    print(f"Average steps (success only): {avg_steps:.1f}")
    print(f"Mean error (success only): {mean_error:.3f} m")
    print(f"Max error (success only): {max_error:.3f} m")
    print(f"Results saved to {json_path}")
    print(f"{'='*60}")

    # Print failed seeds for analysis
    failed = [r for r in results if not r["evaluated_success"]]
    if failed:
        print("\nFailed seeds:")
        for r in failed:
            print(
                f"  seed={r['seed']}, source={r['source_pos']}, "
                f"final_dist={r['final_distance']:.3f}m, max_ppm={r['max_ee_ppm']:.2f}, "
                f"states={r['state_counts']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
