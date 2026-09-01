"""从随机种子批量测试结果中抽取指定数量的种子，用于 OdorSim 视频生成。

用法：
    python scripts/pick_random_seeds_for_video.py --num-pick 5 --output seeds_for_video.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick random seeds for OdorSim video generation.")
    parser.add_argument(
        "--results",
        type=str,
        default="outputs/random_seed_runs/random_seed_benchmark_results.json",
        help="Path to random seed benchmark results JSON.",
    )
    parser.add_argument("--num-pick", type=int, default=5, help="Number of seeds to pick.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/random_seed_runs/seeds_for_video.txt",
        help="Output text file with selected seeds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Picker random seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return 1

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Support both random_seed_benchmark_results.json and
    # evaluate_odorsim_runs.py aggregate_evaluation.json formats.
    if isinstance(results, dict) and "runs" in results:
        runs = results["runs"]
    elif isinstance(results, list):
        runs = results
    else:
        print(f"[ERROR] Unrecognized results format in: {results_path}")
        return 1

    all_seeds = [r["seed"] for r in runs]
    rng = np.random.default_rng(args.seed)
    selected = sorted(rng.choice(all_seeds, size=min(args.num_pick, len(all_seeds)), replace=False).tolist())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, selected)) + "\n")

    print(f"Selected {len(selected)} seeds for OdorSim video generation: {selected}")
    print(f"Saved to {out_path}")

    # Also print details
    for r in runs:
        if r["seed"] in selected:
            status = "SUCCESS" if r["success"] else "FAIL"
            src = r.get("actual_source_pos", r.get("source_pos", "N/A"))
            print(
                f"  seed={r['seed']} ({status}) source={src}, "
                f"error={r['error_distance_m']:.3f}m, steps={r['steps']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
