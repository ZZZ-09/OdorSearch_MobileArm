"""Aggregate independent evaluation across multiple OdorSim/GADEN runs.

Usage:
    python scripts/evaluate_odorsim_runs.py --out-dir outputs/odorsim --seeds 1 2 3 4 5 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import SourceSearchEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation for OdorSim/GADEN search runs."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "odorsim"),
        help="Directory containing history_seed<N>.npz files.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 42],
        help="Seeds to evaluate.",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=1.0,
        help="Distance threshold for success [m].",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional path to save aggregate JSON summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    evaluator = SourceSearchEvaluator(success_distance_threshold=args.success_threshold)

    rows = []
    for seed in args.seeds:
        path = out_dir / f"history_seed{seed}.npz"
        if not path.exists():
            print(f"[WARN] {path} not found, skipping seed {seed}.")
            continue
        hist = np.load(path, allow_pickle=True)
        source_pos = hist["source_pos"]
        ee_positions = hist["ee_positions"]
        final_ee = ee_positions[-1]
        states = hist["states"]
        finished = bool(np.any(states == "FINISHED"))
        # Attempt to recover declared position from history if present
        declared = None
        try:
            # npz saved by export_history does not include declared_source_position;
            # fall back to final_ee when FINISHED.
            declared = final_ee if finished else None
        except Exception:
            pass
        result = evaluator.evaluate(source_pos, declared, final_ee, finished)
        rows.append({"seed": seed, **result})

    if not rows:
        print("No runs found to evaluate.")
        return 1

    successes = sum(1 for r in rows if r["evaluated_success"])
    total = len(rows)
    errors = [r["error_distance_m"] for r in rows]
    steps = []
    for r in rows:
        path = out_dir / f"history_seed{r['seed']}.npz"
        hist = np.load(path, allow_pickle=True)
        steps.append(len(hist["steps"]))

    print("=" * 70)
    print("OdorSim/GADEN Aggregate Evaluation")
    print("=" * 70)
    print(f"Success threshold: {args.success_threshold} m")
    print(f"Evaluated runs   : {total}")
    print(f"Success count    : {successes}/{total} ({100.0 * successes / total:.1f}%)")
    print(f"Mean error       : {float(np.mean(errors)):.3f} m")
    print(f"Median error     : {float(np.median(errors)):.3f} m")
    print(f"Max error        : {float(np.max(errors)):.3f} m")
    print(f"Mean steps       : {float(np.mean(steps)):.1f}")
    print("-" * 70)
    for r in rows:
        print(
            f"seed {r['seed']:2d} | success {str(r['evaluated_success']):5s} | "
            f"error {r['error_distance_m']:.3f} m | steps {steps[rows.index(r)]:4d}"
        )
    print("=" * 70)

    if args.save_json:
        summary = {
            "success_threshold_m": args.success_threshold,
            "total_runs": total,
            "success_count": successes,
            "success_rate": successes / total,
            "mean_error_m": float(np.mean(errors)),
            "median_error_m": float(np.median(errors)),
            "max_error_m": float(np.max(errors)),
            "mean_steps": float(np.mean(steps)),
            "runs": [
                {
                    "seed": r["seed"],
                    "success": bool(r["evaluated_success"]),
                    "error_distance_m": float(r["error_distance_m"]),
                    "steps": steps[i],
                    "actual_source_pos": r["actual_source_pos"].tolist(),
                    "evaluated_position": r["evaluated_position"].tolist(),
                }
                for i, r in enumerate(rows)
            ],
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Aggregate summary saved to {out_path}")

    return 0 if successes == total else 1


if __name__ == "__main__":
    sys.exit(main())
