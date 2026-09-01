"""Quickly inspect a saved history npz for debugging."""
from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, required=True)
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()

    hist = np.load(args.history, allow_pickle=True)
    print("source:", hist["source_pos"])
    print("steps:", len(hist["steps"]))
    print("states:", dict(zip(*np.unique(hist["states"], return_counts=True))))
    print("final base:", hist["base_poses"][-1])
    print("final ee:", hist["ee_positions"][-1])
    print("final distance:", np.linalg.norm(hist["ee_positions"][-1] - hist["source_pos"]))
    print("-" * 70)
    for i in range(max(0, len(hist["steps"]) - args.tail), len(hist["steps"])):
        print(
            f"step {hist['steps'][i]:4d}: base=({hist['base_poses'][i,0]:6.2f}, {hist['base_poses'][i,1]:6.2f}, {hist['base_poses'][i,2]:5.2f}), "
            f"ee=({hist['ee_positions'][i,0]:6.2f}, {hist['ee_positions'][i,1]:6.2f}, {hist['ee_positions'][i,2]:5.2f}), "
            f"state={hist['states'][i]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
