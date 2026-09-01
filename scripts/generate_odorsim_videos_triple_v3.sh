#!/usr/bin/env bash
# Generate triple-view OdorSim/GADEN videos for a list of seeds with obstacle 8x8 config (improved algorithm v3).
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/generate_odorsim_videos_triple_v3.sh [seed1 seed2 ...]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_CONFIG="config/warehouse_obstacle_8x8.yaml"
OUT_DIR="$PROJECT_ROOT/outputs/odorsim_obstacle_new30_v3_videos"
SCENARIO="$PROJECT_ROOT/odorsim_scenarios/warehouse_8x8/environment_configurations/config1"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

# Default seeds will be overridden by command-line arguments; if none provided,
# the script expects the user to supply seeds (e.g., from pick_random_seeds_for_video.py).
SEEDS="${@:-}"
if [ -z "$SEEDS" ]; then
    echo "Usage: bash scripts/generate_odorsim_videos_triple_v3.sh seed1 seed2 seed3 seed4 seed5"
    exit 1
fi

exit_code=0
for seed in $SEEDS; do
    echo "================================ triple video seed $seed ================================"
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --env-config "$ENV_CONFIG" \
        --scenario "$SCENARIO" \
        --out-dir "$OUT_DIR" \
        --video-stride 2 \
        --fps 20 \
        --video-layout triple; then
        echo "[WARN] seed $seed failed; continuing with remaining seeds."
        exit_code=1
    fi
done

exit $exit_code
