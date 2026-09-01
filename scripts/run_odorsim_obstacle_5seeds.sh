#!/usr/bin/env bash
# Run OdorSim/GADEN verification for 5 random seeds with obstacle 8x8 config.
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/run_odorsim_obstacle_5seeds.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS="42 123 456 789 101112"
ENV_CONFIG="config/warehouse_obstacle_8x8.yaml"
OUT_DIR="$PROJECT_ROOT/outputs/odorsim_obstacle_5seeds"
SCENARIO="$PROJECT_ROOT/odorsim_scenarios/warehouse_8x8/environment_configurations/config1"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

for seed in $SEEDS; do
    echo "==================== seed $seed ===================="
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --env-config "$ENV_CONFIG" \
        --scenario "$SCENARIO" \
        --out-dir "$OUT_DIR" \
        --video-stride 2 \
        --fps 20; then
        echo "[WARN] seed $seed failed"
    fi
done

echo "==================== Aggregate evaluation ===================="
python scripts/evaluate_odorsim_runs.py \
    --out-dir "$OUT_DIR" \
    --seeds $SEEDS \
    --save-json "$OUT_DIR/aggregate_evaluation.json"
