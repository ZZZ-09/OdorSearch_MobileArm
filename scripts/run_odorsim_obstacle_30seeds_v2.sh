#!/usr/bin/env bash
# Run OdorSim/GADEN verification for 30 random seeds with obstacle 8x8 config (improved algorithm v2).
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/run_odorsim_obstacle_30seeds_v2.sh
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Same 30 seeds as 2026-09-01 baseline for direct comparison.
SEEDS="63219 800 14849 38588 28488 8252 47760 49790 91814 44874 68324 70542 65144 29778 12420 62126 50541 56090 11025 8984 73404 894 54403 38228 72330 89695 70600 2892 85069 96149"
ENV_CONFIG="config/warehouse_obstacle_8x8.yaml"
OUT_DIR="$PROJECT_ROOT/outputs/odorsim_obstacle_new30_v2"
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
        --fps 20 \
        --no-video; then
        echo "[WARN] seed $seed failed"
    fi
done

echo "==================== Aggregate evaluation ===================="
python scripts/evaluate_odorsim_runs.py \
    --out-dir "$OUT_DIR" \
    --seeds $SEEDS \
    --save-json "$OUT_DIR/aggregate_evaluation.json"

echo "==================== Detailed analysis ===================="
python scripts/analyze_odorsim_runs.py \
    --out-dir "$OUT_DIR" \
    --seeds $SEEDS \
    --output "$OUT_DIR/analysis.md"
