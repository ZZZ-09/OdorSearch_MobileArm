#!/usr/bin/env bash
# Run OdorSim/GADEN verification across multiple seeds with the obstacle 8x8 config.
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/run_odorsim_obstacle_benchmark.sh seed1 seed2 ...
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS="${@:-85200 17975 2739 64027 36611 46780 8074 37113 64380 35556 83123 79072 70472 90523 72148 17817 85859 65313 9936 29900 16650 96699 72823 91993 28321 63623 60457 75297 11889 51563}"
OUT_DIR="$PROJECT_ROOT/outputs/odorsim_obstacle_30seeds"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"
exit_code=0
for seed in $SEEDS; do
    echo "================================ OdorSim obstacle seed $seed ================================"
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --env-config config/warehouse_obstacle_8x8.yaml \
        --scenario "$PROJECT_ROOT/odorsim_scenarios/warehouse_8x8/environment_configurations/config1" \
        --out-dir "$OUT_DIR" \
        --no-video; then
        echo "[WARN] seed $seed failed; continuing with remaining seeds."
        exit_code=1
    fi
done

echo "================================ Aggregate evaluation ================================"
python scripts/evaluate_odorsim_runs.py \
    --out-dir "$OUT_DIR" \
    --seeds $SEEDS \
    --save-json "$OUT_DIR/aggregate_evaluation.json"

exit $exit_code
