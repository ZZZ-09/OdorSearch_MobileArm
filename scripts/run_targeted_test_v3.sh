#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_CONFIG="config/warehouse_obstacle_8x8.yaml"
OUT_DIR="$PROJECT_ROOT/outputs/odorsim_obstacle_test_v3"
SCENARIO="$PROJECT_ROOT/odorsim_scenarios/warehouse_8x8/environment_configurations/config1"

mkdir -p "$OUT_DIR"
cd "$PROJECT_ROOT"

for seed in 44874 91814 28488 800; do
    echo "=== seed $seed ==="
    python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --env-config "$ENV_CONFIG" \
        --scenario "$SCENARIO" \
        --out-dir "$OUT_DIR" \
        --video-stride 2 \
        --fps 20 \
        --no-video || true
done
