#!/usr/bin/env bash
# Generate OdorSim/GADEN videos for a list of seeds with a specified env config.
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/generate_odorsim_videos_env.sh <env_config> <out_dir> [seed1 seed2 ...]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_CONFIG="${1:-config/warehouse_empty_8x8.yaml}"
OUT_DIR="${2:-outputs/odorsim}"
SEEDS="${@:3}"
SEEDS="${SEEDS:-1 2 3 4 5 42}"

mkdir -p "$PROJECT_ROOT/$OUT_DIR"
cd "$PROJECT_ROOT"
exit_code=0
for seed in $SEEDS; do
    echo "================================ video $ENV_CONFIG seed $seed ================================"
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --env-config "$ENV_CONFIG" \
        --scenario "$PROJECT_ROOT/odorsim_scenarios/warehouse_8x8/environment_configurations/config1" \
        --out-dir "$PROJECT_ROOT/$OUT_DIR" \
        --video-stride 2 \
        --fps 20; then
        echo "[WARN] seed $seed failed; continuing with remaining seeds."
        exit_code=1
    fi
done

exit $exit_code
