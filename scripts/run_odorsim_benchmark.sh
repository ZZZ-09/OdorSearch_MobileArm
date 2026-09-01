#!/usr/bin/env bash
# Run OdorSim/GADEN verification across multiple seeds.
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/run_odorsim_benchmark.sh [seed1 seed2 ...]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS="${@:-1 2 3 4 5}"

cd "$PROJECT_ROOT"
exit_code=0
for seed in $SEEDS; do
    echo "================================ seed $seed ================================"
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --video-stride 50 \
        --no-video; then
        echo "[WARN] seed $seed failed; continuing with remaining seeds."
        exit_code=1
    fi
done

exit $exit_code
