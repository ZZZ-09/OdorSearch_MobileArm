#!/usr/bin/env bash
# Generate OdorSim/GADEN verification videos for a list of seeds.
# Usage (from WSL, after sourcing OdorSim/setup/activate.sh):
#   bash scripts/generate_odorsim_videos.sh [seed1 seed2 ...]
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEEDS="${@:-1 2 3 4 5 42}"

cd "$PROJECT_ROOT"
exit_code=0
for seed in $SEEDS; do
    echo "================================ video seed $seed ================================"
    if ! python scripts/run_odorsim_verification.py \
        --seed "$seed" \
        --max-steps 2000 \
        --video-stride 50 \
        --fps 10 \
        --success-threshold 1.0; then
        echo "[WARN] seed $seed failed; continuing with remaining seeds."
        exit_code=1
    fi
done

exit $exit_code
