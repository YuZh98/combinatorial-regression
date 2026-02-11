#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/{runs,figures,tables,logs}

echo "[FULL] Running MH-within-Gibbs (example settings; adjust as needed)..."

export JASA_RUN_TAG=full_example
export JASA_D_LIST="2 5 10"
export JASA_M_LIST="1 2 5"
export JASA_N_REP=2
export JASA_N_ITER=5000
export JASA_N_WARMUP=1000
export JASA_N_THIN=25
export JASA_N_HAR=50
export JASA_METHODS="exponential half_gaussian"
export JASA_SAVE=false
export JASA_PLOT=false

Rscript R/simulations/mh_within_gibbs/Production_Run.R | tee results/logs/mh_within_gibbs_full_example.log

echo "[FULL] Done."
echo "Outputs: results/runs/mh_within_gibbs/full_example/"
echo
echo "To run the full paper grid, set env vars JASA_D_LIST, JASA_M_LIST, JASA_N_REP, JASA_N_ITER, JASA_N_WARMUP, etc."
