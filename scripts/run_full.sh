#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/{runs,figures,tables,logs}

echo "[FULL] Running MH-within-Gibbs (example settings; adjust as needed)..."

export JASA_RUN_TAG="${JASA_RUN_TAG:-full_example}"
export JASA_D_LIST="${JASA_D_LIST:-2 5 10 100 1000}"
export JASA_M_LIST="${JASA_M_LIST:-1 2 5 10 20 50 100}"
export JASA_N_REP="${JASA_N_REP:-1}"
export JASA_N_ITER="${JASA_N_ITER:-50000}"
export JASA_N_WARMUP="${JASA_N_WARMUP:-5000}"
export JASA_N_THIN="${JASA_N_THIN:-25}"
export JASA_N_HAR="${JASA_N_HAR:-100}"
export JASA_METHODS="${JASA_METHODS:-exponential halfgaussian}"
export JASA_SAVE="${JASA_SAVE:-true}"
export JASA_PLOT="${JASA_PLOT:-true}"

Rscript R/simulations/mh_within_gibbs/Production_Run.R | tee results/logs/mh_within_gibbs_full_example.log

echo "[FULL] Done."
echo "Outputs: results/runs/mh_within_gibbs/full_example/"
echo
echo "To run the full paper grid, set env vars JASA_D_LIST, JASA_M_LIST, JASA_N_REP, JASA_N_ITER, JASA_N_WARMUP, etc."
