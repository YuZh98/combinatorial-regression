#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/{runs,logs}

# -----------------------
# Kernel comparison defaults
# -----------------------
export JASA_RUN_TAG="${JASA_RUN_TAG:-kernel_compare}"
export JASA_METHODS="${JASA_METHODS:-exponential,halfgaussian}"

# (d, m) pairs to run
export JASA_D_LIST="${JASA_D_LIST:-5,20,50,200,1000}"
export JASA_M_LIST="${JASA_M_LIST:-1,5,10,50,100}"

# Lighter run for kernel comparison
export JASA_N_ITER="${JASA_N_ITER:-10000}"
export JASA_N_WARMUP="${JASA_N_WARMUP:-5000}"

# Other knobs (keep defaults unless overridden)
export JASA_N_THIN="${JASA_N_THIN:-25}"
export JASA_N_HAR="${JASA_N_HAR:-100}"
export JASA_SAVE="${JASA_SAVE:-false}"
export JASA_PLOT="${JASA_PLOT:-false}"

echo "[KERNEL COMPARE] RUN_TAG=$JASA_RUN_TAG"
echo "[KERNEL COMPARE] METHODS=$JASA_METHODS"
echo "[KERNEL COMPARE] D_LIST=$JASA_D_LIST"
echo "[KERNEL COMPARE] M_LIST=$JASA_M_LIST"
echo "[KERNEL COMPARE] N_ITER=$JASA_N_ITER  N_WARMUP=$JASA_N_WARMUP"
echo

Rscript R/simulations/mh_within_gibbs/Production_Run.R | tee "results/logs/mh_kernel_compare_${JASA_RUN_TAG}.log"

echo
echo "[KERNEL COMPARE] Done."
echo "Outputs: results/runs/mh_within_gibbs/${JASA_RUN_TAG}/"
