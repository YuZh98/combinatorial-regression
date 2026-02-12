#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs

# -----------------------
# Defaults (override via env vars)
# -----------------------
export JASA_RUN_TAG="${JASA_RUN_TAG:-probit_default}"
export JASA_SEED="${JASA_SEED:-1234}"

# Simulation size (Section 5.1 defaults)
export JASA_N="${JASA_N:-100}"
export JASA_P="${JASA_P:-2}"
export JASA_D="${JASA_D:-2}"

# Sampler controls (used by both unconstrained + constrained scripts)
export JASA_PROBIT_N_ITER="${JASA_PROBIT_N_ITER:-20000}"
export JASA_PROBIT_BURN_IN="${JASA_PROBIT_BURN_IN:-5000}"
export JASA_PROBIT_THIN="${JASA_PROBIT_THIN:-10}"

# Constrained sampler-specific
export JASA_PROBIT_N_HAR="${JASA_PROBIT_N_HAR:-100}"
export JASA_PROBIT_LOG_EVERY="${JASA_PROBIT_LOG_EVERY:-1000}"

# Optional plotting (Simulation_probit.R reads JASA_PLOT)
export JASA_PLOT="${JASA_PLOT:-false}"

echo "[PROBIT] RUN_TAG=$JASA_RUN_TAG  SEED=$JASA_SEED"
echo "[PROBIT] n=$JASA_N p=$JASA_P d=$JASA_D"
echo "[PROBIT] N_ITER=$JASA_PROBIT_N_ITER  BURN_IN=$JASA_PROBIT_BURN_IN  THIN=$JASA_PROBIT_THIN"
echo "[PROBIT] N_HAR=$JASA_PROBIT_N_HAR  LOG_EVERY=$JASA_PROBIT_LOG_EVERY"
echo "[PROBIT] PLOT=$JASA_PLOT"
echo

Rscript R/simulations/fitted_value_curve/Simulation_probit.R | tee "results/logs/probit_${JASA_RUN_TAG}.log"

echo
echo "[PROBIT] Done."
echo "Outputs: results/runs/probit/${JASA_RUN_TAG}/"
