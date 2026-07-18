#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/{runs,figures,tables,logs}

echo "[SMOKE] Running MH-within-Gibbs (fast settings)..."

export JASA_RUN_TAG="${JASA_RUN_TAG:-smoke}"
export JASA_D_LIST="${JASA_D_LIST:-2}"
export JASA_M_LIST="${JASA_M_LIST:-1}"
export JASA_N_REP="${JASA_N_REP:-1}"
export JASA_N_ITER="${JASA_N_ITER:-200}"
export JASA_N_WARMUP="${JASA_N_WARMUP:-50}"
export JASA_N_THIN="${JASA_N_THIN:-10}"
export JASA_N_HAR="${JASA_N_HAR:-10}"
export JASA_METHODS="${JASA_METHODS:-exponential}"
export JASA_SAVE="${JASA_SAVE:-false}"
export JASA_PLOT="${JASA_PLOT:-true}"

Rscript R/simulations/mh_within_gibbs/Production_Run.R | tee results/logs/mh_within_gibbs_smoke.log

echo "[SMOKE] Done."
echo "Outputs: results/runs/mh_within_gibbs/smoke/"


echo "[SMOKE] Running duck data analysis (fast settings)..."

export JASA_RUN_TAG="${JASA_RUN_TAG:-duck_smoke}"
export JASA_N_ITER="${JASA_N_ITER:-200}"
export JASA_N_WARMUP="${JASA_N_WARMUP:-50}"
export JASA_SAVE_ZETA="${JASA_SAVE_ZETA:-false}"
export JASA_SAVE_RHO="${JASA_SAVE_RHO:-true}"
export JASA_SAVE_A="${JASA_SAVE_A:-true}"

Rscript R/data_analysis/duck_matching.R | tee results/logs/duck_smoke.log

echo "[SMOKE] Duck analysis done."
