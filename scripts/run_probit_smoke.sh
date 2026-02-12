#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Small, fast settings just to check everything runs
JASA_RUN_TAG="${JASA_RUN_TAG:-probit_smoke}" \
JASA_SEED="${JASA_SEED:-1}" \
JASA_N="${JASA_N:-100}" \
JASA_P="${JASA_P:-2}" \
JASA_D="${JASA_D:-2}" \
JASA_PROBIT_N_ITER="${JASA_PROBIT_N_ITER:-10000}" \
JASA_PROBIT_BURN_IN="${JASA_PROBIT_BURN_IN:-1000}" \
JASA_PROBIT_THIN="${JASA_PROBIT_THIN:-5}" \
JASA_PROBIT_N_HAR="${JASA_PROBIT_N_HAR:-25}" \
JASA_PROBIT_LOG_EVERY="${JASA_PROBIT_LOG_EVERY:-1000}" \
JASA_PLOT="${JASA_PLOT:-true}" \
bash scripts/run_probit.sh
