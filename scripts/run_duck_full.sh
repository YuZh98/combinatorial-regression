#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs

# Defaults (override by exporting env vars before running)
FULL_TAG="${JASA_RUN_TAG:-duck_full}"
N_ITER="${JASA_N_ITER:-50000}"
N_WARMUP="${JASA_N_WARMUP:-5000}"
KAPPA="${JASA_KAPPA:-5}"

SAVE_ZETA="${JASA_SAVE_ZETA:-false}"
SAVE_RHO="${JASA_SAVE_RHO:-true}"
SAVE_A="${JASA_SAVE_A:-true}"

echo "[DUCK FULL] RUN_TAG=$FULL_TAG  N_ITER=$N_ITER  N_WARMUP=$N_WARMUP  KAPPA=$KAPPA"
echo "[DUCK FULL] SAVE_ZETA=$SAVE_ZETA  SAVE_RHO=$SAVE_RHO  SAVE_A=$SAVE_A"
echo

JASA_RUN_TAG="$FULL_TAG" \
JASA_N_ITER="$N_ITER" \
JASA_N_WARMUP="$N_WARMUP" \
JASA_KAPPA="$KAPPA" \
JASA_SAVE_ZETA="$SAVE_ZETA" \
JASA_SAVE_RHO="$SAVE_RHO" \
JASA_SAVE_A="$SAVE_A" \
Rscript R/data_analysis/duck_matching.R | tee "results/logs/duck_full_${FULL_TAG}.log"

echo
echo "[DUCK FULL] Done."
echo "Outputs: results/runs/data_analysis/${FULL_TAG}/"
