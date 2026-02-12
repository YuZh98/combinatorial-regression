#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs

REDUCED_TAG="${JASA_RUN_TAG:-duck_reduced}"
N_ITER="${JASA_N_ITER:-50000}"
N_WARMUP="${JASA_N_WARMUP:-5000}"
KAPPA="${JASA_KAPPA:-5}"

SAVE_ZETA="${JASA_SAVE_ZETA:-false}"
SAVE_RHO="${JASA_SAVE_RHO:-true}"
SAVE_A="${JASA_SAVE_A:-true}"

# Optional: if your reduced script tries to load full samples for comparison
# you can pass FULL_TAG via JASA_FULL_TAG (it is harmless if not used).
FULL_TAG="${JASA_FULL_TAG:-default}"

echo "[DUCK REDUCED] RUN_TAG=$REDUCED_TAG  N_ITER=$N_ITER  N_WARMUP=$N_WARMUP  KAPPA=$KAPPA"
echo "[DUCK REDUCED] SAVE_ZETA=$SAVE_ZETA  SAVE_RHO=$SAVE_RHO  SAVE_A=$SAVE_A"
echo "[DUCK REDUCED] JASA_FULL_TAG=$FULL_TAG"
echo

JASA_RUN_TAG="$REDUCED_TAG" \
JASA_FULL_TAG="$FULL_TAG" \
JASA_N_ITER="$N_ITER" \
JASA_N_WARMUP="$N_WARMUP" \
JASA_KAPPA="$KAPPA" \
JASA_SAVE_ZETA="$SAVE_ZETA" \
JASA_SAVE_RHO="$SAVE_RHO" \
JASA_SAVE_A="$SAVE_A" \
Rscript R/data_analysis/duck_matching_reduced.R | tee "results/logs/duck_reduced_${REDUCED_TAG}.log"

echo
echo "[DUCK REDUCED] Done."
echo "Outputs: results/runs/data_analysis/${REDUCED_TAG}/"
