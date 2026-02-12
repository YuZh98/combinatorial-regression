#!/usr/bin/env bash
set -euo pipefail

# Run from repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/logs

# -----------------------
# User-tunable settings
# -----------------------
N_ITER="${JASA_N_ITER:-50000}"
N_WARMUP="${JASA_N_WARMUP:-2000}"
KAPPA="${JASA_KAPPA:-5}"

# Tags (folders under results/runs/data_analysis/)
FULL_TAG="${JASA_FULL_TAG:-duck_full}"
REDUCED_TAG="${JASA_REDUCED_TAG:-duck_reduced}"
BF_TAG="${JASA_BF_TAG:-bf_1}"

# BF subsampling (matches BF.R env vars)
BF_START="${JASA_BF_START:-2000}"
BF_END="${JASA_BF_END:-$N_ITER}"
BF_BY="${JASA_BF_BY:-20}"

echo "[BF PIPELINE] Settings:"
echo "  FULL_TAG=$FULL_TAG"
echo "  REDUCED_TAG=$REDUCED_TAG"
echo "  BF_TAG=$BF_TAG"
echo "  N_ITER=$N_ITER  N_WARMUP=$N_WARMUP  KAPPA=$KAPPA"
echo "  BF_START=$BF_START  BF_END=$BF_END  BF_BY=$BF_BY"
echo

# -----------------------
# 1) Full model
# -----------------------
echo "[1/3] Running full duck model..."
JASA_RUN_TAG="$FULL_TAG" \
JASA_N_ITER="$N_ITER" \
JASA_N_WARMUP="$N_WARMUP" \
JASA_KAPPA="$KAPPA" \
JASA_SAVE_ZETA=true \
JASA_SAVE_RHO=true \
JASA_SAVE_A=true \
Rscript R/data_analysis/duck_matching.R | tee "results/logs/duck_full_${FULL_TAG}.log"

# -----------------------
# 2) Reduced model
# -----------------------
echo "[2/3] Running reduced duck model..."
JASA_RUN_TAG="$REDUCED_TAG" \
JASA_N_ITER="$N_ITER" \
JASA_N_WARMUP="$N_WARMUP" \
JASA_KAPPA="$KAPPA" \
JASA_SAVE_ZETA=true \
JASA_SAVE_RHO=true \
JASA_SAVE_A=true \
Rscript R/data_analysis/duck_matching_reduced.R | tee "results/logs/duck_reduced_${REDUCED_TAG}.log"

# -----------------------
# 3) Bayes Factor
# -----------------------
echo "[3/3] Computing Bayes factor..."
JASA_RUN_TAG="$BF_TAG" \
JASA_FULL_TAG="$FULL_TAG" \
JASA_REDUCED_TAG="$REDUCED_TAG" \
JASA_N_ITER="$N_ITER" \
JASA_KAPPA="$KAPPA" \
JASA_BF_START="$BF_START" \
JASA_BF_END="$BF_END" \
JASA_BF_BY="$BF_BY" \
Rscript R/data_analysis/BF.R | tee "results/logs/bf_${BF_TAG}.log"

echo
echo "[BF PIPELINE] Done."
echo "Full samples:    results/runs/data_analysis/${FULL_TAG}/"
echo "Reduced samples: results/runs/data_analysis/${REDUCED_TAG}/"
echo "BF outputs:      results/tables/data_analysis/bayes_factor_${BF_TAG}.csv"
