# Reproducibility Guide (Paper ↔ Code)

This document maps paper sections to scripts in this repository and explains how to run them.

## Repository conventions
- Run commands from the **repository root**.
- Generated outputs go to `results/` (gitignored).

## Main paper

### Proposed method: MH-within-Gibbs (R)
Entry script:
- `R/simulations/mh_within_gibbs/Production_Run.R`

This script supports environment-variable overrides, e.g.
- `JASA_D_LIST`, `JASA_M_LIST`, `JASA_N_REP`
- `JASA_N_ITER`, `JASA_N_WARMUP`, `JASA_N_HAR`
- `JASA_METHODS`
- `JASA_RUN_TAG` (controls output folder under `results/runs/mh_within_gibbs/`)

Example smoke run:
```bash
make smoke
Data analysis on waterfowl matching (R)
Scripts:

R/data_analysis/duck_matching.R

R/data_analysis/duck_matching_reduced.R

R/data_analysis/BF.R

Notes:

These scripts may require minor path/output unification (writing into results/), similar to Production_Run.R.

Supplementary materials (Python)
Pseudo-Marginal MCMC (PMMH)
python/pmmh/pmmh_core.py

Hamiltonian Monte Carlo (HMC)
python/hmc/nuts_ilp_model.py

python/hmc/NUTS100_Experiments.ipynb

python/hmc/API.md

Additional results / plotting
Notebooks currently in python/plots/:

python/plots/Integer_Programming.ipynb

Plotted figures are intended to be written under results/figures/ once plotting scripts are unified.

