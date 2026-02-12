# Reproducibility Guide (Paper ↔ Code)

This document explains how to reproduce the main results of the paper and how the repository is organized.

All commands must be run from the **repository root**.

All generated outputs are written under:

```
results/
```

This directory is gitignored.

---

# 1. Repository Conventions

- All scripts assume you run from the repository root.
- All results are written to structured subfolders inside `results/`.
- No manual path editing is required.
- All key settings are controlled via **environment variables** (no code editing needed).

---

# 2. Main Paper Results

## 2.1 Proposed Method: MH-within-Gibbs (Simulation Study)

Entry script:

```
R/simulations/mh_within_gibbs/Production_Run.R
```

### Default behavior (main paper)

- Kernel for auxiliary variable \( u \): **exponential**
- Full simulation grid over:
  - `d_list = 2,5,10,20,50,100,200,500,1000`
  - `m_list = 1,2,5,10,20,50,100`
- Default iterations:
  - `n_iter = 50000`
  - `n_warmup = 5000`

### Run full simulation

```bash
make full
```

Outputs are written to:

```
results/runs/mh_within_gibbs/<RUN_TAG>/
```

(Default `RUN_TAG = default` unless overridden.)

---

## 2.2 Kernel Comparison (Supplementary)

The supplementary material compares two kernels for updating the auxiliary variable \( u \):

- `exponential`
- `half_gaussian`

To run the comparison:

```bash
make kernel_compare
```

This will:

- Use both kernels
- Run across all `(d, m)` combinations (with `m < d`)
- Use lighter defaults:
  - `n_iter = 10000`
  - `n_warmup = 5000`

Outputs are written under:

```
results/runs/mh_within_gibbs/kernel_compare/
```

---

## 2.3 Quick Smoke Test

To verify the entire pipeline works:

```bash
make smoke
```

This runs:
- A lightweight simulation
- A lightweight duck data analysis

Smoke test runs quickly and is intended for sanity checking.

---

# 3. Data Analysis: Waterfowl Matching

Scripts:

```
R/data_analysis/duck_matching.R
R/data_analysis/duck_matching_reduced.R
R/data_analysis/BF.R
```

---

## 3.1 Full Model

```bash
make duck_full
```

Outputs:
```
results/runs/data_analysis/duck_full/
```

---

## 3.2 Reduced Model

```bash
make duck_reduced
```

Outputs:
```
results/runs/data_analysis/duck_reduced/
```

---

## 3.3 Bayes Factor Comparison

To compute the Bayes factor between full and reduced models:

```bash
make bf
```

This will:

1. Run full model (saving required samples)
2. Run reduced model
3. Compute Bayes factor

Final result written to:

```
results/tables/data_analysis/bayes_factor_<RUN_TAG>.csv
```

---

# 4. Environment Variable Controls

All major scripts support overrides via environment variables.

Common variables:

```
JASA_N_ITER
JASA_N_WARMUP
JASA_D_LIST
JASA_M_LIST
JASA_METHODS
JASA_RUN_TAG
```

Example custom run:

```bash
JASA_N_ITER=20000 JASA_RUN_TAG=test_run make full
```

Kernel override example:

```bash
JASA_METHODS=exponential,half_gaussian make full
```

---

# 5. Supplementary Methods (Python)

## 5.1 Pseudo-Marginal MCMC (PMMH)

```
python/pmmh/pmmh_core.py
```

## 5.2 Hamiltonian Monte Carlo (HMC)

```
python/hmc/nuts_ilp_model.py
python/hmc/NUTS100_Experiments.ipynb
```

---

# 6. Plotting

Plotting notebooks currently reside in:

```
python/plots/Integer_Programming.ipynb
```

Generated figures are intended to be saved under:

```
results/figures/
```

---

# 7. Expected Runtime

Approximate runtime on a modern laptop:

- `make smoke`: minutes
- `make full`: potentially hours (large grid + long chains)
- `make kernel_compare`: moderate
- `make duck_full` / `make duck_reduced`: depends on iterations
- `make bf`: fast once samples exist

---

# 8. Summary of Make Targets

| Command | Purpose |
|----------|----------|
| `make smoke` | Quick pipeline sanity check |
| `make full` | Full simulation (main paper) |
| `make kernel_compare` | Supplementary kernel comparison |
| `make duck_full` | Full data analysis model |
| `make duck_reduced` | Reduced data model |
| `make bf` | Compute Bayes factor |
| `make clean` | Remove all generated results |

---

# 9. Notes for Reviewers

- No manual editing of file paths is required.
- All outputs are written under `results/`.
- All model settings are controlled via environment variables.
- Default settings reproduce main paper results.
- Kernel comparison is explicitly separated from main results.
