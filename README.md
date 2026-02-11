# JASA Reproducibility Package

This repository contains code and data to reproduce results for the JASA submission.

**Algorithms**
- MH-within-Gibbs (R; proposed method in the main paper)
- PMMH (Python; supplementary comparison)
- HMC (Python; supplementary comparison / approximate posterior)

**Plotting**
- Plotting code currently lives in Jupyter notebooks under `python/plots/`. (May require manual execution until scripts are unified.)

## Quick start (smoke test)
From the repository root:

```bash
make smoke
