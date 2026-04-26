# Installation Guide

This bundle uses both **R** (core MH-within-Gibbs sampler + waterfowl analysis) and **Python** (NUTS-HMC and PMMH baselines + plotting). Both stacks need to be installed before any of the `make` targets in [README.md](README.md) will work.

> **Time required:** ~10 min for a clean install; less if R / Python are already set up.

---

## 1. R

### 1.1 R version

The code has been run on R ≥ 4.2. Any reasonably recent R should work; check with:

```r
R.version.string
```

### 1.2 Required CRAN packages

From an R prompt at the repo root:

```r
install.packages(c(
  "Rcpp",
  "RcppArmadillo",
  "truncnorm",
  "lpSolve",
  "lintools",
  "bridgesampling",
  "mvtnorm"
  # 'splines' is shipped with base R; no need to install separately
))
```

The first run of any sampling script will trigger Rcpp compilation of the C++ files in `R/src/cpp/`. Compilation requires a working C++ toolchain:

- **macOS:** `xcode-select --install` (Apple Clang) or install GCC via Homebrew.
- **Linux:** `sudo apt install r-base-dev build-essential` (or your distro's equivalent).
- **Windows:** install [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version.

OpenMP is used to parallelize the hit-and-run inner loop. If your toolchain doesn't support OpenMP, the code will still compile (the `#pragma omp` lines become no-ops) but will run single-threaded.

### 1.3 Verifying the R install

```bash
make smoke
```

should complete in well under a minute and write outputs under `results/runs/mh_within_gibbs/smoke/` and `results/runs/data_analysis/duck_smoke/`.

### 1.4 Pinning R package versions (optional, recommended for replication)

For exact replication of the paper's tables, capture the package versions you used:

```r
# from R, at the repo root
renv::init()      # creates renv.lock
renv::snapshot()  # records current package versions
```

Commit the resulting `renv.lock` alongside this file. Future reviewers can then run `renv::restore()` to install the exact set.

---

## 2. Python

### 2.1 Python version

Tested on Python ≥ 3.10. Earlier versions may work but JAX increasingly requires 3.10+.

### 2.2 Quick install

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs loose-pinned versions of:
- numpy, scipy, pandas, matplotlib
- jax, jaxlib, numpyro (for `python/hmc/`)
- arviz (for `python/pmmh/pmmh_diagnostics.py`)
- jupyter (to run the `.ipynb` entry points)

> **GPU note:** `pip install jax jaxlib` installs the CPU build. The paper's tables are reproducible on CPU; if you want GPU acceleration for large NUTS runs, see the [JAX install instructions](https://jax.readthedocs.io/en/latest/installation.html).

### 2.3 Pinning Python package versions (recommended for replication)

The committed `requirements.txt` uses `>=` floors, not exact versions. For an exact replication, freeze the working environment:

```bash
pip freeze > requirements-lock.txt
git add requirements-lock.txt
git commit -m "build: add Python lockfile"
```

Reviewers can then install the exact set with:

```bash
pip install -r requirements-lock.txt
```

### 2.4 Verifying the Python install

The simplest end-to-end check is the PMMH driver:

```bash
python python/pmmh/run_pmmh.py
```

This runs a small (n=1000, p=5, d=2, m=1) PMMH chain in a couple of minutes and writes outputs under `results/runs/pmmh/`.

For NUTS-HMC, open `python/hmc/batch_run_nuts.ipynb` in Jupyter and run all cells; expect ≈ 5–15 min depending on the loop's `(d, m)` grid.

---

## 3. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Rcpp::sourceCpp` fails with "no C++ compiler" | Missing toolchain | See §1.2. |
| `Error in is_totally_unimodular(A)` | `lintools` not installed | `install.packages("lintools")` |
| `ImportError: No module named jax` | Python deps not installed | See §2.2. |
| `make smoke` writes to a different folder | Stale `JASA_*` env vars in your shell | `unset $(env | grep -o '^JASA_[A-Z_]*')` then re-run. |
| Different RMSE numbers than the paper | RNG depends on `OMP_NUM_THREADS` (the C++ hit-and-run uses R's RNG inside an OpenMP region) | Set `OMP_NUM_THREADS=1` for byte-exact replication. |
| Reviewer on Linux / case-sensitive FS can't find `plotting.ipynb` | Tracked filename is `Plotting.ipynb`; see [python/plots/README.md](python/plots/README.md) | Symlink or rename per that file's instructions. |

---

## 4. What this guide does *not* cover

- A bit-exact replication of the paper's tables. The hit-and-run sampler's parallel section consumes R's RNG state in an order that depends on thread scheduling, so two runs with different `OMP_NUM_THREADS` (or different OpenMP runtimes) may produce slightly different `beta_samples`. For the strictest reproducibility, set `OMP_NUM_THREADS=1`.
- CI integration. The repo has no `.github/workflows/`; reviewers should run `make smoke` manually as a sanity check.
- An R `DESCRIPTION` file. The R package list in §1.2 is the canonical source.
