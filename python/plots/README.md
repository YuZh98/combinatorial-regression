# `python/plots/` — Figure & table reproduction notebook

This folder contains a single Jupyter notebook, **`plotting.ipynb`**, that reproduces every figure and table in the paper from previously-saved MCMC outputs. The notebook is purely a *consumer* of saved samples — it does not run any sampler itself.

> **First time?** Set up R + Python dependencies via [`../../INSTALL.md`](../../INSTALL.md) and run the relevant samplers (see [Prerequisites](#prerequisites) below) before opening the notebook.

---

## Contents

- [`plotting.ipynb`](plotting.ipynb) — the notebook described below. Tracked in git as `Plotting.ipynb` (capital `P`); see the [Known issues](#known-issues) section if you're on a case-sensitive filesystem.
- This `README.md`.

---

## Prerequisites

The notebook reads pre-computed MCMC outputs from `../../results/runs/`. Run the corresponding samplers from the **repository root** before plotting; each section of the notebook depends on a specific sampler.

| Notebook section | Required sampler | Run command (from repo root) | Outputs read by notebook |
|---|---|---|---|
| §2 Fitted value curve | Probit (R) | `make probit` | `results/runs/probit/probit_default/*.rds` |
| §3 Signal recovery & efficiency | MH-within-Gibbs (R) | `make full` | `results/runs/mh_within_gibbs/full_example/*.rds` |
| §4 Waterfowl matching — full model | Duck full (R) | `make duck_full` | `results/runs/data_analysis/duck_full/*.rds`, `B_matrix.csv` |
| §4 Waterfowl matching — reduced model | Duck reduced (R) | `make duck_reduced` | `results/runs/data_analysis/duck_reduced/*.rds`, `B_matrix.csv` |
| §5 NUTS-HMC | NUTS-HMC (Python) | run `python/hmc/batch_run_nuts.ipynb` | `results/runs/hmc/*__samples.npz` |
| §6 PMMH | PMMH (Python) | `python python/pmmh/run_pmmh.py` | `results/runs/pmmh/...PMMH_iter*.npz` (see [Known issues](#known-issues) for path layout) |

For the smoke-test variants (`make smoke`, `make probit_smoke`) the output filenames have a different `RUN_TAG`; you'll need to update the hard-coded paths in the corresponding notebook cells if you want to plot smoke-test results. See [`REPRODUCIBILITY.md` §6](../../REPRODUCIBILITY.md) for the full env-var reference.

---

## Notebook structure

The notebook has six top-level sections, each with its own H1 heading. The very first cell is an overview cell containing a clickable table of contents and a paper-figure → producing-cell map (anchor-based; see [Navigation](#navigation) below).

| § | Section | Purpose |
|---|---|---|
| 0 | **Preparation** | Imports, the `MCMCAnalysis` class, `batch_save_trace_plots` / `batch_save_ACF_plots` helpers, the fitted-value-curve helpers (`ilp_map_batch`, `compute_ilp_probs`, `smooth_df_probs`), and the waterfowl helper `gen_prob_dict`. Each helper has a NumPy-style docstring; you don't need to read the implementations to follow the figure cells below. |
| 1 | **Fitted value curve** | Reproduces Figure 2 (predicted class probabilities under unconstrained vs constrained vs ground-truth models). |
| 2 | **Signal recovery and computational efficiency under various dimensions** | Reproduces Figure 3 (trace + ACF) and Figure 4 / Supp. Figure 3 (violin plot) for the MH-within-Gibbs simulation grid. |
| 3 | **Data analysis on waterfowl matching** | Reproduces Figures 5–8 (main paper) and Supp. Figures 1–2 from the duck full and reduced models. |
| 4 | **Simulation using Hamiltonian Monte Carlo** | Reproduces Supp. Figure 4 (trace + ACF) for NUTS-HMC. |
| 5 | **Simulation using Pseudo-Marginal MCMC** | Reproduces Supp. Figure 5 (trace + ACF) for PMMH. |

### Coefficient-naming note

The waterfowl-matching analysis uses two coefficient blocks whose names in the codebase do **not** match the paper:

| Paper | Codebase | Shape | Meaning |
|---|---|---|---|
| $\beta_0$ | `a[0]` | scalar | intercept |
| $\beta_{1:1}$ | `a[1]` | scalar | male-weight coefficient |
| $\beta_{1:2}$ | `a[2]` | scalar | female-weight coefficient |
| $\beta_2$ | `rho` | $(\kappa, K)$ | spline coefficients per species |

So `a` (length 3) carries the linear coefficients and `rho` carries the spline matrix. The notebook's overview cell repeats this table; the trace / ACF / posterior plots are produced separately for each block.

---

## Outputs

By default the notebook writes:

- **Figures** to `../../results/figures/<sub>/` (`mh_within_gibbs/`, `duck_matching/`, `hmc/`, `pmmh/`, …).
- **Intermediate artifacts** (e.g., the credible-band Monte-Carlo dictionaries) to `../../results/runs/data_analysis/duck_*/ci_related_reduced/*.pkl`.

Display-only cells (the stack-plot, the corrected-vs-uncorrected line plots) call `plt.show()` and don't write files; export them by hand if you want PNG/PDF copies.

---

## Navigation

The notebook supports two complementary ways to jump to a specific paper artifact:

1. **Anchor links from the overview cell.** The figure map at the top contains clickable `[↗](#fig-7)` links that scroll directly to the markdown cell immediately above each producing code cell. Anchors are stable under cell insertion / reordering because they live inside cell *content*, not at a numeric position. They render in JupyterLab, classic Jupyter, [nbviewer](https://nbviewer.org), GitHub's notebook renderer, and VS Code.
2. **Greppable producer markers.** Every figure-producing code cell starts with a header comment of the form `# === Producer: <Figure X> ===`. From a terminal:
   ```bash
   grep -n "Producer: Figure 7" plotting.ipynb
   ```
   gives you the line number inside the raw `.ipynb` JSON, which translates one-to-one to a cell.

Cell **indices** (e.g., "cell [17]") are *not* used as a stable reference because Jupyter doesn't display them in any consistent way and they shift on every cell insertion.

---

## Running the notebook

From the repository root, with the dependencies in [`../../requirements.txt`](../../requirements.txt) installed (see [`../../INSTALL.md`](../../INSTALL.md) §2):

```bash
# JupyterLab (recommended)
jupyter lab python/plots/plotting.ipynb

# Classic Jupyter
jupyter notebook python/plots/plotting.ipynb

# Headless (run all cells, write outputs in place)
jupyter nbconvert --to notebook --execute --inplace python/plots/plotting.ipynb
```

Most cells expect to be run **in order**, top to bottom, from a clean kernel. A handful of variable names (`sample_dir`, `image_dir`, `n, p, d, m`, `Analyzer`, `legend_order`, …) are reused across sections, so re-running cells out of order may silently produce wrong plots. If in doubt: *Kernel → Restart & Run All*.

The most expensive section is §4's credible-band Monte-Carlo loop (≈ 24 h on a MacBook Pro for the documented settings). To shorten it, reduce `n_sim` or thin `a_subsamples` / `rho_subsamples` more aggressively; the loop also writes a checkpoint every 100 samples so you can resume from a `.pkl`.

---

## Known issues

### 1. Filename casing on Linux / case-sensitive filesystems

The notebook is tracked in git as **`Plotting.ipynb`** (capital `P`) but the working-tree filename on the maintainer's macOS system is `plotting.ipynb` (lowercase). APFS / HFS+ on macOS is case-insensitive by default, so both names resolve to the same file there.

On Linux or any case-sensitive filesystem, `git checkout` produces `Plotting.ipynb` and references to the lowercase `plotting.ipynb` (in this README and in `REPRODUCIBILITY.md`) will fail to resolve. Workaround until this is renamed in the repo:

```bash
# Linux / CI: pick one
ln -s Plotting.ipynb plotting.ipynb
# or, properly fix it on the branch:
git mv Plotting.ipynb plotting.ipynb && git commit -m "fix: rename to lowercase"
```

The intended canonical name is **lowercase** `plotting.ipynb`.

### 2. PMMH output path depends on a source-side fix

The PMMH cell (§6) sets `sample_dir = "../../results/runs/pmmh/pmmh_samples"` and a clean `sample_files = ["PMMH_iter3000_..."]`. This assumes `python/pmmh/run_pmmh.py` writes its `.npz` to a `pmmh_samples/` subdirectory. As shipped, `run_pmmh.py` joins its `npz_dir` and `base_filename` with `+` (no path separator), so the file actually lands at `results/runs/pmmh/pmmh_samples<filename>` — `pmmh_samples` becomes part of the filename rather than a subdirectory.

Until `run_pmmh.py` is patched to use `os.path.join(npz_dir, base_filename)`, change the §6 cell to:

```python
sample_dir = "../../results/runs/pmmh"
sample_files = [
    "pmmh_samplesPMMH_iter3000_n1000_p5_d2_m1_M1000_tau1_ps0p01_alpha0p5_adapt0_seed1234.npz"
]
```

The notebook contains a markdown warning above the §6 cell with the same instructions.

### 3. The notebook is shipped without cell outputs

To keep diffs small, `plotting.ipynb` is committed without execution counts or rendered outputs. To preview a figure without re-running the whole pipeline, execute the relevant section's cells with `Run All Above` after running the corresponding sampler.

---

## Where to look next

- **Paper ↔ code mapping:** the overview cell at the top of `plotting.ipynb`, plus [`../../REPRODUCIBILITY.md` §2 and §3](../../REPRODUCIBILITY.md).
- **Env-var configuration for samplers:** [`../../REPRODUCIBILITY.md` §6](../../REPRODUCIBILITY.md).
- **Dependency installation:** [`../../INSTALL.md`](../../INSTALL.md).
- **NUTS-HMC API:** [`../hmc/API.md`](../hmc/API.md).
