# Statistical Modeling for Combinatorial Response Data Reproducibility Repository

This repository contains code and data to reproduce the results in the paper.

All commands must be run from the repository root.

Generated outputs are written to:

```
results/
```

> **First time?** Set up R and Python dependencies first — see [INSTALL.md](INSTALL.md).

---

# Quick Start

### 1️⃣ Sanity check

```bash
make smoke
make probit_smoke
```

Runs a lightweight simulation and data analysis to verify everything works.

---

### 2️⃣ Main simulation study (MH-within-Gibbs)

```bash
make full
```

- The default setting does not run the full grid of simulation settings in the paper, to save time. To run the sampler with custom settings, change the corresponding environment variables first. See §6 in ```REPRODUCIBILITY.md``` for details.

- **Note:** `scripts/run_full.sh` defaults to `JASA_METHODS="exponential halfgaussian"`, so `make full` produces results for **both** kernels in one pass. The main paper uses only the exponential kernel; the half-Gaussian results feed into Supplementary Table 3. To run only the main-paper kernel:
  ```bash
  JASA_METHODS=exponential make full
  ```

- Outputs saved under:
  ```
  results/runs/mh_within_gibbs/
  ```

---

### 3️⃣ Kernel comparison (supplementary)

```bash
make kernel_compare
```

- Runs both:
  - exponential
  - half_gaussian

- Outputs saved under:
  ```
  results/runs/mh_within_gibbs/kernel_compare/
  ```

---

### 4️⃣ Data analysis (waterfowl matching)

Full model:

```bash
make duck_full
```

Reduced model:

```bash
make duck_reduced
```

- Outputs saved under:
  ```
  results/data_analysis/<RUN_TAG>/
  ```


---

# ⚙️ Custom Settings

All scripts support environment-variable overrides. For more details, see §6 in ```REPRODUCIBILITY.md```.

Example:

```bash
JASA_N_ITER=20000 make full
```
```bash
JASA_METHODS=exponential,half_gaussian make full
```

---

# 📄 Full Documentation

See:

```
REPRODUCIBILITY.md
```

for detailed mapping between paper sections and scripts.
