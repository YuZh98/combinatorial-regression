# JASA Reproducibility Repository

This repository contains code and data to reproduce the results in the paper.

All commands must be run from the repository root.

Generated outputs are written to:

```
results/
```

---

# 🚀 Quick Start

### 1️⃣ Smoke test (sanity check)

```bash
make smoke
```

Runs a lightweight simulation and data analysis to verify everything works.

---

### 2️⃣ Main simulation study (MH-within-Gibbs)

```bash
make full
```

- Uses exponential kernel (main paper default)
- Runs full grid of simulation settings
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
- Uses lighter iteration defaults
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

Bayes factor comparison:

```bash
make bf
```

Bayes factor result is written to:

```
results/tables/data_analysis/
```

---

# ⚙️ Custom Settings

All scripts support environment-variable overrides.

Example:

```bash
JASA_N_ITER=20000 make full
```

Kernel override:

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
