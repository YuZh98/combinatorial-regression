# API Reference

Quick reference for the main user-facing functions in `nuts_ilp_model.py`.

---

## Main Entry Point

### `run_experiment`

Complete end-to-end experiment: data generation, MCMC sampling, diagnostics, and saving.

**Signature:**
```python
run_experiment(
    algorithm: Literal["baseline", "marginal_joint", "marginal_gibbs"],
    n: int,           # number of observations
    p: int,           # number of predictors
    d: int,           # dimension of binary outcome
    m: int,           # number of constraints
    tau_beta: float = 1.0,        # prior variance for β
    mcmc_config: Optional[MCMCConfig] = None,
    gibbs_config: Optional[GibbsConfig] = None,
    diagnostic_config: Optional[DiagnosticConfig] = None,
    data_seed: int = 123,
    mcmc_seed: int = 456,
    use_simulated_data: bool = True,
    save_results: bool = True,
    run_diagnostics: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[mcmc_or_info, samples, diagnostics]
```

**Returns:**
- `mcmc_or_info`: MCMC object (for baseline/marginal_joint) or dict (for marginal_gibbs)
- `samples`: dict with keys "beta", "u" (and "v", "zeta" for baseline)
- `diagnostics`: dict with ESS, R-hat, RMSE, warnings, etc.

**Basic Example:**
```python
mcmc, samples, diagnostics = run_experiment(
    algorithm="marginal_joint",
    n=1000, p=5, d=50, m=20,
    tau_beta=1.0,
)
```

---

## Configuration Classes

### `MCMCConfig`

Settings for NUTS/HMC sampling.

**Fields:**
- `num_warmup: int = 2000` — warmup iterations
- `num_samples: int = 3000` — post-warmup samples
- `num_chains: int = 1` — number of chains
- `target_accept_prob: float = 0.75` — target acceptance probability
- `dense_mass: bool = False` — use dense mass matrix
- `step_size: float = 0.05` — initial step size (ignored by NUTS)

### `GibbsConfig`

Settings for Gibbs sampler (only for `marginal_gibbs`).

**Fields:**
- `num_beta_steps: int = 5` — NUTS iterations for β | U block
- `num_u_steps: int = 5` — NUTS iterations for U | β block
- `num_outer_iterations: int = 1000` — total Gibbs iterations
- `num_warmup_iterations: int = 500` — warmup Gibbs iterations

### `DiagnosticConfig`

Settings for diagnostic plots.

**Fields:**
- `max_plots: int = 8` — number of β entries to plot
- `max_lag: int = 50` — maximum lag for ACF plots

---

## Saving & Outputs

### Naming Convention

All saved artifacts share a common **run stub** encoding all replication parameters:

```
{alg}__{dims}__{tau}__{mcmc}__{seeds}[__{gibbs}]
```

**Example:**
```
mj__n1000_p5_d50_m20__tau1.0__w2000_s3000_c1_ap0.75__ds123_ms456__samples.npz
```

**Components:**
- `alg`: Algorithm abbreviation (`bl`=baseline, `mj`=marginal_joint, `mg`=marginal_gibbs)
- `dims`: `n{n}_p{p}_d{d}_m{m}`
- `tau`: `tau{tau_beta}`
- `mcmc`: `w{num_warmup}_s{num_samples}_c{num_chains}_ap{target_accept_prob}` (adds `_dm1` if dense_mass=True)
- `seeds`: `ds{data_seed}_ms{mcmc_seed}`
- `gibbs` (marginal_gibbs only): `gb{num_beta_steps}_gu{num_u_steps}_go{num_outer}_gw{num_warmup}`

### Saved Artifacts

| Suffix | Format | Contents |
|--------|--------|----------|
| `__samples.npz` | NumPy compressed | Posterior samples (beta, u, v, zeta), beta_true |
| `__meta.json` | JSON | Full config, shapes, seeds, version info, timestamp |
| `__summary.json` | JSON | Key scalars only (for batch loading), artifact paths |
| `__diagnostics.pdf` | PDF | Trace plots, ACF plots, posterior comparison |

### Utility Functions

#### `build_run_stub`

Build the replication-complete filename stub.

```python
stub = build_run_stub(
    algorithm="marginal_joint",
    n=1000, p=5, d=50, m=20,
    tau_beta=1.0,
    mcmc_config=MCMCConfig(),
    gibbs_config=None,
    data_seed=123,
    mcmc_seed=456,
)
```

#### `get_artifact_paths`

Get all artifact file paths from a stub.

```python
paths = get_artifact_paths(stub, output_dir="./results")
```

### Meta File Contents

- `timestamp`: ISO format datetime
- `algorithm`: Full algorithm name
- `dimensions`: `{n, p, d, m, num_u_active}`
- `hyperparameters`: `{tau_beta, constraint_tolerance}`
- `mcmc_config`: Full MCMCConfig as dict
- `gibbs_config`: Full GibbsConfig as dict (or null)
- `seeds`: `{data_seed, mcmc_seed}`
- `shapes`: Dictionary of array shapes
- `version_info`: Python, JAX, NumPyro versions

### Summary File Contents

Scalars for batch loading:
- `stub`, `algorithm`, `n`, `p`, `d`, `m`, `tau_beta`
- `num_warmup`, `num_samples`, `num_chains`, `target_accept_prob`
- `data_seed`, `mcmc_seed`
- `artifact_paths`, `runtime_seconds`
- Placeholders: `ess_min`, `ess_median`, `rhat_max`, `rmse`

---

## Direct Sampler Functions

### `run_nuts_baseline`

Run baseline NUTS on (β, U, V).

### `run_nuts_marginal`

Run marginal joint NUTS on (β, U) with V integrated out.

### `run_gibbs_marginal`

Run Gibbs sampler with alternating NUTS kernels.

---

## Data Generation

### `generate_simulation_data`

Generate synthetic data for the constrained latent variable model.

```python
y, X, A, b, beta_true, zeta_true = generate_simulation_data(
    n=1000, p=5, d=100, m=30,
    tau_beta=1.0,
    rng_seed=42,
)
```

---

## Diagnostic Functions

### `compute_beta_diagnostics`

Compute ESS, R-hat, RMSE, coverage for β samples.

### `plot_beta_diagnostics`

Create trace plots, ACF plots, posterior comparison.

### `check_mcmc_health`

Check for convergence issues and return warnings.

---

## Batch Loading Example

```python
import json
import glob
import pandas as pd

summaries = []
for path in glob.glob("./results/*__summary.json"):
    with open(path) as f:
        summaries.append(json.load(f))

df = pd.DataFrame(summaries)
print(df[["algorithm", "d", "runtime_seconds"]])
```

---

## Constants

- `CONSTRAINT_TOLERANCE = 1e-6`
- `DEFAULT_SIM_OUT_DIR = "./Constraints/Results/Simulation_HMC"`
