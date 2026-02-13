"""
Constrained Latent Variable Model with ILP/LP Mapping
======================================================

This module implements HMC/NUTS sampling for a binary outcome model with:
- Latent Gaussian variables ζ_i | β ~ N(x_i^T β, I_d)
- Binary outcomes y_i = argmax_{z ∈ {0,1}^d ∩ P} ζ_i^T z via LP relaxation
- Polytope constraints P = {z : Az ≤ b}
- Augmentation variables: U (dual) and V (residual)

TARGET DISTRIBUTION (drop-c surrogate):
The code targets p(β, U, V | data) with intractable normalizing constants
c(y_i, ζ_i) dropped for computational feasibility. This is the surrogate
posterior and is the agreed-upon target for all algorithms.

THREE ALGORITHMS:
1. Baseline: Joint NUTS on (β, U, V)
2. Marginal Joint: Joint NUTS on (β, U) with V analytically marginalized

AUTHOR: Research implementation for single-user advanced use
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
import numpyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.optimize import linprog
import os
import time
import json
import platform
import sys
import gc
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Literal
import warnings

# ============================================================
# SECTION 1: CONFIGURATION & CONSTANTS
# ============================================================

# Numerical tolerance for constraint detection (used in build_struct)
CONSTRAINT_TOLERANCE = 1e-6

# Directory paths with convenient defaults
DEFAULT_SIM_DATA_DIR = "./Constraints/Simulation_DATA"
DEFAULT_SIM_OUT_DIR = "./Constraints/Results/Simulation_HMC"
DEFAULT_DIAG_DIR = "./Constraints/Results/NUTS_Diagnostics"

# Validation tolerance for post-sampling constraint checks
VALIDATION_EPSILON = 1e-5


@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling."""
    num_warmup: int = 2000
    num_samples: int = 3000
    num_chains: int = 1
    target_accept_prob: float = 0.75
    dense_mass: bool = False
    step_size: float = 0.05  # Only used for HMC, ignored by NUTS
    green_mode: bool = True  # If True, marginal_joint is forced to be marginal_joint_green
    only_save_beta_samples: bool = True   # If True, marginal_joint and baseline only save beta samples after MCMC (During MCMC runs, other parameters are still saved) 


@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic outputs."""
    max_plots: int = 8
    max_lag: int = 50
    num_corr_beta: int = 6
    include_ppc: bool = False
    num_ppc_draws: int = 20
    num_ppc_obs: int = 100


# ============================================================
# SECTION 1.5: NAMING AND SAVING UTILITIES
# ============================================================

def build_run_stub(
    algorithm: str,
    n: int,
    p: int,
    d: int,
    m: int,
    tau_beta: float,
    mcmc_config: "MCMCConfig",
    data_seed: int,
    mcmc_seed: int,
) -> str:
    """
    Build a replication-complete filename stub from run configuration.
    
    The stub encodes all parameters needed to replicate the run exactly.
    Format uses double underscores (__) to separate parameter groups
    and single underscores (_) within values.
    
    Parameters
    ----------
    algorithm : str
        Algorithm name: "baseline", "marginal_joint", or "marginal_gibbs"
    n, p, d, m : int
        Problem dimensions
    tau_beta : float
        Prior variance for β
    mcmc_config : MCMCConfig
        MCMC configuration
    data_seed : int
        Seed for data generation
    mcmc_seed : int
        Seed for MCMC sampling
    
    Returns
    -------
    stub : str
        Filename stub (no extension, no directory)
    
    Example
    -------
    >>> stub = build_run_stub("marginal_joint", 1000, 5, 50, 20, 1.0,
    ...                       MCMCConfig(), None, 123, 456)
    >>> # Returns: "mj__n1000_p5_d50_m20__tau1.0__w2000_s3000_c1_ap0.75__ds123_ms456"
    """
    # Algorithm aliasing
    alg_alias = {
        "baseline": "baseline",
        "marginal_joint": "marginal_joint_ptntl",
        "marginal_joint_model_version": "marginal_joint_mdl",
        "marginal_joint_green": "marginal_joint_green_ptntl",
    }
    alg_str = alg_alias.get(algorithm, algorithm[:3])
    
    # Problem dimensions
    dims_str = f"n{n}_p{p}_d{d}_m{m}"
    
    # Model hyperparameter
    tau_str = f"tau{tau_beta}"
    
    # MCMC config (run-critical parameters only)
    mcmc_str = (
        f"w{mcmc_config.num_warmup}_"
        f"s{mcmc_config.num_samples}_"
        f"c{mcmc_config.num_chains}_"
        f"ap{mcmc_config.target_accept_prob}"
    )
    
    # Add dense_mass only if True (non-default)
    if mcmc_config.dense_mass:
        mcmc_str += "_dm1"
    
    # Seeds
    seed_str = f"ds{data_seed}_ms{mcmc_seed}"
    
    # Build base stub
    stub = f"{alg_str}__{dims_str}__{tau_str}__{mcmc_str}__{seed_str}"
    
    # Add Gibbs-specific config if applicable
    if algorithm == "marginal_gibbs" and gibbs_config is not None:
        gibbs_str = (
            f"gb{gibbs_config.num_beta_steps}_"
            f"gu{gibbs_config.num_u_steps}_"
            f"go{gibbs_config.num_outer_iterations}_"
            f"gw{gibbs_config.num_warmup_iterations}"
        )
        stub = f"{stub}__{gibbs_str}"
    
    return stub


def get_artifact_paths(stub: str, output_dir: str) -> Dict[str, str]:
    """
    Generate all artifact file paths from a run stub.
    
    Parameters
    ----------
    stub : str
        Run stub from build_run_stub()
    output_dir : str
        Output directory path
    
    Returns
    -------
    paths : dict
        Dictionary with keys:
        - "samples": path to samples .npz file
        - "meta": path to metadata .json file
        - "summary": path to summary .json file
        - "diagnostics_pdf": path to diagnostics .pdf file
    """
    return {
        "samples": os.path.join(output_dir, f"{stub}__samples.npz"),
        "meta": os.path.join(output_dir, f"{stub}__meta.json"),
        "summary": os.path.join(output_dir, f"{stub}__summary.json"),
        "diagnostics_pdf": os.path.join(output_dir, f"{stub}__diagnostics.pdf"),
    }


def get_version_info() -> Dict[str, str]:
    """
    Get version information for reproducibility.
    
    Returns
    -------
    info : dict
        Dictionary with Python, JAX, NumPyro versions and platform info.
    """
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "numpyro_version": numpyro.__version__,
        "numpy_version": np.__version__,
    }
    return info


def save_meta_json(
    path: str,
    algorithm: str,
    n: int,
    p: int,
    d: int,
    m: int,
    tau_beta: float,
    mcmc_config: "MCMCConfig",
    diagnostic_config: "DiagnosticConfig",
    data_seed: int,
    mcmc_seed: int,
    shapes: Dict[str, Tuple],
    num_u_active: int,
) -> None:
    """
    Save comprehensive metadata JSON for a run.
    
    Parameters
    ----------
    path : str
        Output file path
    algorithm : str
        Algorithm name
    n, p, d, m : int
        Problem dimensions
    tau_beta : float
        Prior variance
    mcmc_config : MCMCConfig
    diagnostic_config : DiagnosticConfig
    data_seed, mcmc_seed : int
        Random seeds
    shapes : dict
        Dictionary of array shapes (e.g., {"y": (n, d), "X": (n, p), ...})
    num_u_active : int
        Number of active constraint positions (L)
    """
    meta = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "dimensions": {
            "n": n,
            "p": p,
            "d": d,
            "m": m,
            "num_u_active": num_u_active,
        },
        "hyperparameters": {
            "tau_beta": tau_beta,
            "constraint_tolerance": CONSTRAINT_TOLERANCE,
        },
        "mcmc_config": asdict(mcmc_config),
        "diagnostic_config": asdict(diagnostic_config),
        "seeds": {
            "data_seed": data_seed,
            "mcmc_seed": mcmc_seed,
        },
        "shapes": {k: list(v) for k, v in shapes.items()},
        "version_info": get_version_info(),
    }
    
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def save_summary_json(
    path: str,
    stub: str,
    algorithm: str,
    n: int,
    p: int,
    d: int,
    m: int,
    tau_beta: float,
    mcmc_config: "MCMCConfig",
    data_seed: int,
    mcmc_seed: int,
    artifact_paths: Dict[str, str],
    runtime_seconds: Optional[float] = None,
    efficiency_metrics: Optional[Dict] = None,
    diagnostics: Optional[Dict] = None,
) -> None:
    """
    Save a summary JSON with key scalars only.
    
    This file is designed for easy batch loading and aggregation.
    
    Parameters
    ----------
    path : str
        Output file path
    stub : str
        Run stub
    algorithm : str
        Algorithm name
    n, p, d, m : int
        Problem dimensions
    tau_beta : float
        Prior variance
    mcmc_config : MCMCConfig
    data_seed, mcmc_seed : int
        Random seeds
    artifact_paths : dict
        Paths to all saved artifacts
    runtime_seconds : float or None
        MCMC runtime in seconds
    efficiency_metrics : dict or None
        Output from compute_efficiency_metrics
    diagnostics : dict or None
        Output from compute_beta_diagnostics (for ESS/Rhat/RMSE)
    """
    summary = {
        "stub": stub,
        "timestamp": datetime.now().isoformat(),
        "algorithm": algorithm,
        "n": n,
        "p": p,
        "d": d,
        "m": m,
        "tau_beta": tau_beta,
        "num_warmup": mcmc_config.num_warmup,
        "num_samples": mcmc_config.num_samples,
        "num_chains": mcmc_config.num_chains,
        "target_accept_prob": mcmc_config.target_accept_prob,
        "dense_mass": mcmc_config.dense_mass,
        "data_seed": data_seed,
        "mcmc_seed": mcmc_seed,
        "artifact_paths": artifact_paths,
        "runtime_seconds": runtime_seconds,
        # Diagnostic scalars (populated from diagnostics dict if available)
        "ess_min": None,
        "ess_median": None,
        "rhat_max": None,
        "rmse": None,
        # Efficiency metrics (populated from efficiency_metrics dict if available)
        "ess_per_second_min": None,
        "ess_per_second_median": None,
        "seconds_per_ess_median": None,
        "mean_accept_prob": None,
        "num_divergences": None,
        "frac_divergences": None,
    }
    
    
    # Populate diagnostic scalars if available
    if diagnostics is not None:
        if "ess_min" in diagnostics:
            summary["ess_min"] = diagnostics["ess_min"]
        if "ess_median" in diagnostics:
            summary["ess_median"] = diagnostics["ess_median"]
        if "rhat_max" in diagnostics:
            summary["rhat_max"] = diagnostics["rhat_max"]
        if "rmse" in diagnostics:
            summary["rmse"] = diagnostics["rmse"]
    
    # Populate efficiency metrics if available
    if efficiency_metrics is not None:
        for key in ["ess_per_second_min", "ess_per_second_median", 
                    "seconds_per_ess_median", "mean_accept_prob",
                    "num_divergences", "frac_divergences"]:
            if key in efficiency_metrics and efficiency_metrics[key] is not None:
                summary[key] = efficiency_metrics[key]
    
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def save_samples_npz(
    path: str,
    samples: Dict,
    beta_true: Optional[np.ndarray] = None,
) -> None:
    """
    Save samples to a compressed .npz file.
    
    Parameters
    ----------
    path : str
        Output file path
    samples : dict
        Dictionary of samples (keys depend on algorithm)
    beta_true : ndarray or None
        True β if available (for simulations)
    """
    # Prepare samples dict, converting JAX arrays to numpy
    save_dict = {}
    for key, val in samples.items():
        arr = np.array(val)
        # Flatten chains if present (4D -> 3D for beta)
        if arr.ndim == 4:
            arr = arr.reshape(-1, arr.shape[2], arr.shape[3])
        save_dict[key] = arr
    
    # Add beta_true if provided
    if beta_true is not None:
        save_dict["beta_true"] = np.array(beta_true)
    
    np.savez_compressed(path, **save_dict)




# ============================================================
# SECTION 2: DATA GENERATION
# ============================================================

def generate_tu_incidence_matrix(d: int, m: int, rng=None, random_b: bool = True):
    """
    Generate a totally unimodular matrix A (m × d) and RHS vector b (m,).

    Construction:
    - A is a node-arc incidence matrix of a directed graph on d nodes.
    - Each row k picks two distinct columns (j1, j2) and sets
      A[k, j1] = +1, A[k, j2] = -1 (or vice versa).
    - Such matrices are totally unimodular.
    - b is nonnegative (either all ones or random 0/1), so y = 0 is always feasible.

    Parameters
    ----------
    d : int
        Number of binary variables (columns of A).
    m : int
        Number of constraints (rows).
    rng : np.random.Generator or None
        Random number generator.
    random_b : bool
        If True, b[k] ∈ {0, 1}. If False, b[k] = 1 for all k.

    Returns
    -------
    A : (m, d) ndarray of ints in {-1, 0, 1}
    b : (m,) ndarray of ints (0 or 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    A = np.zeros((m, d), dtype=int)
    for k in range(m):
        # Pick two distinct columns
        j1, j2 = rng.choice(d, size=2, replace=False)
        # Random orientation
        if rng.random() < 0.5:
            A[k, j1] = 1
            A[k, j2] = -1
        else:
            A[k, j1] = -1
            A[k, j2] = 1

    if random_b:
        b = rng.integers(low=0, high=2, size=m, dtype=int)  # 0 or 1
    else:
        b = np.ones(m, dtype=int)

    return A, b


def lp_map(eta: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve max_z eta^T z s.t. Az ≤ b, 0 ≤ z_j ≤ 1 using LP relaxation.

    For totally unimodular A, the LP optimum is at an integral vertex,
    so the solution is binary.

    Parameters
    ----------
    eta : (d,) array
        Objective coefficients.
    A : (m, d) array
        Constraint matrix.
    b : (m,) array
        RHS vector.

    Returns
    -------
    z_opt : (d,) int array in {0, 1}
    """
    d = eta.shape[0]
    res = linprog(
        c=-eta,  # maximize eta^T z <=> minimize -eta^T z
        A_ub=A,
        b_ub=b,
        bounds=[(0.0, 1.0)] * d,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP solve failed: {res.message}")
    
    # For TU A, solution should be integral; rounding is safety net
    z_opt = np.round(res.x).astype(int)
    return z_opt


def generate_simulation_data(
    n: int = 1000,
    p: int = 5,
    d: int = 2,
    m: int = 1,
    tau_beta: float = 10.0,
    rng_seed: int = 123,
    random_b: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data consistent with the ILP-based model.

    Process:
    1. Generate TU matrix A and compatible b (Ay ≤ b non-empty).
    2. Generate X ~ N(0, I_p).
    3. Generate β_true ~ N(0, τ_β) elementwise (τ_β is variance).
    4. Generate ζ_true = X β_true + ε, ε ~ N(0, I_d).
    5. For each i, set y_i = argmax_z ζ_i^T z s.t. Az ≤ b, z ∈ {0,1}^d.

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of predictors.
    d : int
        Dimension of outcome vector y_i.
    m : int
        Number of linear constraints.
    tau_beta : float
        Prior variance for each β entry.
    rng_seed : int
        RNG seed for reproducibility.
    random_b : bool
        If True, b_k ∈ {0,1}. If False, b_k = 1 for all k.

    Returns
    -------
    y : (n, d) int array
    X : (n, p) float array
    A : (m, d) int array
    b : (m,) int array
    beta_true : (p, d) float array
    zeta_true : (n, d) float array
    """
    rng = np.random.default_rng(rng_seed)

    # 1) TU constraint matrix A and RHS b
    A, b = generate_tu_incidence_matrix(d=d, m=m, rng=rng, random_b=random_b)

    # 2) Covariates X
    X = rng.normal(size=(n, p))

    # 3) True β (variance = tau_beta)
    beta_true = rng.normal(loc=0.0, scale=np.sqrt(tau_beta), size=(p, d))

    # 4) Latent ζ
    eps = rng.normal(size=(n, d))
    zeta_true = X @ beta_true + eps

    # 5) LP mapping to y
    y = np.zeros((n, d), dtype=int)
    for i in range(n):
        eta = zeta_true[i, :]
        y[i, :] = lp_map(eta, A, b)

    return y, X, A, b, beta_true, zeta_true


# ============================================================
# SECTION 3: MODEL STRUCTURES & PREPROCESSING
# ============================================================

def validate_inputs(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float
) -> None:
    """
    Validate input data and hyperparameters.

    Checks:
    - Shape compatibility
    - Binary y
    - Positive tau_beta
    - Feasibility: Ay_i ≤ b for all i

    Raises
    ------
    ValueError or AssertionError if validation fails.
    """
    n, d = y.shape
    n2, p = X.shape
    m, d2 = A.shape
    
    assert n == n2, "X and y must have same number of rows"
    assert d == d2, "A and y must have compatible dimensions"
    assert b.shape == (m,), f"b must be length-m, got {b.shape}"
    assert tau_beta > 0, "tau_beta must be positive"
    
    # Check y is binary
    assert np.all((y == 0) | (y == 1)), "y must be binary {0, 1}"
    
    # Check feasibility: Ay ≤ b for all observations
    Ay = y @ A.T  # (n, m)
    violations = Ay - b  # (n, m)
    if np.any(violations > CONSTRAINT_TOLERANCE):
        raise ValueError("Some y_i violate Ay ≤ b constraint; data is infeasible.")


def build_struct(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float = 10.0
) -> Dict:
    """
    Precompute static quantities and identify active constraints.

    Active constraints: (Ay_i - b)_k ≈ 0 means constraint k is active for obs i.
    Inactive constraints: (Ay_i - b)_k < 0 means constraint k is slack for obs i.

    For active (i, k): u_{ik} > 0 (dual variable)
    For inactive (i, k): u_{ik} = 0 (fixed)

    Parameters
    ----------
    y : (n, d) array, binary {0, 1}
    X : (n, p) array, covariates
    A : (m, d) array, constraint matrix
    b : (m,) array, constraint RHS (Ay ≤ b)
    tau_beta : float
        Prior variance for each entry of β.

    Returns
    -------
    struct : dict with keys:
        n, d, p, m : int
            Dimensions.
        y, X, A, b : jnp arrays
            Data in JAX format.
        sign_v : (n, d) in {+1, -1}
            Sign for v_{ij} based on y_{ij}.
        idx_u_i : (L,) int32
            Row indices of active constraint positions.
        idx_u_k : (L,) int32
            Column indices of active constraint positions.
        num_u_active : int
            Total number of active positions L.
        tau_beta : float
            Prior variance.
    """
    # Validate first
    validate_inputs(y, X, A, b, tau_beta)
    
    # Convert to JAX arrays
    y = jnp.asarray(y)
    X = jnp.asarray(X)
    A = jnp.asarray(A)
    b = jnp.asarray(b)

    n, d = y.shape
    n2, p = X.shape
    m, d2 = A.shape

    # sign(v_{ij}) = +1 if y_{ij}=1, -1 if y_{ij}=0
    sign_v = jnp.where(y == 1, 1.0, -1.0)

    # Compute Ay - b for each observation
    Ay_minus_b = y @ A.T - b  # shape (n, m)

    # Active constraints: (Ay_i - b)_k ≈ 0
    # Note: We use isclose with tolerance for real-valued constraints
    active_mask = jnp.isclose(Ay_minus_b, 0.0, atol=CONSTRAINT_TOLERANCE)

    # Extract active positions as (i, k) pairs
    pos_idx = jnp.argwhere(active_mask)  # shape (L, 2)
    if pos_idx.shape[0] == 0:
        raise ValueError(
            "No active constraints (Ay == b); U would be trivial. "
            "Check data generation or increase m."
        )

    idx_u_i = pos_idx[:, 0].astype(jnp.int32)
    idx_u_k = pos_idx[:, 1].astype(jnp.int32)
    num_u_active = pos_idx.shape[0]

    struct = dict(
        n=n,
        d=d,
        p=p,
        m=m,
        y=y,
        X=X,
        A=A,
        b=b,
        sign_v=sign_v,
        idx_u_i=idx_u_i,
        idx_u_k=idx_u_k,
        num_u_active=num_u_active,
        tau_beta=tau_beta,
    )
    return struct


def validate_constraints(
    u: np.ndarray,
    v: np.ndarray,
    y: np.ndarray,
    idx_u_i: np.ndarray,
    idx_u_k: np.ndarray,
    tol: float = VALIDATION_EPSILON
) -> bool:
    """
    Validate that constrained samples satisfy all constraints.

    Checks:
    - v has correct signs: (y_{ij} - 0.5) * v_{ij} > -tol
    - u is positive for active constraints
    - u is zero for inactive constraints (not explicitly checked here)

    Parameters
    ----------
    u : (n, m) array
        Dual variables.
    v : (n, d) array
        Residual variables.
    y : (n, d) array
        Binary outcomes.
    idx_u_i, idx_u_k : (L,) arrays
        Active constraint indices.
    tol : float
        Tolerance for numerical errors.

    Returns
    -------
    valid : bool
        True if all constraints satisfied.
    """
    n, d = v.shape
    
    # Check v sign constraints
    sign_check = (y - 0.5) * v  # Should be positive
    if np.any(sign_check < -tol):
        warnings.warn(f"Sign constraint violated: min(sign_check) = {sign_check.min()}")
        return False
    
    # Check u positivity for active constraints
    u_active = u[idx_u_i, idx_u_k]
    if np.any(u_active < -tol):
        warnings.warn(f"Positivity constraint violated: min(u_active) = {u_active.min()}")
        return False
    
    return True


# ============================================================
# SECTION 4: TARGET DENSITIES
# ============================================================

def make_potential_fn_baseline(struct: Dict):
    """
    Build potential function U(params) = -log p(β, U, V | data) for baseline.

    This is the DROP-C SURROGATE target with full augmentation (β, U, V).

    Unconstrained parameters:
    - β : (p, d) (unconstrained real)
    - z_u : (L,) (unconstrained real → active u via softplus)
    - z_v : (n, d) (unconstrained real → signed v via softplus)

    Target:
    log p(β, z_u, z_v) ∝ 
      -½||β||²/τ_β                           [prior on β]
      - Σ_active softplus(z_u)                [exp(1) prior on u]
      + Σ_active log_sigmoid(z_u)             [Jacobian for u transform]
      + Σ_{i,j} log_sigmoid(z_v_{ij})        [Jacobian for v transform]
      - ½ Σ_i ||ζ_i - x_i^T β||²             [likelihood]

    where:
      u = build_u_from_z(z_u)
      v = build_v_from_z(z_v)
      ζ = u @ A + v  [Note: u @ A computes row i as u^{(i)} @ A = (A^T u^{(i)})^T]

    Parameters
    ----------
    struct : dict from build_struct

    Returns
    -------
    potential_fn : callable
        Function taking params dict, returning scalar potential energy.
    """
    y = struct["y"]
    X = struct["X"]
    A = struct["A"]
    sign_v = struct["sign_v"]
    idx_u_i = struct["idx_u_i"]
    idx_u_k = struct["idx_u_k"]
    n, d, m = struct["n"], struct["d"], struct["m"]
    tau_beta = struct["tau_beta"]

    def potential_fn(params):
        beta = params["beta"]   # (p, d)
        z_u = params["z_u"]     # (L,)
        z_v = params["z_v"]     # (n, d)

        # Transform z_u → u (n, m)
        u_active = jax.nn.softplus(z_u)  # positive
        logJ_u = jnp.sum(jax.nn.log_sigmoid(z_u))

        u = jnp.zeros((n, m))
        u = u.at[idx_u_i, idx_u_k].set(u_active)

        # Transform z_v → v (n, d) with sign from y
        v_abs = jax.nn.softplus(z_v)
        v = sign_v * v_abs
        logJ_v = jnp.sum(jax.nn.log_sigmoid(z_v))

        # Construct ζ_i = A^T u^{(i)} + v_i
        # In code: ζ = u @ A + v where u @ A gives (n,m) @ (m,d) = (n,d)
        # Row i of (u @ A) is u^{(i)} @ A = (A^T u^{(i)})^T, which when
        # transposed back gives the d-vector A^T u^{(i)}.
        zeta = u @ A + v  # (n, d)

        # Likelihood: ζ_i ~ N(x_i^T β, I_d)
        mean = X @ beta  # (n, d)
        quad_zeta = jnp.sum((zeta - mean) ** 2)
        log_lik_zeta = -0.5 * quad_zeta

        # Prior for β: i.i.d. N(0, τ_β)
        quad_beta = jnp.sum(beta ** 2)
        log_prior_beta = -0.5 * quad_beta / tau_beta

        # Prior for active u: i.i.d. Exp(1)
        log_prior_u = -jnp.sum(u_active)

        # Combine
        log_post = (
            log_lik_zeta
            + log_prior_beta
            + log_prior_u
            + logJ_u
            + logJ_v
        )
        return -log_post  # potential = -log posterior

    return potential_fn


def make_potential_fn_marginal(struct: Dict):
    """
    Build potential function for marginal target with V integrated out.

    This is the DROP-C SURROGATE with V marginalized analytically.

    Unconstrained parameters:
    - β : (p, d)
    - z_u : (L,)

    Target:
    log p(β, z_u) ∝ 
      -½||β||²/τ_β                           [prior on β]
      - Σ_active softplus(z_u)                [exp(1) prior on u]
      + Σ_active log_sigmoid(z_u)             [Jacobian for u transform]
      + Σ_i Σ_j log Φ(s_{ij} (μ_{ij} - (u@A)_{ij}))  [marginal likelihood]

    where:
      s_{ij} = 2y_{ij} - 1 ∈ {-1, +1}
      μ_i = x_i^T β
      Φ = standard normal CDF

    Derivation: V marginalization gives product of truncated normal CDFs.

    Parameters
    ----------
    struct : dict from build_struct

    Returns
    -------
    potential_fn : callable
    """
    y = struct["y"]
    X = struct["X"]
    A = struct["A"]
    idx_u_i = struct["idx_u_i"]
    idx_u_k = struct["idx_u_k"]
    n, d, m = struct["n"], struct["d"], struct["m"]
    tau_beta = struct["tau_beta"]

    # Precompute sign matrix: s_{ij} = 2y_{ij} - 1
    sign_matrix = 2.0 * y - 1.0  # (n, d) in {-1, +1}

    def potential_fn(params):
        beta = params["beta"]   # (p, d)
        z_u = params["z_u"]     # (L,)

        # Transform z_u → u (n, m)
        u_active = jax.nn.softplus(z_u)
        logJ_u = jnp.sum(jax.nn.log_sigmoid(z_u))

        u = jnp.zeros((n, m))
        u = u.at[idx_u_i, idx_u_k].set(u_active)

        # Mean: μ_i = x_i^T β
        mean = X @ beta  # (n, d)

        # u @ A gives the dual contribution
        u_A = u @ A  # (n, d)

        # Argument to Φ: s_{ij} * (μ_{ij} - (u@A)_{ij})
        phi_arg = sign_matrix * (mean - u_A)  # (n, d)

        # log Φ(z) using JAX's numerically stable log_ndtr
        log_phi = jax.scipy.special.log_ndtr(phi_arg)
        log_lik_marginal = jnp.sum(log_phi)

        # Prior for β
        quad_beta = jnp.sum(beta ** 2)
        log_prior_beta = -0.5 * quad_beta / tau_beta

        # Prior for active u
        log_prior_u = -jnp.sum(u_active)

        # Combine
        log_post = (
            log_lik_marginal
            + log_prior_beta
            + log_prior_u
            + logJ_u
        )
        return -log_post

    return potential_fn




def make_model_marginal(struct: Dict):
    """
    Build NumPyro model function equivalent to make_potential_fn_marginal.

    Samples (β, z_u) in unconstrained space with:
    - β ~ N(0, τ_β I)
    - z_u ~ ImproperUniform (via manual factor)
    - u_active = softplus(z_u) with Exp(1) prior
    - Explicit Jacobian for softplus transform

    Parameters
    ----------
    struct : dict from build_struct

    Returns
    -------
    model : callable
        NumPyro model function
    """
    y = struct["y"]
    X = struct["X"]
    A = struct["A"]
    idx_u_i = struct["idx_u_i"]
    idx_u_k = struct["idx_u_k"]
    n, d, m = struct["n"], struct["d"], struct["m"]
    p = struct.get("p", X.shape[1])
    tau_beta = struct["tau_beta"]
    L = len(idx_u_i)

    sign_matrix = 2.0 * y - 1.0

    def model():
        # Prior for β: N(0, τ_β * I)
        # Contributes: -1/(2τ_β) ||β||_F^2 + const
        beta = numpyro.sample(
            "beta",
            dist.Normal(0, jnp.sqrt(tau_beta)).expand([p, d]).to_event(2)
        )

        # Sample z_u in unconstrained space
        # Use Normal(0, large_std) as approximate improper prior
        # The std is large enough that the quadratic penalty is negligible
        # BUT we subtract it out explicitly to get exact equivalence!
        large_std = 1000.0
        z_u = numpyro.sample(
            "z_u",
            dist.Normal(0, large_std).expand([L]).to_event(1)
        )
        
        # Cancel out the Normal prior to simulate improper uniform
        # Normal contributes: -1/(2*std^2) * ||z_u||^2
        # We add this back to cancel it:
        numpyro.factor("cancel_z_u_prior", 0.5 * jnp.sum(z_u ** 2) / (large_std ** 2))

        # Transform: u_active = softplus(z_u)
        u_active = jax.nn.softplus(z_u)

        # Jacobian: d(softplus)/dz = sigmoid(z)
        # log|Jacobian| = log(sigmoid(z)) = log_sigmoid(z)
        # Contributes: Σ log σ(z_u)
        numpyro.factor("jacobian_u", jnp.sum(jax.nn.log_sigmoid(z_u)))

        # Prior on u_active: Exp(1)
        # log p(u) = -u (up to const)
        # Contributes: -Σ softplus(z_u)
        numpyro.factor("prior_u", -jnp.sum(u_active))

        # Construct full u matrix (n, m)
        u = jnp.zeros((n, m))
        u = u.at[idx_u_i, idx_u_k].set(u_active)

        # Mean: μ = Xβ
        mean = X @ beta  # (n, d)

        # Dual contribution: uA
        u_A = u @ A  # (n, d)

        # Marginal likelihood: Σ log Φ(s_ij * (μ_ij - (uA)_ij))
        phi_arg = sign_matrix * (mean - u_A)
        log_phi = jax.scipy.special.log_ndtr(phi_arg)
        numpyro.factor("marginal_likelihood", jnp.sum(log_phi))

    return model


def make_potential_fn_beta_given_u(struct: Dict, u_fixed: jnp.ndarray):
    """
    Build potential function for β | u (used in Gibbs sampler).

    Target: p(β | u, data) ∝ p(β) × p(data | β, u)

    Parameters
    ----------
    struct : dict from build_struct
    u_fixed : (n, m) array
        Fixed dual variables.

    Returns
    -------
    potential_fn : callable taking params["beta"] only
    """
    y = struct["y"]
    X = struct["X"]
    A = struct["A"]
    tau_beta = struct["tau_beta"]

    sign_matrix = 2.0 * y - 1.0
    u_A_fixed = u_fixed @ A  # (n, d)

    def potential_fn(params):
        beta = params["beta"]  # (p, d)

        mean = X @ beta  # (n, d)
        phi_arg = sign_matrix * (mean - u_A_fixed)
        log_phi = jax.scipy.special.log_ndtr(phi_arg)
        log_lik = jnp.sum(log_phi)

        quad_beta = jnp.sum(beta ** 2)
        log_prior_beta = -0.5 * quad_beta / tau_beta

        log_post = log_lik + log_prior_beta
        return -log_post

    return potential_fn


def make_potential_fn_u_given_beta(struct: Dict, beta_fixed: jnp.ndarray):
    """
    Build potential function for u | β (used in Gibbs sampler).

    Target: p(u | β, data) ∝ p(u) × p(data | β, u)

    Parameters
    ----------
    struct : dict from build_struct
    beta_fixed : (p, d) array
        Fixed β.

    Returns
    -------
    potential_fn : callable taking params["z_u"] only
    """
    y = struct["y"]
    X = struct["X"]
    A = struct["A"]
    idx_u_i = struct["idx_u_i"]
    idx_u_k = struct["idx_u_k"]
    n, m = struct["n"], struct["m"]

    sign_matrix = 2.0 * y - 1.0
    mean_fixed = X @ beta_fixed  # (n, d)

    def potential_fn(params):
        z_u = params["z_u"]  # (L,)

        u_active = jax.nn.softplus(z_u)
        logJ_u = jnp.sum(jax.nn.log_sigmoid(z_u))

        u = jnp.zeros((n, m))
        u = u.at[idx_u_i, idx_u_k].set(u_active)

        u_A = u @ A  # (n, d)
        phi_arg = sign_matrix * (mean_fixed - u_A)
        log_phi = jax.scipy.special.log_ndtr(phi_arg)
        log_lik = jnp.sum(log_phi)

        log_prior_u = -jnp.sum(u_active)

        log_post = log_lik + log_prior_u + logJ_u
        return -log_post

    return potential_fn


# ============================================================
# SECTION 5: INITIALIZATION
# ============================================================

def init_params_baseline(
    struct: Dict,
    rng_key: jax.random.PRNGKey,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
    scale_v: float = 1.0,
) -> Dict:
    """
    Initialize unconstrained parameters for baseline algorithm.

    Note: Initialization scales are for numerical stability during warmup,
    not tied to prior variances.

    Parameters
    ----------
    struct : dict from build_struct
    rng_key : JAX PRNGKey
    scale_beta, scale_u, scale_v : float
        Initialization scales.

    Returns
    -------
    params : dict with keys "beta", "z_u", "z_v"
    """
    n, d, p = struct["n"], struct["d"], struct["p"]
    L = struct["num_u_active"]

    key_beta, key_u, key_v = jax.random.split(rng_key, 3)

    beta0 = scale_beta * jax.random.normal(key_beta, shape=(p, d))
    z_u0 = scale_u * jax.random.normal(key_u, shape=(L,))
    z_v0 = scale_v * jax.random.normal(key_v, shape=(n, d))

    return {"beta": beta0, "z_u": z_u0, "z_v": z_v0}


def init_params_marginal(
    struct: Dict,
    rng_key: jax.random.PRNGKey,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
) -> Dict:
    """
    Initialize unconstrained parameters for marginal algorithms.

    Parameters
    ----------
    struct : dict from build_struct
    rng_key : JAX PRNGKey
    scale_beta, scale_u : float
        Initialization scales.

    Returns
    -------
    params : dict with keys "beta", "z_u"
    """
    d, p = struct["d"], struct["p"]
    L = struct["num_u_active"]

    key_beta, key_u = jax.random.split(rng_key, 2)

    beta0 = scale_beta * jax.random.normal(key_beta, shape=(p, d))
    z_u0 = scale_u * jax.random.normal(key_u, shape=(L,))

    return {"beta": beta0, "z_u": z_u0}


# ============================================================
# SECTION 6: SAMPLERS
# ============================================================

def run_nuts_baseline(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float = 10.0,
    mcmc_config: Optional[MCMCConfig] = None,
    rng_seed: int = 0,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
    scale_v: float = 1.0,
) -> Tuple[MCMC, Dict]:
    """
    Run baseline NUTS on (β, U, V) with drop-c surrogate.

    This is the full augmentation algorithm with no marginalization.

    Parameters
    ----------
    y, X, A, b : data and constraints
    tau_beta : float
        Prior variance for β entries.
    mcmc_config : MCMCConfig or None
        MCMC settings. Uses default if None.
    rng_seed : int
        Random seed.
    scale_beta, scale_u, scale_v : float
        Initialization scales.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        Fitted MCMC object.
    samples : dict
        Dictionary with keys "beta", "u", "v", "zeta".
        Samples have shape (num_samples, ...) with chains flattened.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    numpyro.set_host_device_count(mcmc_config.num_chains)

    # Build model structure
    struct = build_struct(y, X, A, b, tau_beta=tau_beta)
    potential_fn = make_potential_fn_baseline(struct)

    # Initialize
    key = jax.random.PRNGKey(rng_seed)
    init_key, mcmc_key = jax.random.split(key)

    if mcmc_config.num_chains == 1:
        init_params_dict = init_params_baseline(
            struct, init_key, scale_beta, scale_u, scale_v
        )
    else:
        # Per-chain initialization
        init_keys = jax.random.split(init_key, mcmc_config.num_chains)

        def _init_for_chain(k):
            return init_params_baseline(struct, k, scale_beta, scale_u, scale_v)

        init_params_dict = jax.vmap(_init_for_chain)(init_keys)

    # NUTS kernel
    kernel = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=mcmc_config.target_accept_prob,
        dense_mass=mcmc_config.dense_mass,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_config.num_warmup,
        num_samples=mcmc_config.num_samples,
        num_chains=mcmc_config.num_chains,
        progress_bar=True,
    )

    mcmc.run(mcmc_key, init_params=init_params_dict)

    # Get samples (flattened over chains)
    unconstrained_samples = mcmc.get_samples(group_by_chain=False)

    if mcmc_config.only_save_beta_samples:
        return mcmc, {"beta": unconstrained_samples["beta"]}
    else:
        beta_samps = unconstrained_samples["beta"]  # (C, S, p, d)
        z_u_samps = unconstrained_samples["z_u"]    # (C, S, L)
        z_v_samps = unconstrained_samples["z_v"]    # (C, S, n, d)

        # Transform to constrained space
        idx_u_i = struct["idx_u_i"]
        idx_u_k = struct["idx_u_k"]
        n, d, m = struct["n"], struct["d"], struct["m"]
        sign_v = struct["sign_v"]
        A_mat = struct["A"]

        def map_sample(z_u, z_v, beta):
            # u
            u_active = jax.nn.softplus(z_u)
            u = jnp.zeros((n, m))
            u = u.at[idx_u_i, idx_u_k].set(u_active)
            # v
            v_abs = jax.nn.softplus(z_v)
            v = sign_v * v_abs
            # zeta
            zeta = u @ A_mat + v
            return u, v, zeta

        map_sample_vv = jax.vmap(map_sample, in_axes=(0, 0, 0))
        u_samps, v_samps, zeta_samps = map_sample_vv(z_u_samps, z_v_samps, beta_samps)

        samples = dict(
            beta=beta_samps,
            u=u_samps,
            v=v_samps,
            zeta=zeta_samps,
        )

        return mcmc, samples

    


def run_nuts_marginal(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float = 10.0,
    mcmc_config: Optional[MCMCConfig] = None,
    rng_seed: int = 0,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
) -> Tuple[MCMC, Dict]:
    """
    Run marginal joint NUTS on (β, U) with V integrated out.

    This uses the drop-c surrogate with V analytically marginalized.

    Parameters
    ----------
    y, X, A, b : data and constraints
    tau_beta : float
        Prior variance for β entries.
    mcmc_config : MCMCConfig or None
    rng_seed : int
    scale_beta, scale_u : float
        Initialization scales.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
    samples : dict
        Keys: "beta", "u". No "v" or "zeta" (V is marginalized).
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    numpyro.set_host_device_count(mcmc_config.num_chains)

    struct = build_struct(y, X, A, b, tau_beta=tau_beta)
    potential_fn = make_potential_fn_marginal(struct)

    key = jax.random.PRNGKey(rng_seed)
    init_key, mcmc_key = jax.random.split(key)

    if mcmc_config.num_chains == 1:
        init_params_dict = init_params_marginal(struct, init_key, scale_beta, scale_u)
    else:
        init_keys = jax.random.split(init_key, mcmc_config.num_chains)

        def _init_for_chain(k):
            return init_params_marginal(struct, k, scale_beta, scale_u)

        init_params_dict = jax.vmap(_init_for_chain)(init_keys)

    kernel = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=mcmc_config.target_accept_prob,
        dense_mass=mcmc_config.dense_mass,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_config.num_warmup,
        num_samples=mcmc_config.num_samples,
        num_chains=mcmc_config.num_chains,
        progress_bar=True,
    )

    mcmc.run(mcmc_key, init_params=init_params_dict)

    unconstrained_samples = mcmc.get_samples(group_by_chain=False)
    
    # Only save β samples if requested to save memory
    if mcmc_config.only_save_beta_samples:
        return None, {"beta": unconstrained_samples["beta"]}
    else:
        beta_samps = unconstrained_samples["beta"]  # (C, S, p, d)
        z_u_samps = unconstrained_samples["z_u"]    # (C, S, L)

        # Transform u
        idx_u_i = struct["idx_u_i"]
        idx_u_k = struct["idx_u_k"]
        n, m = struct["n"], struct["m"]

        def map_u(z_u):
            u_active = jax.nn.softplus(z_u)
            u = jnp.zeros((n, m))
            u = u.at[idx_u_i, idx_u_k].set(u_active)
            return u

        u_samps = jax.vmap(map_u)(z_u_samps)

        samples = dict(
            beta=beta_samps,
            u=u_samps,
        )

        return mcmc, samples




def run_nuts_marginal_model_version(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float = 10.0,
    mcmc_config: Optional["MCMCConfig"] = None,
    rng_seed: int = 0,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
) -> Tuple[MCMC, Dict]:
    """
    Run marginal joint NUTS on (β, z_u) with V integrated out.
    
    Uses model function with same parameterization as potential_fn.
    """
    if mcmc_config is None:
        mcmc_config = MCMCConfig()

    numpyro.set_host_device_count(mcmc_config.num_chains)

    struct = build_struct(y, X, A, b, tau_beta=tau_beta)
    model = make_model_marginal(struct)

    key = jax.random.PRNGKey(rng_seed)
    init_key, mcmc_key = jax.random.split(key)

    # Initialize parameters
    if mcmc_config.num_chains == 1:
        init_params_dict = init_params_marginal(struct, init_key, scale_beta, scale_u)
    else:
        init_keys = jax.random.split(init_key, mcmc_config.num_chains)
        def _init_for_chain(k):
            return init_params_marginal(struct, k, scale_beta, scale_u)
        init_params_dict = jax.vmap(_init_for_chain)(init_keys)

    kernel = NUTS(
        model,
        target_accept_prob=mcmc_config.target_accept_prob,
        dense_mass=mcmc_config.dense_mass,
    )

    mcmc = MCMC(
        kernel,
        num_warmup=mcmc_config.num_warmup,
        num_samples=mcmc_config.num_samples,
        num_chains=mcmc_config.num_chains,
        progress_bar=True,
    )

    mcmc.run(mcmc_key, init_params=init_params_dict)

    # Get samples - both beta and z_u are sampled
    # Only return beta (z_u samples are discarded here, saving memory)
    unconstrained_samples = mcmc.get_samples(group_by_chain=False)

    # Only save β samples if requested to save memory
    if mcmc_config.only_save_beta_samples:
        return None, {"beta": unconstrained_samples["beta"]}
    else:
        beta_samps = unconstrained_samples["beta"]  # (C, S, p, d)
        z_u_samps = unconstrained_samples["z_u"]    # (C, S, L)

        # Transform u
        idx_u_i = struct["idx_u_i"]
        idx_u_k = struct["idx_u_k"]
        n, m = struct["n"], struct["m"]

        def map_u(z_u):
            u_active = jax.nn.softplus(z_u)
            u = jnp.zeros((n, m))
            u = u.at[idx_u_i, idx_u_k].set(u_active)
            return u

        u_samps = jax.vmap(map_u)(z_u_samps)

        samples = dict(
            beta=beta_samps,
            u=u_samps,
        )

        return mcmc, samples


def run_nuts_marginal_green(
    y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    tau_beta: float = 10.0,
    mcmc_config: Optional[MCMCConfig] = None,
    rng_seed: int = 0,
    scale_beta: float = 1.0,
    scale_u: float = 1.0,
) -> Dict:
    """
    Memory-efficient NUTS for marginal posterior, returning only beta samples.
    
    This function runs MCMC in chunks to avoid out-of-memory crashes when
    the nuisance parameter z_u is high-dimensional. Only beta samples are
    stored; z_u samples are discarded after each chunk.
    
    NOTE: This function only supports num_chains=1.
    
    Parameters
    ----------
    y : np.ndarray
        Binary response matrix of shape (n, d).
    X : np.ndarray
        Design matrix of shape (n, p).
    A : np.ndarray
        Constraint matrix of shape (m, d).
    b : np.ndarray
        Constraint vector of shape (m,).
    tau_beta : float
        Prior variance for beta entries.
    mcmc_config : MCMCConfig, optional
        MCMC configuration. If first_chunk_samples or chunk_size are not set,
        defaults to 500 and 2000 respectively.
    rng_seed : int
        Random seed for reproducibility.
    scale_beta : float
        Initialization scale for beta.
    scale_u : float
        Initialization scale for z_u.
        
    Returns
    -------
    samples : dict
        Dictionary with single key "beta" containing samples of shape 
        (num_samples, p, d).
        
    Raises
    ------
    ValueError
        If mcmc_config.num_chains != 1.
    """
    # Default configuration
    if mcmc_config is None:
        mcmc_config = MCMCConfig()
    
    # Enforce single chain
    if mcmc_config.num_chains != 1:
        raise ValueError("run_nuts_marginal_green only supports num_chains=1")
    
    # Set default chunking parameters
    first_chunk_samples = 500
    chunk_size = 2000
    
    # Build data structure and potential function
    struct = build_struct(y, X, A, b, tau_beta=tau_beta)
    potential_fn = make_potential_fn_marginal(struct)
    
    # Initialize RNG
    key = jax.random.PRNGKey(rng_seed)
    init_key, mcmc_key = jax.random.split(key)
    
    # Initialize parameters
    init_params = init_params_marginal(struct, init_key, scale_beta, scale_u)
    
    # Storage for beta samples
    beta_samples_list = []
    
    # =========================================================================
    # CHUNK 1: Warmup + Adaptation + Initial Samples
    # =========================================================================
    mcmc_key, chunk1_key = jax.random.split(mcmc_key)
    
    kernel_chunk1 = NUTS(
        potential_fn=potential_fn,
        target_accept_prob=mcmc_config.target_accept_prob,
        dense_mass=mcmc_config.dense_mass,
    )
    
    mcmc_chunk1 = MCMC(
        kernel_chunk1,
        num_warmup=mcmc_config.num_warmup,
        num_samples=first_chunk_samples,
        num_chains=1,
        progress_bar=True,
    )
    
    mcmc_chunk1.run(chunk1_key, init_params=init_params)
    
    # Extract adapted parameters
    last_state = mcmc_chunk1.last_state
    
    if hasattr(last_state, 'adapt_state') and last_state.adapt_state is not None:
        adapt_state = last_state.adapt_state
        step_size = float(adapt_state.step_size)
        inverse_mass_matrix = adapt_state.inverse_mass_matrix
    else:
        step_size = float(last_state.step_size) if hasattr(last_state, 'step_size') else 0.01
        inverse_mass_matrix = last_state.inverse_mass_matrix if hasattr(last_state, 'inverse_mass_matrix') else None
    
    # Extract samples from chunk 1
    chunk1_samples = mcmc_chunk1.get_samples(group_by_chain=False)
    beta_samples_list.append(np.array(chunk1_samples["beta"]))
    
    # Get last sample for continuation
    current_params = {
        "beta": chunk1_samples["beta"][-1],
        "z_u": chunk1_samples["z_u"][-1],
    }
    
    # Clean up chunk 1
    del chunk1_samples, mcmc_chunk1
    gc.collect()
    
    samples_collected = first_chunk_samples
    
    # =========================================================================
    # SUBSEQUENT CHUNKS: Fixed Step Size Sampling
    # =========================================================================
    remaining_samples = mcmc_config.num_samples - samples_collected
    
    while remaining_samples > 0:
        mcmc_key, chunk_key = jax.random.split(mcmc_key)
        
        samples_this_chunk = min(chunk_size, remaining_samples)
        
        kernel = NUTS(
            potential_fn=potential_fn,
            target_accept_prob=mcmc_config.target_accept_prob,
            dense_mass=mcmc_config.dense_mass,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=False,
            adapt_mass_matrix=False,
        )
        
        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=samples_this_chunk,
            num_chains=1,
            progress_bar=True,
        )
        
        mcmc.run(chunk_key, init_params=current_params)
        
        # Extract samples
        chunk_samples = mcmc.get_samples(group_by_chain=False)
        
        # Store only beta
        beta_samples_list.append(np.array(chunk_samples["beta"]))
        
        # Update current_params for next chunk
        current_params = {
            "beta": chunk_samples["beta"][-1],
            "z_u": chunk_samples["z_u"][-1],
        }
        
        samples_collected += samples_this_chunk
        remaining_samples -= samples_this_chunk
        
        # Clean up
        del chunk_samples, mcmc
        gc.collect()
    
    # =========================================================================
    # Finalize Results
    # =========================================================================
    beta_samples = np.concatenate(beta_samples_list, axis=0)
    
    # Ensure we have exactly the requested number of samples
    if len(beta_samples) > mcmc_config.num_samples:
        beta_samples = beta_samples[:mcmc_config.num_samples]
    
    return {"beta": beta_samples}




# ============================================================
# SECTION 7: DIAGNOSTICS
# ============================================================

def compute_acf_1d(x: np.ndarray, max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation function for 1D array."""
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    var = np.dot(x, x) / n
    if var == 0:
        return np.arange(max_lag + 1), np.ones(max_lag + 1)

    acf = np.empty(max_lag + 1, dtype=float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        cov = np.dot(x[:-lag], x[lag:]) / (n - lag)
        acf[lag] = cov / var
    lags = np.arange(max_lag + 1)
    return lags, acf


def compute_beta_diagnostics(
    samples: Dict,
    beta_true: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute comprehensive diagnostics for β samples.

    Parameters
    ----------
    samples : dict
        Must contain "beta" key with shape:
          - (S, p, d) → treated as one chain
          - (C, S, p, d) → C chains
    beta_true : (p, d) array or None
        True β for RMSE and coverage.

    Returns
    -------
    diagnostics : dict
        Includes mean, sd, ess, (rhat if C>=2), rmse, coverage_95, etc.
    """
    # Validate samples dict contains "beta" key
    if "beta" not in samples:
        raise KeyError("samples dict must contain 'beta' key")
    
    beta_samps = np.asarray(samples["beta"])
    
    # Validate numeric dtype
    if not np.issubdtype(beta_samps.dtype, np.number):
        raise TypeError(f"beta samples must be numeric, got dtype {beta_samps.dtype}")
    
    diagnostics = {"input_shape": beta_samps.shape}

    # Step 1: Standardize to (C, S, p, d)
    if beta_samps.ndim == 3:  # (S, p, d) → one chain
        S, p, d = beta_samps.shape
        beta_chains = beta_samps[None, :, :, :]  # (1, S, p, d)
        C = 1
    elif beta_samps.ndim == 4:  # (C, S, p, d)
        C, S, p, d = beta_samps.shape
        beta_chains = beta_samps
    else:
        raise ValueError(f"Expected beta shape (S, p, d) or (C, S, p, d), got {beta_samps.shape}")

    # Validate dimensions are positive
    if S == 0:
        raise ValueError("Number of samples S must be positive")
    if p == 0 or d == 0:
        raise ValueError(f"Parameter dimensions must be positive, got p={p}, d={d}")

    # Flatten for marginal stats (mean, sd, coverage)
    beta_flat = beta_chains.reshape(-1, p, d)  # (C*S, p, d)
    beta_mean = beta_flat.mean(axis=0)
    beta_sd = beta_flat.std(axis=0, ddof=1)

    diagnostics.update({
        "mean": beta_mean,
        "sd": beta_sd,
        "p": p,
        "d": d,
        "num_chains": C,
        "draws_per_chain": S,
        "total_draws": C * S,
    })

    # Step 2: RMSE and coverage (if beta_true provided)
    if beta_true is not None:
        beta_true = np.asarray(beta_true)
        if beta_true.shape != (p, d):
            raise ValueError(f"beta_true shape {beta_true.shape} != ({p}, {d})")
        
        rmse = np.sqrt(np.mean((beta_mean - beta_true) ** 2))
        diagnostics["rmse"] = float(rmse)

        lower = np.percentile(beta_flat, 2.5, axis=0)
        upper = np.percentile(beta_flat, 97.5, axis=0)
        coverage = np.mean((beta_true >= lower) & (beta_true <= upper))
        diagnostics["coverage_95"] = float(coverage)

    # Step 3: ESS (always computable, even with 1 chain)
    # Convert to JAX array once for both ESS and R-hat
    beta_jax = None
    try:
        beta_jax = jnp.asarray(beta_chains)  # (C, S, p, d)
        ess = effective_sample_size(beta_jax)  # (p, d)
        diagnostics["ess"] = np.array(ess)
        diagnostics["ess_min"] = float(np.min(ess))
        diagnostics["ess_median"] = float(np.median(ess))
    except Exception as e:
        diagnostics["ess_error"] = str(e)

    # Step 4: R-hat (only if C >= 2)
    if C >= 2:
        try:
            # Use existing beta_jax if available, otherwise convert again
            if beta_jax is None:
                beta_jax = jnp.asarray(beta_chains)
            rhat = split_gelman_rubin(beta_jax)  # (p, d)
            diagnostics["rhat"] = np.array(rhat)
            diagnostics["rhat_max"] = float(np.max(rhat))
        except Exception as e:
            diagnostics["rhat_error"] = str(e)
    
    return diagnostics


def plot_beta_diagnostics(
    samples: Dict,
    diagnostics: Dict,
    beta_true: Optional[np.ndarray] = None,
    max_plots: int = 4,
    max_lag: int = 50,
    save_path: Optional[str] = None,
) -> None:
    """
    Create diagnostic plots for β.

    Generates:
    - Trace plots
    - ACF plots
    - Posterior vs true comparison (if beta_true provided)

    Parameters
    ----------
    samples : dict
    diagnostics : dict from compute_beta_diagnostics
    beta_true : (p, d) array or None
    max_plots : int
    max_lag : int
    save_path : str or None
        If provided, saves to PDF instead of showing.
    """
    beta_samps = np.array(samples["beta"])
    if beta_samps.ndim == 4:  # (C, S, p, d)
        beta_samps = beta_samps.reshape(-1, beta_samps.shape[2], beta_samps.shape[3])
    
    S, p, d = beta_samps.shape
    
    # Select coordinates to plot
    idx_list = []
    for j in range(p):
        for k in range(d):
            idx_list.append((j, k))
            if len(idx_list) >= max_plots:
                break
        if len(idx_list) >= max_plots:
            break
    
    figures = []
    
    # Trace plots
    for (j, k) in idx_list:
        fig, ax = plt.subplots(figsize=(8, 3))
        series = beta_samps[:, j, k]
        ax.plot(series, lw=0.8, alpha=0.8)
        if beta_true is not None:
            ax.axhline(beta_true[j, k], color='red', linestyle='--', 
                      label=f'True: {beta_true[j, k]:.3f}')
            ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel(f"β[{j},{k}]")
        ax.set_title(f"Trace plot: β[{j},{k}]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        figures.append(fig)
    
    # ACF plots
    conf = 1.96 / np.sqrt(S)
    for (j, k) in idx_list:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        series = beta_samps[:, j, k]
        lags, acf = compute_acf_1d(series, max_lag=max_lag)
        
        markerline, stemlines, baseline = ax.stem(lags, acf, basefmt=" ")
        plt.setp(stemlines, linewidth=1.0)
        plt.setp(markerline, markersize=4)
        
        ax.hlines([conf, -conf], xmin=0, xmax=max_lag,
                 colors="red", linestyles="--", linewidth=1.0,
                 label="95% CI")
        
        ax.set_title(f"ACF: β[{j},{k}]")
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        figures.append(fig)
    
    # Posterior mean vs true (heatmap if available)
    if beta_true is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        
        beta_mean = diagnostics["mean"]
        diff = beta_mean - beta_true
        
        im0 = axes[0].imshow(beta_mean, cmap="RdBu_r", aspect="auto")
        axes[0].set_title("Posterior Mean")
        axes[0].set_xlabel("d")
        axes[0].set_ylabel("p")
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(beta_true, cmap="RdBu_r", aspect="auto")
        axes[1].set_title("True β")
        axes[1].set_xlabel("d")
        axes[1].set_ylabel("p")
        plt.colorbar(im1, ax=axes[1])
        
        im2 = axes[2].imshow(diff, cmap="coolwarm", aspect="auto")
        axes[2].set_title("Difference (Mean - True)")
        axes[2].set_xlabel("d")
        axes[2].set_ylabel("p")
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        figures.append(fig)
    
    # Save or show
    if save_path:
        with PdfPages(save_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Saved diagnostic plots to {save_path}")
    else:
        for fig in figures:
            plt.show()


def check_mcmc_health(
    mcmc: Optional[MCMC],
    diagnostics: Dict,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Check MCMC health and return warnings.

    Parameters
    ----------
    mcmc : MCMC object or None
    diagnostics : dict from compute_beta_diagnostics
    verbose : bool
        If True, print warnings.

    Returns
    -------
    warnings : dict
        Keys are warning types, values are messages.
    """
    warnings_dict = {}
    
    # Check R-hat
    if "rhat" in diagnostics:
        rhat = diagnostics["rhat"]
        if np.any(rhat > 1.1):
            msg = f"R-hat > 1.1 detected (max: {rhat.max():.3f}). Chains may not have converged."
            warnings_dict["rhat"] = msg
    
    # Check ESS
    if "ess" in diagnostics:
        ess = diagnostics["ess"]
        ess_min = ess.min()
        ess_median = np.median(ess)
        
        if ess_min < 100:
            msg = f"Min ESS = {ess_min:.1f} < 100. Very low effective sample size."
            warnings_dict["ess_min"] = msg
        
        if ess_median < 500:
            msg = f"Median ESS = {ess_median:.1f} < 500. Low effective sample size."
            warnings_dict["ess_median"] = msg
    
    # Check divergences
    if mcmc is not None:
        try:
            extra = mcmc.get_extra_fields(group_by_chain=True)
            if "diverging" in extra:
                div = np.array(extra["diverging"])
                frac_div = div.mean()
                if frac_div > 0.01:
                    msg = f"Divergent transitions: {frac_div:.2%}. Consider increasing num_warmup or target_accept_prob."
                    warnings_dict["divergences"] = msg
        except:
            pass
    
    if verbose and warnings_dict:
        print("\n" + "="*60)
        print("MCMC HEALTH WARNINGS")
        print("="*60)
        for key, msg in warnings_dict.items():
            print(f"[{key.upper()}] {msg}")
        print("="*60 + "\n")
    
    return warnings_dict


def compute_efficiency_metrics(
    diagnostics: Dict,
    runtime_seconds: float,
    mcmc: Optional[MCMC] = None,
) -> Dict:
    """
    Compute efficiency metrics from diagnostics and runtime.
    
    This function computes derived efficiency quantities that help
    compare algorithms fairly. All computations are based on already-
    computed diagnostics; no inference behavior is changed.
    
    Parameters
    ----------
    diagnostics : dict
        Output from compute_beta_diagnostics, must contain ESS if available.
    runtime_seconds : float
        Wall-clock time for MCMC sampling (excludes data generation).
    mcmc : MCMC object or None
        If provided, extracts additional info (acceptance rate, divergences).
    
    Returns
    -------
    efficiency : dict with keys:
        - runtime_seconds: float, total MCMC runtime
        - ess_per_second_min: float or None, min(ESS) / runtime
        - ess_per_second_median: float or None, median(ESS) / runtime
        - ess_per_second_mean: float or None, mean(ESS) / runtime  
        - seconds_per_ess_median: float or None, runtime / median(ESS)
        - mean_accept_prob: float or None, mean acceptance probability
        - num_divergences: int or None, total divergent transitions
        - frac_divergences: float or None, fraction of divergent transitions
    """
    efficiency = {
        "runtime_seconds": runtime_seconds,
        "ess_per_second_min": None,
        "ess_per_second_median": None,
        "ess_per_second_mean": None,
        "seconds_per_ess_median": None,
        "mean_accept_prob": None,
        "num_divergences": None,
        "frac_divergences": None,
    }
    
    # ESS-based efficiency (only if ESS was computed)
    if "ess" in diagnostics and runtime_seconds > 0:
        ess = diagnostics["ess"]
        ess_min = float(np.min(ess))
        ess_median = float(np.median(ess))
        ess_mean = float(np.mean(ess))
        
        efficiency["ess_per_second_min"] = ess_min / runtime_seconds
        efficiency["ess_per_second_median"] = ess_median / runtime_seconds
        efficiency["ess_per_second_mean"] = ess_mean / runtime_seconds
        
        if ess_median > 0:
            efficiency["seconds_per_ess_median"] = runtime_seconds / ess_median
    
    # MCMC-specific diagnostics (acceptance rate, divergences)
    if mcmc is not None:
        try:
            extra = mcmc.get_extra_fields(group_by_chain=True)
            
            # Acceptance probability
            if "accept_prob" in extra:
                accept_prob = np.array(extra["accept_prob"])
                efficiency["mean_accept_prob"] = float(np.mean(accept_prob))
            
            # Divergences
            if "diverging" in extra:
                div = np.array(extra["diverging"])
                efficiency["num_divergences"] = int(np.sum(div))
                efficiency["frac_divergences"] = float(np.mean(div))
        except Exception:
            # Silently skip if extra fields not available
            pass
    
    return efficiency



# ============================================================
# SECTION 8: EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    algorithm: Literal["baseline", "marginal_joint", "marginal_gibbs"],
    n: int,
    p: int,
    d: int,
    m: int,
    tau_beta: float = 1.0,
    mcmc_config: Optional[MCMCConfig] = None,
    diagnostic_config: Optional[DiagnosticConfig] = None,
    data_seed: int = 123,
    mcmc_seed: int = 456,
    use_simulated_data: bool = True,
    save_results: bool = True,
    run_diagnostics: bool = True,
    output_dir: Optional[str] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Run a complete experiment with data generation, MCMC, and diagnostics.

    This is the main user-facing entry point.

    Parameters
    ----------
    algorithm : str
        One of: "baseline", "marginal_joint", "marginal_gibbs".
    n, p, d, m : int
        Problem dimensions.
    tau_beta : float
        Prior variance for β.
    mcmc_config : MCMCConfig or None
    diagnostic_config : DiagnosticConfig or None
    data_seed : int
        Seed for data generation.
    mcmc_seed : int
        Seed for MCMC sampling.
    use_simulated_data : bool
        If True, generate data. If False, would load from files (not implemented).
    save_results : bool
        If True, save samples and diagnostics to disk.
    run_diagnostics : bool
        If True, compute and plot diagnostics.
    output_dir : str or None
        Directory for outputs. Uses DEFAULT_SIM_OUT_DIR if None.

    Returns
    -------
    mcmc_or_info : MCMC object or dict
        For baseline/marginal_joint: MCMC object.
        For marginal_gibbs: dict with metadata.
    samples : dict
        Posterior samples.
    diagnostics : dict
        Diagnostic results.
    """
    # Resolve defaults
    if mcmc_config is None:
        mcmc_config = MCMCConfig()
    if diagnostic_config is None:
        diagnostic_config = DiagnosticConfig()
    if output_dir is None:
        output_dir = DEFAULT_SIM_OUT_DIR

    if mcmc_config.green_mode and algorithm == "marginal_joint":
        algorithm = "marginal_joint_green"
        print("Due to green_mode=True in MCMCConfig, forcing algorithm to marginal_joint_green for stability (Only beta samples returned)")

    
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build the run stub for consistent naming
    stub = build_run_stub(
        algorithm=algorithm,
        n=n, p=p, d=d, m=m,
        tau_beta=tau_beta,
        mcmc_config=mcmc_config,
        data_seed=data_seed,
        mcmc_seed=mcmc_seed,
    )
    
    # Get all artifact paths
    artifact_paths = get_artifact_paths(stub, output_dir)
    
    print("="*70)
    print(f"EXPERIMENT: {algorithm.upper()}")
    print(f"Dimensions: n={n}, p={p}, d={d}, m={m}")
    print(f"tau_beta={tau_beta}, data_seed={data_seed}, mcmc_seed={mcmc_seed}")
    print(f"Run stub: {stub}")
    print("="*70)
    
    # Generate or load data
    if use_simulated_data:
        print("\nGenerating simulation data...")
        y, X, A, b, beta_true, zeta_true = generate_simulation_data(
            n=n, p=p, d=d, m=m,
            tau_beta=tau_beta,
            rng_seed=data_seed,
            random_b=True,
        )
        print(f"Data generated: y {y.shape}, X {X.shape}, A {A.shape}, b {b.shape}")
    else:
        raise NotImplementedError("Loading data from files not implemented.")
    
    # Build struct to get num_u_active for metadata
    struct = build_struct(y, X, A, b, tau_beta=tau_beta)
    num_u_active = int(struct["num_u_active"])
    print(f"Active constraint positions: L = {num_u_active}")
    
    # Run MCMC
    print(f"\nRunning {algorithm} algorithm...")
    t0 = time.time()

    if algorithm == "marginal_joint_green":
        print("To turn off green mode, set MCMCConfig.green_mode=False")
        samples = run_nuts_marginal_green(
            y, X, A, b,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            rng_seed=mcmc_seed,
        )
        mcmc = None
        mcmc_or_info = None
    
    elif algorithm == "baseline":
        mcmc, samples = run_nuts_baseline(
            y, X, A, b,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            rng_seed=mcmc_seed,
        )
        mcmc_or_info = mcmc
        
    elif algorithm == "marginal_joint":
        mcmc, samples = run_nuts_marginal(
            y, X, A, b,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            rng_seed=mcmc_seed,
        )
        mcmc_or_info = mcmc
    
    elif algorithm == "marginal_joint_model_version":
        mcmc, samples = run_nuts_marginal_model_version(
            y, X, A, b,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            rng_seed=mcmc_seed,
        )
        mcmc_or_info = mcmc
        
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    elapsed = time.time() - t0
    print(f"MCMC completed in {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
    
    # Compute diagnostics
    diagnostics = {}
    efficiency_metrics = {}
    if run_diagnostics:
        print("\nComputing diagnostics...")
        diagnostics = compute_beta_diagnostics(
            samples,
            beta_true=beta_true,
        )
        
        # Compute efficiency metrics
        efficiency_metrics = compute_efficiency_metrics(
            diagnostics=diagnostics,
            runtime_seconds=elapsed,
            mcmc=mcmc,
        )
        
        print("\n" + "="*60)
        print("BETA DIAGNOSTICS SUMMARY")
        print("="*60)
        if "rmse" in diagnostics:
            print(f"RMSE: {diagnostics['rmse']:.4f}")
        if "coverage_95" in diagnostics:
            print(f"95% CI Coverage: {diagnostics['coverage_95']:.3f}")
        if "ess_min" in diagnostics:
            print(f"ESS: min={diagnostics['ess_min']:.1f}, median={diagnostics['ess_median']:.1f}")
        if "rhat_max" in diagnostics:
            print(f"R-hat: max={diagnostics['rhat_max']:.3f}")
        print("="*60)
        
        # Print efficiency summary
        print("\n" + "="*60)
        print("EFFICIENCY SUMMARY")
        print("="*60)
        print(f"Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
        if efficiency_metrics.get("ess_per_second_median") is not None:
            print(f"ESS/second (median): {efficiency_metrics['ess_per_second_median']:.2f}")
            print(f"ESS/second (min): {efficiency_metrics['ess_per_second_min']:.2f}")
            print(f"Seconds per ESS (median): {efficiency_metrics['seconds_per_ess_median']:.3f}")
        if efficiency_metrics.get("mean_accept_prob") is not None:
            print(f"Mean acceptance prob: {efficiency_metrics['mean_accept_prob']:.3f}")
        if efficiency_metrics.get("num_divergences") is not None:
            print(f"Divergences: {efficiency_metrics['num_divergences']} ({efficiency_metrics['frac_divergences']:.2%})")
        print("="*60 + "\n")
        
        # Health check
        warnings_dict = check_mcmc_health(mcmc, diagnostics, verbose=True)
        diagnostics["warnings"] = warnings_dict
        
        # Store efficiency metrics in diagnostics for return
        diagnostics["efficiency"] = efficiency_metrics
        
        # Plots - use the stub-based path
        if diagnostic_config.max_plots > 0:
            plot_beta_diagnostics(
                samples,
                diagnostics,
                beta_true=beta_true,
                max_plots=diagnostic_config.max_plots,
                max_lag=diagnostic_config.max_lag,
                save_path=artifact_paths["diagnostics_pdf"],
            )
    
    # Save results
    if save_results:
        print("\nSaving results...")
        
        # Collect shapes for metadata
        shapes = {
            "y": y.shape,
            "X": X.shape,
            "A": A.shape,
            "b": b.shape,
            "beta_true": beta_true.shape,
            "beta_samples": np.array(samples["beta"]).shape,
        }
        if "u" in samples:
            shapes["u_samples"] = np.array(samples["u"]).shape
        if "v" in samples:
            shapes["v_samples"] = np.array(samples["v"]).shape
        if "zeta" in samples:
            shapes["zeta_samples"] = np.array(samples["zeta"]).shape
        
        # Save samples
        save_samples_npz(
            artifact_paths["samples"],
            samples,
            beta_true=beta_true,
        )
        print(f"Saved samples to {artifact_paths['samples']}")
        
        # Save metadata
        save_meta_json(
            artifact_paths["meta"],
            algorithm=algorithm,
            n=n, p=p, d=d, m=m,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            diagnostic_config=diagnostic_config,
            data_seed=data_seed,
            mcmc_seed=mcmc_seed,
            shapes=shapes,
            num_u_active=num_u_active,
        )
        print(f"Saved metadata to {artifact_paths['meta']}")
        
        # Save summary
        save_summary_json(
            artifact_paths["summary"],
            stub=stub,
            algorithm=algorithm,
            n=n, p=p, d=d, m=m,
            tau_beta=tau_beta,
            mcmc_config=mcmc_config,
            data_seed=data_seed,
            mcmc_seed=mcmc_seed,
            artifact_paths=artifact_paths,
            runtime_seconds=elapsed,
            efficiency_metrics=efficiency_metrics if efficiency_metrics else None,
            diagnostics=diagnostics if diagnostics else None,
        )
        print(f"Saved summary to {artifact_paths['summary']}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print(f"Run stub: {stub}")
    print("="*70 + "\n")
    
    return mcmc_or_info, samples, diagnostics


# ============================================================
# SECTION 9: MAIN EXECUTION (demo)
# ============================================================

if __name__ == "__main__":
    # Example usage: Run all three algorithms on a moderate problem
    
    print("\n" + "="*70)
    print("DEMO: Comparing Three Algorithms")
    print("="*70 + "\n")
    
    # Problem setup
    n, p, d, m = 1000, 5, 10, 5
    tau_beta = 1.0
    data_seed = 123
    
    # MCMC configuration
    mcmc_config = MCMCConfig(
        num_warmup=1000,
        num_samples=1500,
        num_chains=1,
        target_accept_prob=0.75,
    )
    
    
    diagnostic_config = DiagnosticConfig(
        max_plots=4,
        max_lag=50,
    )
    
    # Run algorithm
    algorithms = ["marginal_joint"]
    results = {}
    
    for alg in algorithms:
        print(f"\n{'='*70}")
        print(f"Running {alg.upper()}")
        print(f"{'='*70}\n")
        
        try:
            mcmc, samples, diagnostics = run_experiment(
                algorithm=alg,
                n=n, p=p, d=d, m=m,
                tau_beta=tau_beta,
                mcmc_config=mcmc_config,
                diagnostic_config=diagnostic_config,
                data_seed=data_seed,
                mcmc_seed=456,
                use_simulated_data=True,
                save_results=True,
                run_diagnostics=True,
                output_dir="./results/hmc/demo_results",
            )
            results[alg] = (mcmc, samples, diagnostics)
            print(f"\n✓ {alg.upper()} completed successfully\n")
        except Exception as e:
            print(f"\n✗ {alg.upper()} failed: {e}\n")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Algorithm':<20} {'RMSE':<10} {'ESS (median)':<15} {'R-hat (max)':<12}")
    print("-"*70)
    
    for alg in algorithms:
        if alg in results:
            diag = results[alg][2]
            rmse = diag.get("rmse", np.nan)
            ess_med = diag.get("ess_median", np.nan)
            rhat_max = diag.get("rhat_max", np.nan)
            print(f"{alg:<20} {rmse:<10.4f} {ess_med:<15.1f} {rhat_max:<12.3f}")
        else:
            print(f"{alg:<20} {'FAILED':<10}")
    
    print("="*70 + "\n")