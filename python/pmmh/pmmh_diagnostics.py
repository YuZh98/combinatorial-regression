"""
Diagnostic tools for PMMH results.
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from typing import List, Tuple, Optional
from pmmh_core import PMMMHResult


# ============================================================================
# Summary Statistics
# ============================================================================

def compute_ess(result: PMMMHResult) -> np.ndarray:
    """
    Compute effective sample size (ESS) for each parameter using ArviZ.
    
    Uses full chain (including burn-in) for ESS calculation.
    
    Args:
        result: PMMH result object
        
    Returns:
        (p, d) array of ESS values
    """
    samples = result.samples  # Full chain
    p, d = samples.shape[1], samples.shape[2]
    
    ess_matrix = np.zeros((p, d))
    for j in range(p):
        for k in range(d):
            ess_matrix[j, k] = float(az.ess(samples[:, j, k]))
    
    return ess_matrix


def compute_summary_statistics(
    result: PMMMHResult,
    beta_true: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute posterior summary statistics.
    
    Args:
        result: PMMH result object
        beta_true: (p, d) true parameters (if available)
        
    Returns:
        Dictionary with summary statistics
    """
    samples_post = result.samples_post_burnin
    
    stats = {
        'posterior_mean': samples_post.mean(axis=0),
        'posterior_std': samples_post.std(axis=0),
        'posterior_median': np.median(samples_post, axis=0),
        'ess_matrix': compute_ess(result),
        'acceptance_rate': result.acceptance_rate,
        'n_effective_samples': result.n_effective_samples,
    }
    
    if beta_true is not None:
        beta_true = np.asarray(beta_true)
        diff = stats['posterior_mean'] - beta_true
        stats['rmse'] = float(np.sqrt(np.mean(diff ** 2)))
        stats['bias'] = diff
    
    return stats


def print_summary(result: PMMMHResult, beta_true: Optional[np.ndarray] = None):
    """Print formatted summary statistics."""
    stats = compute_summary_statistics(result, beta_true)
    
    print("=" * 70)
    print("PMMH Summary")
    print("=" * 70)
    print(f"Total iterations: {result.samples.shape[0]}")
    print(f"Burn-in: {result.config.burn_in}")
    print(f"Effective samples: {stats['n_effective_samples']}")
    print(f"Acceptance rate: {stats['acceptance_rate']:.3f}")
    print()
    
    print("Posterior mean:")
    print(stats['posterior_mean'])
    print()
    
    print("Posterior std:")
    print(stats['posterior_std'])
    print()
    
    print("ESS matrix:")
    print(stats['ess_matrix'])
    print()
    
    # Find smallest ESS
    ess_flat = stats['ess_matrix'].flatten()
    p, d = stats['ess_matrix'].shape
    idx_sorted = np.argsort(ess_flat)
    print("Smallest ESS entries:")
    for idx in idx_sorted[:min(5, len(idx_sorted))]:
        j = idx // d
        k = idx % d
        print(f"  ESS={ess_flat[idx]:.1f} at beta[{j},{k}]")
    print()
    
    if beta_true is not None:
        print(f"RMSE: {stats['rmse']:.4f}")
        print("Bias:")
        print(stats['bias'])
    
    print("=" * 70)


# ============================================================================
# Autocorrelation Function
# ============================================================================

def compute_acf(x: np.ndarray, max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation function.
    
    Args:
        x: 1D array
        max_lag: Maximum lag
        
    Returns:
        (lags, acf) where both are (max_lag+1,) arrays
    """
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    
    var = np.dot(x, x) / n
    if var == 0:
        return np.arange(max_lag + 1), np.ones(max_lag + 1)
    
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    
    for lag in range(1, max_lag + 1):
        cov = np.dot(x[:-lag], x[lag:]) / (n - lag)
        acf[lag] = cov / var
    
    return np.arange(max_lag + 1), acf


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_traces(
    result: PMMMHResult,
    beta_idx_list: Optional[List[Tuple[int, int]]] = None,
    max_plots: int = 4,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot trace plots for selected parameters.
    
    Args:
        result: PMMH result object
        beta_idx_list: List of (j, k) indices to plot
        max_plots: Maximum number of plots
        figsize: Figure size for each plot
    """
    samples = result.samples_post_burnin
    p, d = samples.shape[1], samples.shape[2]
    
    if beta_idx_list is None:
        beta_idx_list = [(j, k) for j in range(min(2, p)) for k in range(min(2, d))]
    
    for j, k in beta_idx_list[:max_plots]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(samples[:, j, k], lw=0.8, alpha=0.7)
        ax.set_xlabel("Iteration (post-burn-in)")
        ax.set_ylabel(f"beta[{j},{k}]")
        ax.set_title(f"Trace Plot: beta[{j},{k}]")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_acf(
    result: PMMMHResult,
    beta_idx_list: Optional[List[Tuple[int, int]]] = None,
    max_lag: int = 50,
    max_plots: int = 4,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot ACF for selected parameters."""
    samples = result.samples_post_burnin
    p, d = samples.shape[1], samples.shape[2]
    
    if beta_idx_list is None:
        beta_idx_list = [(j, k) for j in range(min(2, p)) for k in range(min(2, d))]
    
    for j, k in beta_idx_list[:max_plots]:
        lags, acf = compute_acf(samples[:, j, k], max_lag=max_lag)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.stem(lags, acf, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.set_title(f"Autocorrelation: beta[{j},{k}]")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_histograms(
    result: PMMMHResult,
    beta_true: Optional[np.ndarray] = None,
    beta_idx_list: Optional[List[Tuple[int, int]]] = None,
    max_plots: int = 4,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot posterior histograms with true values."""
    samples = result.samples_post_burnin
    p, d = samples.shape[1], samples.shape[2]
    
    if beta_idx_list is None:
        beta_idx_list = [(j, k) for j in range(min(2, p)) for k in range(min(2, d))]
    
    for j, k in beta_idx_list[:max_plots]:
        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(samples[:, j, k], bins=bins, density=True, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f"beta[{j},{k}]")
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior Distribution: beta[{j},{k}]")
        
        if beta_true is not None:
            ax.axvline(beta_true[j, k], color='red', linestyle='--', lw=2, label='True value')
            ax.legend()
        
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_loglik_trace(result: PMMMHResult, figsize: Tuple[int, int] = (12, 5)):
    """Plot log-likelihood trace."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result.loglik_trace, lw=0.8, alpha=0.7)
    ax.axvline(result.config.burn_in, color='red', linestyle='--', label='Burn-in')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("log L̂(β)")
    ax.set_title("Log-Likelihood Trace")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acceptance_rate(result: PMMMHResult, window: int = 100, figsize: Tuple[int, int] = (12, 5)):
    """Plot rolling acceptance rate."""
    acceptance = result.acceptance_trace.astype(float)
    
    # Compute rolling mean
    if len(acceptance) >= window:
        rolling = np.convolve(acceptance, np.ones(window)/window, mode='valid')
        x = np.arange(window-1, len(acceptance))
    else:
        rolling = acceptance
        x = np.arange(len(acceptance))
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, rolling, lw=1.0)
    ax.axhline(result.acceptance_rate, color='red', linestyle='--', label=f'Overall: {result.acceptance_rate:.3f}')
    ax.axhline(0.234, color='green', linestyle=':', label='Target: 0.234')
    ax.axvline(result.config.burn_in, color='orange', linestyle='--', alpha=0.5, label='Burn-in')
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Acceptance Rate (window={window})")
    ax.set_title("Rolling Acceptance Rate")
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_all_diagnostics(
    result: PMMMHResult,
    beta_true: Optional[np.ndarray] = None,
    beta_idx_list: Optional[List[Tuple[int, int]]] = None,
):
    """Generate all standard diagnostic plots."""
    print_summary(result, beta_true)
    
    print("\nGenerating diagnostic plots...")
    plot_loglik_trace(result)
    plot_acceptance_rate(result)
    plot_traces(result, beta_idx_list)
    plot_acf(result, beta_idx_list)
    plot_histograms(result, beta_true, beta_idx_list)
    print("Done.")
