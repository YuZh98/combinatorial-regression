"""
Pseudo-Marginal Metropolis-Hastings (PMMH) Sampler

This module implements a theoretically correct PMMH algorithm for Bayesian inference
when the likelihood is intractable but can be unbiasedly estimated via Monte Carlo.

Reference:
    Andrieu, C., & Roberts, G. O. (2009). The pseudo-marginal approach for efficient
    Monte Carlo computations. The Annals of Statistics, 37(2), 697-725.

Because PMMH does not work well on high-dimensional problems, the dimension considered is low. So, for simplicity, we enumerate all feasible solutions to solve the optimization problem in this file.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Callable, Union
from abc import ABC, abstractmethod
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime


# ============================================================================
# Filename Generation Utilities
# ============================================================================

def _float_to_tag(x: float, ndigits: int = 2) -> str:
    """Convert float to filename-safe string: 0.05 -> '0p05'."""
    s = f"{x:.{ndigits}g}"
    return s.replace(".", "p").replace("-", "m")


def make_pmmh_filename(
    n_iter: int,
    n: int,
    p: int,
    d: int,
    m: int,
    M_mc: int,
    tau_prior: float,
    proposal_scale_initial: float,
    alpha_smooth: float,
    adapt_proposal: bool,
    seed: int,
) -> str:
    """
    Generate standardized PMMH filename (no extension).
    
    Includes only quantities known BEFORE the run.
    Does NOT include final adapted scale.
    
    Example: PMMH_iter10000_n500_p3_d2_m1_M1000_tau5p0_ps0p05_alpha0p5_adapt1_seed123
    """
    tau_tag = _float_to_tag(tau_prior)
    ps_tag = _float_to_tag(proposal_scale_initial)
    alpha_tag = _float_to_tag(alpha_smooth)
    adapt_tag = 1 if adapt_proposal else 0
    
    filename = (
        f"PMMH_iter{n_iter}_n{n}_p{p}_d{d}_m{m}_"
        f"M{M_mc}_tau{tau_tag}_ps{ps_tag}_alpha{alpha_tag}_"
        f"adapt{adapt_tag}_seed{seed}"
    )
    
    return filename


# ============================================================================
# Beta Shape Validation (Single Source of Truth)
# ============================================================================

def validate_beta_shape(beta: np.ndarray, p: int, d: int, context: str = "") -> None:
    """
    Validate that beta has the correct shape (p, d).
    """
    if beta.ndim != 2:
        raise ValueError(
            f"{context}beta must be 2D, got {beta.ndim}D with shape {beta.shape}"
        )
    
    if beta.shape != (p, d):
        raise ValueError(
            f"{context}beta must have shape ({p}, {d}), got {beta.shape}. "
        )


# ============================================================================
# Configuration & Data Structures
# ============================================================================

@dataclass
class PMMMHConfig:
    """Configuration for PMMH sampler with validation."""
    M_mc: int = 1000  # Monte Carlo samples for likelihood estimation
    alpha_smooth: float = 0.5  # Laplace smoothing parameter (0 = no smoothing)
    proposal_scale: float = 0.05  # Initial scale of Gaussian random walk proposal
    n_iter: int = 10000  # Total MCMC iterations
    burn_in: int = 2000  # Burn-in iterations (discarded from output)
    seed: int = 123  # Random seed for reproducibility
    
    # Adaptive proposal scaling (Robbins-Monro during burn-in)
    adapt_proposal: bool = False  # Enable adaptive scaling
    target_accept: float = 0.234  # Target acceptance rate (optimal for Gaussian)
    adapt_rate: float = 0.6  # Step size decay: γ_t ∝ t^(-adapt_rate)
    adapt_interval: int = 50  # Update scale every N iterations
    adapt_start: int = 100  # Start adapting after this many iterations
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.M_mc < 1:
            raise ValueError(f"M_mc must be >= 1, got {self.M_mc}")
        if self.alpha_smooth < 0:
            raise ValueError(f"alpha_smooth must be >= 0, got {self.alpha_smooth}")
        if self.proposal_scale <= 0:
            raise ValueError(f"proposal_scale must be > 0, got {self.proposal_scale}")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")
        if self.burn_in < 0:
            raise ValueError(f"burn_in must be >= 0, got {self.burn_in}")
        if self.burn_in >= self.n_iter:
            raise ValueError(f"burn_in ({self.burn_in}) must be < n_iter ({self.n_iter})")
        if self.adapt_proposal:
            if not 0 < self.target_accept < 1:
                raise ValueError(f"target_accept must be in (0,1), got {self.target_accept}")
            if self.adapt_rate <= 0:
                raise ValueError(f"adapt_rate must be > 0, got {self.adapt_rate}")
            if self.adapt_interval < 1:
                raise ValueError(f"adapt_interval must be >= 1, got {self.adapt_interval}")
            if self.adapt_start < 0:
                raise ValueError(f"adapt_start must be >= 0, got {self.adapt_start}")
            if self.adapt_start >= self.burn_in:
                warnings.warn(
                    f"adapt_start ({self.adapt_start}) >= burn_in ({self.burn_in}): "
                    f"No adaptation will occur"
                )


@dataclass
class PMMMHResult:
    """Container for PMMH sampling results."""
    samples: np.ndarray  # (n_iter, p, d) posterior samples
    loglik_trace: np.ndarray  # (n_iter,) estimated log-likelihood trace
    acceptance_trace: np.ndarray  # (n_iter,) boolean acceptance indicators
    config: PMMMHConfig  # Configuration used
    metadata: dict = field(default_factory=dict)  # Additional info (timing, beta_true, etc.)
    total_runtime_seconds: float = 0.0
    runtime_per_iter_seconds: float = 0.0
    timestamp_start: Optional[str] = None  # ISO format
    timestamp_end: Optional[str] = None    # ISO format
    
    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        return float(np.mean(self.acceptance_trace))
    
    @property
    def samples_post_burnin(self) -> np.ndarray:
        """Samples after burn-in."""
        return self.samples[self.config.burn_in:, :, :]
    
    @property
    def n_effective_samples(self) -> int:
        """Number of post-burn-in samples."""
        return self.samples.shape[0] - self.config.burn_in


# ============================================================================
# Abstract Base Classes for Priors and Proposals
# ============================================================================

class Prior(ABC):
    """Abstract base class for prior distributions."""
    
    @abstractmethod
    def log_density(self, beta: np.ndarray) -> float:
        """Compute log π(β) up to additive constant."""
        pass
    
    @abstractmethod
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw β ~ π(β)."""
        pass
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of parameter matrix (p, d) or (p, d-1)."""
        pass


class Proposal(ABC):
    """Abstract base class for proposal distributions."""
    
    @abstractmethod
    def propose(self, beta_current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate proposal β' ~ q(·|β)."""
        pass
    
    @abstractmethod
    def log_proposal_ratio(self, beta_proposed: np.ndarray, beta_current: np.ndarray) -> float:
        """Compute log[q(β_curr | β_prop) / q(β_prop | β_curr)]."""
        pass
    
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int]:
        """Shape of parameter matrix (p, d) or (p, d-1)."""
        pass


# ============================================================================
# Prior Distribution
# ============================================================================

class IsotropicGaussianPrior(Prior):
    """
    Isotropic Gaussian prior: β ~ N(0, τ * I_{p×d})
    
    Log-density (up to additive constant):
        log π(β) = -0.5 * ||β||² / τ
    
    Additive constants cancel in MH ratio, so we omit them.
    
    Beta shape convention: Always (p, d).
    """
    
    def __init__(self, tau: float, shape: Tuple[int, int]):
        """
        Args:
            tau: Prior variance parameter (τ in N(0, τI))
            shape: (p, d) - shape of parameter matrix (STRICT: no baseline)
        """
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        if len(shape) != 2:
            raise ValueError(f"shape must be 2D tuple (p, d), got {shape}")
        self.tau = tau
        self._shape = shape
        self.p = shape[0]
        self.d = shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    def log_density(self, beta: np.ndarray) -> float:
        """Compute log π(β) up to additive constant."""
        validate_beta_shape(beta, self.p, self.d, context="IsotropicGaussianPrior: ")
        return -0.5 * np.sum(beta ** 2) / self.tau
    
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw β ~ N(0, τI)."""
        return rng.normal(loc=0.0, scale=np.sqrt(self.tau), size=self._shape)


# Backward compatibility alias
GaussianPrior = IsotropicGaussianPrior


class DiagonalGaussianPrior(Prior):
    """
    Diagonal Gaussian prior: β_jk ~ N(0, τ_jk) independently.
    
    Log-density: log π(β) = -0.5 * Σ_{jk} β_jk² / τ_jk
    
    Beta shape convention: Always (p, d).
    """
    
    def __init__(self, tau_matrix: np.ndarray):
        """
        Args:
            tau_matrix: (p, d) matrix of variances (STRICT: no baseline)
        """
        self.tau_matrix = np.asarray(tau_matrix)
        if self.tau_matrix.ndim != 2:
            raise ValueError(f"tau_matrix must be 2D, got shape {self.tau_matrix.shape}")
        if np.any(self.tau_matrix <= 0):
            raise ValueError("All tau values must be > 0")
        self._shape = self.tau_matrix.shape
        self.p = self._shape[0]
        self.d = self._shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    def log_density(self, beta: np.ndarray) -> float:
        validate_beta_shape(beta, self.p, self.d, context="DiagonalGaussianPrior: ")
        return -0.5 * np.sum((beta ** 2) / self.tau_matrix)
    
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=0.0, scale=np.sqrt(self.tau_matrix), size=self._shape)


# ============================================================================
# Proposal Distribution
# ============================================================================

class GaussianRandomWalkProposal(Proposal):
    """
    Symmetric Gaussian random walk proposal:
        β' | β ~ N(β, σ² I)
    
    Since this is symmetric, q(β'|β) = q(β|β'), so the proposal ratio
    in the MH acceptance probability is 1 (log ratio = 0).
    
    Supports adaptive scaling via the `scale` attribute.
    
    Beta shape convention: Always (p, d).
    """
    
    def __init__(self, scale: float, shape: Tuple[int, int]):
        """
        Args:
            scale: Standard deviation of proposal (σ in N(β, σ²I))
            shape: (p, d) - shape of parameter matrix (STRICT: no baseline)
        """
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if len(shape) != 2:
            raise ValueError(f"shape must be 2D tuple (p, d), got {shape}")
        self.scale = scale  # Mutable for adaptation
        self._shape = shape
        self.p = shape[0]
        self.d = shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    def propose(self, beta_current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Generate proposal β' ~ N(β, σ²I)."""
        validate_beta_shape(beta_current, self.p, self.d, context="GaussianRandomWalkProposal: ")
        return beta_current + self.scale * rng.normal(size=self._shape)
    
    def log_proposal_ratio(self, beta_proposed: np.ndarray, beta_current: np.ndarray) -> float:
        """
        Compute log[q(β_curr | β_prop) / q(β_prop | β_curr)].
        
        For symmetric proposals, this is always 0.
        """
        return 0.0


class DiagonalGaussianRWProposal(Proposal):
    """
    Diagonal Gaussian random walk: β' = β + Σ ⊙ ε, where ε ~ N(0, I)
    and Σ is a diagonal scale matrix.
    
    Symmetric, so log proposal ratio = 0.
    
    Beta shape convention: Always (p, d).
    """
    
    def __init__(self, scale_matrix: np.ndarray):
        """
        Args:
            scale_matrix: (p, d) matrix of scales (STRICT: no baseline)
        """
        self.scale_matrix = np.asarray(scale_matrix)
        if self.scale_matrix.ndim != 2:
            raise ValueError(f"scale_matrix must be 2D, got shape {self.scale_matrix.shape}")
        if np.any(self.scale_matrix <= 0):
            raise ValueError("All scales must be > 0")
        self._shape = self.scale_matrix.shape
        self.p = self._shape[0]
        self.d = self._shape[1]
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape
    
    def propose(self, beta_current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        validate_beta_shape(beta_current, self.p, self.d, context="DiagonalGaussianRWProposal: ")
        return beta_current + self.scale_matrix * rng.normal(size=self._shape)
    
    def log_proposal_ratio(self, beta_proposed: np.ndarray, beta_current: np.ndarray) -> float:
        return 0.0


# ============================================================================
# Likelihood Estimator
# ============================================================================

class PseudoMarginalEstimator:
    """
    Unbiased Monte Carlo estimator for log-likelihood in discrete choice model.
    
    Model:
        For each observation i:
            ζ_i | β, x_i ~ N(x_i^T β, I_d)
            y_i = argmax_{z ∈ Z_feasible} ζ_i^T z
        
        True likelihood:
            P(y_i | β, x_i) = P(y_i = argmax_z ζ_i^T z)
        
        Estimator (Monte Carlo):
            1. Draw M_mc samples: ζ_i^(j) ~ N(x_i^T β, I_d)
            2. Compute: ẑ_i^(j) = argmax_z ζ_i^(j)^T z
            3. Estimate: p̂_i = (1/M_mc) Σ_j 1{ẑ_i^(j) = y_i}
            4. Log-likelihood: log L̂(β) = Σ_i log(p̂_i)
    
    Smoothing (optional):
        To avoid p̂_i = 0 (which gives log L̂ = -∞), we use Laplace smoothing:
            p̂_i = (count_i + α) / (M_mc + K·α)
        where K = |Z_feasible| and α = alpha_smooth.
    
    Vectorization:
        If vectorized=True, processes observations in batches for speed.
        Preserves exact same random number sequence and numerical results.
    
    PMMH Validity:
        - E[p̂_i] = P(y_i | β, x_i), so estimator is unbiased
        - Each call draws fresh random numbers (independence)
        - Smoothing introduces small bias but stabilizes variance
    """
    
    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        Z_feasible: np.ndarray,
        M_mc: int,
        alpha_smooth: float = 0.0,
        vectorized: bool = False,
        batch_size: int = 100,
    ):
        """
        Args:
            y: (n, d) observed discrete choices
            X: (n, p) covariates
            Z_feasible: (K, d) feasible discrete choices satisfying constraints
            M_mc: Number of Monte Carlo samples per observation
            alpha_smooth: Laplace smoothing parameter (0 = no smoothing)
            vectorized: Use vectorized implementation for speed
            batch_size: Observations per batch in vectorized mode
        """
        self.y = np.asarray(y)
        self.X = np.asarray(X)
        self.Z_feasible = np.asarray(Z_feasible)
        
        if self.y.ndim != 2:
            raise ValueError(f"y must be 2D, got shape {self.y.shape}")
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        if self.Z_feasible.ndim != 2:
            raise ValueError(f"Z_feasible must be 2D, got shape {self.Z_feasible.shape}")
        
        self.n, self.d = self.y.shape
        self.p = self.X.shape[1]
        self.K = self.Z_feasible.shape[0]  # Number of feasible choices
        
        if self.X.shape[0] != self.n:
            raise ValueError(f"X and y must have same n, got {self.X.shape[0]} vs {self.n}")
        if self.Z_feasible.shape[1] != self.d:
            raise ValueError(f"Z_feasible and y must have same d, got {self.Z_feasible.shape[1]} vs {self.d}")
        if self.K == 0:
            raise ValueError("Z_feasible is empty")
        
        if M_mc < 1:
            raise ValueError(f"M_mc must be >= 1, got {M_mc}")
        if alpha_smooth < 0:
            raise ValueError(f"alpha_smooth must be >= 0, got {alpha_smooth}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        
        self.M_mc = M_mc
        self.alpha_smooth = alpha_smooth
        self.vectorized = vectorized
        self.batch_size = batch_size
        
        # Warn if M_mc is too small
        if M_mc < 10 * self.K:
            warnings.warn(
                f"M_mc={M_mc} may be too small for K={self.K} feasible choices. "
                f"Consider M_mc >= {10 * self.K} for reliable estimates."
            )
    
    def estimate_loglik(self, beta: np.ndarray, rng: np.random.Generator) -> float:
        """
        Estimate log L(β) via Monte Carlo.
        
        This is the ONLY likelihood estimator used in PMMH.
        Every call draws fresh random numbers to ensure independence.
        
        Beta shape convention: Always (p, d).
        
        Args:
            beta: (p, d) parameter matrix
            rng: NumPy random generator
            
        Returns:
            Estimated log-likelihood (scalar, possibly -∞)
        """
        # Strict (p, d) validation - no baseline support
        validate_beta_shape(beta, self.p, self.d, context="PseudoMarginalEstimator: ")
        
        # Compute mean utilities: μ = Xβ
        mu = self.X @ beta  # (n, d)
        
        if self.vectorized:
            return self._estimate_loglik_vectorized(mu, rng)
        else:
            return self._estimate_loglik_sequential(mu, rng)
    
    def _estimate_loglik_sequential(self, mu: np.ndarray, rng: np.random.Generator) -> float:
        """Original sequential implementation."""
        loglik = 0.0
        
        for i in range(self.n):
            y_i = self.y[i, :]  # (d,)
            mu_i = mu[i, :]  # (d,)
            
            # Draw M_mc samples of latent utility
            eps = rng.normal(size=(self.M_mc, self.d))  # (M_mc, d)
            zeta = mu_i + eps  # (M_mc, d)
            
            # Find optimal choice for each sample
            scores = self.Z_feasible @ zeta.T  # (K, M_mc)
            idx_best = np.argmax(scores, axis=0)  # (M_mc,)
            z_hat = self.Z_feasible[idx_best, :]  # (M_mc, d)
            
            # Count matches with observed choice
            matches = np.all(z_hat == y_i, axis=1)  # (M_mc,) boolean
            count_match = np.sum(matches)
            
            # Estimate probability with Laplace smoothing
            if self.alpha_smooth > 0:
                p_hat_i = (count_match + self.alpha_smooth) / (self.M_mc + self.K * self.alpha_smooth)
            else:
                p_hat_i = count_match / self.M_mc
                if p_hat_i == 0.0:
                    return -np.inf
            
            loglik += np.log(p_hat_i)
        
        # Safety check for NaN
        if np.isnan(loglik):
            raise RuntimeError("NaN detected in log-likelihood estimate")
        
        return float(loglik)
    
    def _estimate_loglik_vectorized(self, mu: np.ndarray, rng: np.random.Generator) -> float:
        """Vectorized implementation with chunking for memory efficiency."""
        loglik = 0.0
        
        # Process observations in batches
        for batch_start in range(0, self.n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.n)
            batch_size_actual = batch_end - batch_start
            
            y_batch = self.y[batch_start:batch_end, :]  # (B, d)
            mu_batch = mu[batch_start:batch_end, :]  # (B, d)
            
            # Draw M_mc samples for all observations in batch
            # Shape: (B, M_mc, d)
            eps_batch = rng.normal(size=(batch_size_actual, self.M_mc, self.d))
            
            # Broadcast: (B, 1, d) + (B, M_mc, d) → (B, M_mc, d)
            zeta_batch = mu_batch[:, None, :] + eps_batch
            
            # Compute scores for all samples and choices
            # Z_feasible: (K, d), zeta_batch: (B, M_mc, d)
            # Use einsum: (K, d) @ (B, M_mc, d) → (B, K, M_mc)
            scores_batch = np.einsum('kd,bmd->bkm', self.Z_feasible, zeta_batch)
            
            # Find best choice for each sample
            idx_best_batch = np.argmax(scores_batch, axis=1)  # (B, M_mc)
            
            # Get chosen vectors
            z_hat_batch = self.Z_feasible[idx_best_batch, :]  # (B, M_mc, d)
            
            # Count matches for each observation
            matches_batch = np.all(z_hat_batch == y_batch[:, None, :], axis=2)  # (B, M_mc)
            count_matches_batch = np.sum(matches_batch, axis=1)  # (B,)
            
            # Compute probabilities with smoothing
            if self.alpha_smooth > 0:
                p_hat_batch = (count_matches_batch + self.alpha_smooth) / (
                    self.M_mc + self.K * self.alpha_smooth
                )
            else:
                p_hat_batch = count_matches_batch / self.M_mc
                if np.any(p_hat_batch == 0.0):
                    return -np.inf
            
            # Accumulate log-likelihood
            loglik += np.sum(np.log(p_hat_batch))
        
        # Safety check for NaN
        if np.isnan(loglik):
            raise RuntimeError("NaN detected in log-likelihood estimate")
        
        return float(loglik)
    

# ============================================================================
# Core PMMH Step Function
# ============================================================================

def pmmh_step(
    beta_current: np.ndarray,
    loglik_current: float,
    log_prior_current: float,
    estimator: PseudoMarginalEstimator,
    prior: GaussianPrior,
    proposal: GaussianRandomWalkProposal,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Single PMMH Metropolis-Hastings step.
    
    This is the SINGLE SOURCE OF TRUTH for the PMMH acceptance logic.
    
    PMMH Algorithm:
        1. Propose β' ~ q(·|β) = N(β, σ²I)
        2. Estimate log L̂(β') using fresh random numbers
        3. Compute log α = log π(β') + log L̂(β') - log π(β) - log L̂(β)
        4. Accept with probability min(1, exp(log α))
        5. Return β' if accepted, else return β
    
    Theoretical Guarantees:
        - Satisfies detailed balance w.r.t. π(β | y) ∝ π(β) L(β | y)
        - Converges to correct posterior as n_iter → ∞
        - Requires: unbiased estimator, independent estimates, log-scale numerics
    
    Args:
        beta_current: Current state (p, d)
        loglik_current: Pre-computed log L̂(β_current)
        log_prior_current: Pre-computed log π(β_current)
        estimator: Likelihood estimator
        prior: Prior distribution
        proposal: Proposal distribution
        rng: Random number generator
        
    Returns:
        (beta_next, loglik_next, log_prior_next, accepted)
            beta_next: Next state (p, d)
            loglik_next: log L̂(β_next)
            log_prior_next: log π(β_next)
            accepted: Boolean, True if proposal was accepted
    """
    # Step 1: Propose new state
    beta_proposed = proposal.propose(beta_current, rng)
    
    # Step 2: Evaluate proposal
    log_prior_proposed = prior.log_density(beta_proposed)
    loglik_proposed = estimator.estimate_loglik(beta_proposed, rng)
    
    # Step 3: Compute MH acceptance ratio (in log scale)
    # log α = [log π(β') + log L̂(β')] - [log π(β) + log L̂(β)] + log[q(β|β') / q(β'|β)]
    log_proposal_ratio = proposal.log_proposal_ratio(beta_proposed, beta_current)
    
    log_alpha = (
        (log_prior_proposed + loglik_proposed) -
        (log_prior_current + loglik_current) +
        log_proposal_ratio
    )
    
    # Safety check: log_alpha should never be NaN
    # (Can be -∞ if loglik_proposed = -∞, which is valid and leads to rejection)
    if np.isnan(log_alpha):
        raise RuntimeError(
            f"NaN in acceptance ratio: "
            f"log_prior_prop={log_prior_proposed}, loglik_prop={loglik_proposed}, "
            f"log_prior_curr={log_prior_current}, loglik_curr={loglik_current}"
        )
    
    # Step 4: Accept/reject (log-scale comparison for numerical stability)
    log_u = np.log(rng.random())  # log(U) where U ~ Uniform(0, 1)
    
    if log_u < log_alpha:
        # Accept proposal
        return beta_proposed, loglik_proposed, log_prior_proposed, True
    else:
        # Reject proposal, keep current state
        return beta_current, loglik_current, log_prior_current, False


# ============================================================================
# PMMH Sampler
# ============================================================================

class PMMMHSampler:
    """
    Pseudo-Marginal Metropolis-Hastings sampler.
    
    Combines estimator, prior, and proposal into a complete MCMC algorithm.
    Supports adaptive proposal scaling during burn-in (Robbins-Monro).
    """
    
    def __init__(
        self,
        estimator: PseudoMarginalEstimator,
        prior: Prior,
        proposal: Proposal,
    ):
        """
        Args:
            estimator: Likelihood estimator
            prior: Prior distribution with shape (p, d)
            proposal: Proposal distribution with shape (p, d)
        """
        # Strict shape validation - all must be (p, d)
        if estimator.p != prior.p or estimator.d != prior.d:
            raise ValueError(
                f"Estimator shape ({estimator.p}, {estimator.d}) != "
                f"Prior shape ({prior.p}, {prior.d}). "
                f"All components must have shape (p, d)."
            )
        if estimator.p != proposal.p or estimator.d != proposal.d:
            raise ValueError(
                f"Estimator shape ({estimator.p}, {estimator.d}) != "
                f"Proposal shape ({proposal.p}, {proposal.d}). "
                f"All components must have shape (p, d)."
            )
        
        self.estimator = estimator
        self.prior = prior
        self.proposal = proposal
        self.shape = (estimator.p, estimator.d)  # Always (p, d)
    
    def run(
        self,
        config: PMMMHConfig,
        init_beta: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> PMMMHResult:
        """
        Run PMMH sampler for n_iter iterations with optional adaptive scaling.
        
        Adaptive Scaling (if config.adapt_proposal=True):
            - Adaptation occurs during burn-in only [adapt_start, burn_in)
            - Uses Robbins-Monro rule: log(scale) += γ_t * (α_t - target)
            - After burn-in, scale is frozen → standard PMMH with fixed proposal
            - This preserves detailed balance for post-burn-in samples
        
        Args:
            config: PMMH configuration
            init_beta: Initial state (p, d). If None, sample from prior.
            verbose: Print progress every 1000 iterations
            
        Returns:
            PMMMHResult with samples, traces, runtime, and metadata
        """
        import time
        
        # Initialize RNG
        rng = np.random.default_rng(config.seed)
        
        # Track start time
        timestamp_start = datetime.now().isoformat()
        time_start = time.time()
        
        # Initialize state
        if init_beta is None:
            beta = self.prior.sample(rng)
        else:
            beta = np.array(init_beta, copy=True)
            # Strict validation: init_beta must be (p, d)
            validate_beta_shape(beta, self.shape[0], self.shape[1], context="init_beta: ")
        
        # Evaluate initial state
        log_prior_curr = self.prior.log_density(beta)
        loglik_curr = self.estimator.estimate_loglik(beta, rng)
        
        if verbose:
            print(f"Initial state: log π(β) = {log_prior_curr:.2f}, log L̂(β) = {loglik_curr:.2f}")
            if config.adapt_proposal:
                print(f"Adaptive scaling enabled: target_accept={config.target_accept:.3f}, "
                      f"adapt during [{config.adapt_start}, {config.burn_in})")
        
        # Storage
        samples = np.zeros((config.n_iter, self.shape[0], self.shape[1]))
        loglik_trace = np.zeros(config.n_iter)
        acceptance_trace = np.zeros(config.n_iter, dtype=bool)
        
        # Adaptive scaling state
        if config.adapt_proposal and hasattr(self.proposal, 'scale'):
            adapt_enabled = True
            scale_trace = np.zeros(config.n_iter)  # Track scale evolution
            accept_window = []  # Rolling window for acceptance rate
        else:
            adapt_enabled = False
            if config.adapt_proposal:
                warnings.warn("Adaptation requested but proposal does not have 'scale' attribute")
        
        # MCMC loop
        n_accept = 0
        for t in range(config.n_iter):
            # Single PMMH step
            beta, loglik_curr, log_prior_curr, accepted = pmmh_step(
                beta_current=beta,
                loglik_current=loglik_curr,
                log_prior_current=log_prior_curr,
                estimator=self.estimator,
                prior=self.prior,
                proposal=self.proposal,
                rng=rng,
            )
            
            # Record
            samples[t, :, :] = beta
            loglik_trace[t] = loglik_curr
            acceptance_trace[t] = accepted
            
            if accepted:
                n_accept += 1
            
            # Adaptive scaling (Robbins-Monro during burn-in)
            if adapt_enabled:
                scale_trace[t] = self.proposal.scale
                
                if config.adapt_start <= t < config.burn_in:
                    # Update rolling window
                    accept_window.append(1.0 if accepted else 0.0)
                    
                    # Adapt every adapt_interval iterations
                    if (t + 1) % config.adapt_interval == 0 and len(accept_window) >= config.adapt_interval:
                        # Compute acceptance rate over last interval
                        recent_accept_rate = np.mean(accept_window[-config.adapt_interval:])
                        
                        # Robbins-Monro step size with decay
                        adapt_iter = (t - config.adapt_start) // config.adapt_interval + 1
                        gamma_t = 1.0 / (adapt_iter ** config.adapt_rate)
                        
                        # Update log(scale)
                        log_scale = np.log(self.proposal.scale)
                        log_scale += gamma_t * (recent_accept_rate - config.target_accept)
                        
                        # Update proposal scale
                        self.proposal.scale = np.exp(log_scale)
                        
                        if verbose and (t + 1) % 1000 == 0:
                            print(
                                f"  [Adapt] t={t+1}: accept_rate={recent_accept_rate:.3f}, "
                                f"scale={self.proposal.scale:.4f}"
                            )
            
            # Progress
            if verbose and (t + 1) % 100 == 0:
                acc_rate = n_accept / (t + 1)
                status = "[Burn-in]" if t < config.burn_in else "[Post-burn]"
                print(
                    f"{status} Iteration {t+1}/{config.n_iter}: "
                    f"log L̂ = {loglik_curr:.2f}, "
                    f"acceptance rate = {acc_rate:.3f}"
                )
        
        # Track end time
        time_end = time.time()
        timestamp_end = datetime.now().isoformat()
        total_runtime = time_end - time_start
        runtime_per_iter = total_runtime / config.n_iter
        
        # Final acceptance rate
        final_acc_rate = n_accept / config.n_iter
        if verbose:
            print(f"\nFinal acceptance rate: {final_acc_rate:.3f}")
            print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
            print(f"Runtime per iteration: {runtime_per_iter*1000:.2f} ms")
            if adapt_enabled:
                print(f"Final proposal scale: {self.proposal.scale:.4f}")
        
        # Metadata
        metadata = {
            'final_acceptance_rate': final_acc_rate,
            'init_beta': init_beta,
        }
        if adapt_enabled:
            metadata['scale_trace'] = scale_trace
            metadata['final_scale'] = self.proposal.scale
        
        # Return results with runtime
        return PMMMHResult(
            samples=samples,
            loglik_trace=loglik_trace,
            acceptance_trace=acceptance_trace,
            config=config,
            metadata=metadata,
            total_runtime_seconds=total_runtime,
            runtime_per_iter_seconds=runtime_per_iter,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
        )
        

# ============================================================================
# Helper Functions for Data Generation
# ============================================================================

def enumerate_feasible_Z(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Enumerate all binary vectors z ∈ {0,1}^d satisfying Az ≤ b.
    
    WARNING: Computational cost is O(2^d). Only feasible for d ≤ 20.
    
    Args:
        A: (m, d) constraint matrix
        b: (m,) constraint RHS
        
    Returns:
        (K, d) array of feasible binary vectors
        
    Raises:
        ValueError: If no feasible vectors exist or d > 20
    """
    A = np.asarray(A)
    b = np.asarray(b)
    
    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {A.shape}")
    if b.ndim != 1:
        raise ValueError(f"b must be 1D, got shape {b.shape}")
    
    m, d = A.shape
    
    if b.shape[0] != m:
        raise ValueError(f"A and b must have matching dimensions, got {A.shape} and {b.shape}")
    
    if d > 20:
        raise ValueError(
            f"d={d} is too large for enumeration (2^d = {2**d} vectors). "
            f"Maximum supported: d=20."
        )
    
    # Generate all 2^d binary vectors
    from itertools import product
    all_Z = np.array(list(product([0, 1], repeat=d)), dtype=int)  # (2^d, d)
    
    # Check which satisfy Az ≤ b
    mask = (A @ all_Z.T <= b.reshape(-1, 1)).all(axis=0)
    Z_feasible = all_Z[mask]
    
    if Z_feasible.shape[0] == 0:
        raise ValueError("No feasible binary vectors satisfy Az ≤ b. Check constraints.")
    
    return Z_feasible


def generate_simulation_data(
    n: int = 1000,
    p: int = 5,
    d: int = 2,
    m: int = 1,
    tau_beta: float = 1.0,
    rng_seed: int = 123,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data from the discrete choice model.
    
    Model:
        β_true ~ N(0, τI)  [prior]
        ζ_i = X_i β_true + ε_i,  ε_i ~ N(0, I_d)  [latent utility]
        y_i = argmax_{z ∈ Z_feasible} ζ_i^T z  [observed choice]
    
    Args:
        n: Number of observations
        p: Number of covariates
        d: Dimension of discrete choice
        m: Number of linear constraints
        tau_beta: Prior variance for β_true
        rng_seed: Random seed
        verbose: Print data shapes
        
    Returns:
        (y, X, Z_feasible, A, b, beta_true)
            y: (n, d) observed choices
            X: (n, p) covariates
            Z_feasible: (K, d) feasible choices
            A: (m, d) constraint matrix
            b: (m,) constraint RHS
            beta_true: (p, d) true parameters
    """
    rng = np.random.default_rng(rng_seed)
    
    # Generate constraints (simple TU structure)
    # Special case for d=2, m=1 to ensure non-trivial feasible set
    if d == 2 and m == 1:
        A = np.array([[1, 1]])
        b = np.array([1])
    else:
        A = np.zeros((m, d), dtype=int)
        for k in range(m):
            if d >= 2:
                j1, j2 = rng.choice(d, size=2, replace=False)
                if rng.random() < 0.5:
                    A[k, j1] = 1
                    A[k, j2] = -1
                else:
                    A[k, j1] = -1
                    A[k, j2] = 1
        b = rng.integers(low=0, high=2, size=m, dtype=int)
    
    # Enumerate feasible set
    Z_feasible = enumerate_feasible_Z(A, b)
    K = Z_feasible.shape[0]
    
    # Generate covariates
    X = rng.normal(size=(n, p))
    
    # Sample true parameters from prior
    beta_true = rng.normal(loc=0.0, scale=np.sqrt(tau_beta), size=(p, d))
    
    # Generate latent utilities
    eps = rng.normal(size=(n, d))
    zeta_true = X @ beta_true + eps
    
    # Generate observed choices via argmax
    y = np.zeros((n, d), dtype=int)
    for i in range(n):
        scores_i = Z_feasible @ zeta_true[i, :]
        idx = np.argmax(scores_i)
        y[i, :] = Z_feasible[idx, :]
    
    if verbose:
        print("Generated simulation data:")
        print(f"  n={n}, p={p}, d={d}, m={m}")
        print(f"  y shape: {y.shape}")
        print(f"  X shape: {X.shape}")
        print(f"  Z_feasible shape: {Z_feasible.shape} (K={K} feasible choices)")
        print(f"  A shape: {A.shape}")
        print(f"  b shape: {b.shape}")
        print(f"  beta_true:\n{beta_true}")
    
    return y, X, Z_feasible, A, b, beta_true