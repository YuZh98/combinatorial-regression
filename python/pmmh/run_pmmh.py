import numpy as np
from pmmh_core import (
    generate_simulation_data,
    PMMMHConfig,
    PMMMHSampler,
    PseudoMarginalEstimator,
    GaussianPrior,
    GaussianRandomWalkProposal,
)


def main():
    print("="*70)
    print("PMMH Example: Discrete Choice Model with Linear Constraints")
    print("="*70)
    
    # ========================================================================
    # Step 1: Generate Synthetic Data
    # ========================================================================
    print("\n[Step 1] Generating synthetic data...")
    
    # Simulation parameters
    n = 1000          # Number of observations
    p = 5            # Number of covariates
    d = 5            # Dimension of discrete choice
    m = 2            # Number of constraints
    tau_prior = 1.0  # Prior variance (use same for data generation and inference)
    data_seed = 42
    
    y, X, Z_feasible, A, b, beta_true = generate_simulation_data(
        n=n,
        p=p,
        d=d,
        m=m,
        tau_beta=tau_prior,
        rng_seed=data_seed,
        verbose=True,
    )
    
    
    # ========================================================================
    # Step 2: Configure PMMH
    # ========================================================================
    print("\n[Step 2] Configuring PMMH sampler...")
    
    config = PMMMHConfig(
        M_mc=1000,              # Monte Carlo samples for likelihood estimation
        alpha_smooth=0.5,       # Laplace smoothing parameter
        proposal_scale=0.01,    # Std dev of random walk proposal
        n_iter=3000,           # Total MCMC iterations
        burn_in=1000,           # Burn-in period
        seed=1234,               # Random seed for reproducibility
    )
    
    print(f"Configuration:")
    print(f"  M_mc: {config.M_mc}")
    print(f"  alpha_smooth: {config.alpha_smooth}")
    print(f"  proposal_scale: {config.proposal_scale}")
    print(f"  n_iter: {config.n_iter}")
    print(f"  burn_in: {config.burn_in}")
    print(f"  seed: {config.seed}")
    
    # ========================================================================
    # Step 3: Build PMMH Components
    # ========================================================================
    print("\n[Step 3] Building PMMH components...")
    
    # Likelihood estimator
    estimator = PseudoMarginalEstimator(
        y=y,
        X=X,
        Z_feasible=Z_feasible,
        M_mc=config.M_mc,
        alpha_smooth=config.alpha_smooth,
    )
    print(f"  Estimator: n={estimator.n}, p={estimator.p}, d={estimator.d}, K={estimator.K}")
    
    # Prior distribution
    prior = GaussianPrior(tau=tau_prior, shape=(p, d))
    print(f"  Prior: β ~ N(0, {tau_prior}*I)")
    
    # Proposal distribution
    proposal = GaussianRandomWalkProposal(scale=config.proposal_scale, shape=(p, d))
    print(f"  Proposal: β' ~ N(β, {config.proposal_scale}²*I)")
    
    # Sampler
    sampler = PMMMHSampler(estimator, prior, proposal)
    print("  Sampler: Ready")
    
    # ========================================================================
    # Step 4: Run PMMH
    # ========================================================================
    print("\n[Step 4] Running PMMH sampler...")
    print(f"This will take approximately {config.n_iter * config.M_mc * n / 1e6:.1f}M likelihood evaluations")
    print("(Progress printed every 1000 iterations)\n")
    
    result = sampler.run(config, init_beta=None, verbose=True)
    
    # ========================================================================
    # Step 5: Display Results
    # ========================================================================
    print("\n[Step 5] Results summary...")
    
    # Basic info
    print(f"\nSamples shape: {result.samples.shape}")
    print(f"Post-burn-in samples: {result.n_effective_samples}")
    print(f"Acceptance rate: {result.acceptance_rate:.3f}")
    
    # Posterior mean vs true
    posterior_mean = result.samples_post_burnin.mean(axis=0)
    print(f"\nPosterior mean:")
    print(posterior_mean)
    print(f"\nTrue β:")
    print(beta_true)
    print(f"\nDifference:")
    print(posterior_mean - beta_true)
    
    # RMSE
    rmse = np.sqrt(np.mean((posterior_mean - beta_true)**2))
    print(f"\nRMSE: {rmse:.4f}")
    
    return result, beta_true



if __name__ == "__main__":
    # Run main example
    result, beta_true = main()
    
    
