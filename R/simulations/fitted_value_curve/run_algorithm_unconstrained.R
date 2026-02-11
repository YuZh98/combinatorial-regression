# ----------------------------------------
# Baseline Gibbs Sampler for Constrained Multivariate Probit
# ----------------------------------------

# Assumes X, y, beta_true, A, b are already defined
library(truncnorm)

set.seed(1234)

n <- nrow(X)
p <- ncol(X)
d <- ncol(y)

# ----------------------------------------
# Prior and Precomputations
# ----------------------------------------
b0 <- rep(0, p)
B0 <- 10
V <- solve(diag(1 / B0, p) + t(X) %*% X)
L <- t(chol(V))  # Cholesky factor of V

# ----------------------------------------
# Sampler Setup
# ----------------------------------------
n_iter <- 20000
beta_samples <- array(NA, dim = c(n_iter, p, d))  # to store beta draws
beta <- matrix(0, nrow = p, ncol = d)              # initial value for beta

# -------------------------------
# Gibbs sampler iterations
# -------------------------------
running_time <- system.time(
  for (iter in 1:n_iter) {
    # Step 1: Sample latent variables zeta given beta and y (vectorized over observations)
    zeta <- matrix(NA, n, d)
    for (j in 1:d) {
      # Compute the mean for column j
      mu_j <- X %*% beta[, j]
      # Set vectorized lower and upper bounds based on observed y
      lower_bound <- ifelse(y[, j] == 1, 0, -Inf)
      upper_bound <- ifelse(y[, j] == 1, Inf, 0)
      # Sample all n latent variables for outcome j in one call
      zeta[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu_j, sd = 1)
    }
    
    # Step 2: Sample beta for each outcome dimension using the precomputed Cholesky factor
    beta <- V %*% (matrix((1/B0) * b0, nrow = p, ncol = d) + t(X) %*% zeta) + L %*% matrix(rnorm(p * d), p, d)
    
    
    # Save the beta draws for this iteration
    beta_samples[iter, , ] <- beta
  }
)
cat(n_iter, "iterations completed.\n")
print(running_time)

# ----------------------------------------
# Posterior Summary
# ----------------------------------------
burn_in <- 5000
beta_post_mean <- apply(beta_samples[seq(burn_in+1, n_iter, by = 10), , ], c(2, 3), mean)

cat("Posterior mean (baseline method):\n"); print(beta_post_mean)

