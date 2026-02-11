# ----------------------------------------
# Augmented Gibbs Sampler with Hit-and-Run for Combinatorial Probit Model
# ----------------------------------------

library(truncnorm)
library(lpSolve)
library(lintools)
library(Rcpp)

# Compile C++ routines (ensure files are in the same directory)
sourceCpp("hit_and_run.cpp")
# sourceCpp("tum_check.cpp")

# ----------------------------------------
# Assumes: X, y, beta_true, A, b already defined
# ----------------------------------------

n <- nrow(X)
p <- ncol(X)
d <- ncol(y)
m <- nrow(A)

set.seed(1234)

# Initialize hit-and-run component
U_free <- 1 * (t(A %*% t(y) == b))
U <- matrix(rexp(n * m), nrow = n, ncol = m) * U_free
UA <- U %*% A

# Prior and Cholesky prep
b0 <- rep(0, p)
B0 <- 10
V <- solve(diag(1 / B0, p) + t(X) %*% X)
L <- t(chol(V))

# Sampler setup
n_iter <- 20000
beta_augmented <- matrix(0, p, d)
beta_samples_augmented <- array(NA, dim = c(n_iter, p, d))
zeta <- matrix(NA, n, d)

# Initialization
for (j in 1:d) {
  mu_j <- X %*% beta_augmented[, j]
  lower <- ifelse(y[, j] == 1, UA[, j], -Inf)
  upper <- ifelse(y[, j] == 1, Inf, UA[, j])
  zeta[, j] <- rtruncnorm(n, a = lower, b = upper, mean = mu_j, sd = 1)
}

# ----------------------------------------
# Gibbs Sampler Loop
# ----------------------------------------
running_time_augmented <- system.time({
  for (iter in 1:n_iter) {
    if (iter %% 100 == 0) {
      cat("Iter:", iter, "Acceptance Rate:", mean(accept_or_not), "\n")
    }
    
    # Update U using zeta
    greaterequal <- 1 - y
    U <- loop_hit_and_run(t(A), zeta, greaterequal, U, U_free, n_iter = 100)
    UA <- U %*% A
    
    # Propose new zeta
    zeta_new <- zeta
    for (j in 1:d) {
      mu_j <- X %*% beta_augmented[, j]
      lower <- ifelse(y[, j] == 1, UA[, j], -Inf)
      upper <- ifelse(y[, j] == 1, Inf, UA[, j])
      zeta_new[, j] <- rtruncnorm(n, a = lower, b = upper, mean = mu_j, sd = 1)
    }
    
    # Apply dual thresholding
    zeta_tilde <- ifelse(y > 0.5, pmax(zeta, zeta_new), pmin(zeta, zeta_new))
    
    # Sample new U and check feasibility
    U_star <- loop_hit_and_run(t(A), zeta_tilde, greaterequal, U, U_free, n_iter = 100)
    accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)
    
    # Accept/reject zeta update
    zeta <- zeta_new * accept_or_not + zeta * (1 - accept_or_not)
    
    # Update beta
    mean_normal <- V %*% ((1 / B0) * matrix(b0, p, d) + t(X) %*% zeta)
    beta_augmented <- mean_normal + L %*% matrix(rnorm(p * d), p, d)
    
    beta_samples_augmented[iter, , ] <- beta_augmented
  }
})
cat(n_iter, "iterations (augmented) completed.\n")
print(running_time_augmented)

# ----------------------------------------
# Posterior Summary
# ----------------------------------------
burn_in <- 5000
beta_post_mean_augmented <- apply(beta_samples_augmented[seq(burn_in+1, n_iter, by = 10), , ], c(2, 3), mean)
rmse_augmented <- sqrt(mean((beta_post_mean_augmented - beta_true)^2))

cat("Posterior mean (augmented method):\n"); print(beta_post_mean_augmented)
cat("RMSE (augmented method):", rmse_augmented, "\n")
