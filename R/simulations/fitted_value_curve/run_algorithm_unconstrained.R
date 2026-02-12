# ------------------------------------------------------------
# run_algorithm_unconstrained.R
# Baseline Gibbs sampler for (unconstrained) multivariate probit
#
# EXPECTS in caller environment:
#   X (n x p), y (n x d), beta_true (p x d)
#
# PRODUCES in caller environment:
#   beta_samples (n_iter x p x d)
#   beta         (p x d) final state
#   running_time_unconstrained (system.time output)
#   beta_post_mean (p x d) [if burn-in < n_iter]
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(truncnorm)
})

# -------------------------------
# Helpers: env vars
# -------------------------------
get_env_int <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  as.integer(x)
}

# -------------------------------
# Validate required objects exist
# -------------------------------
required_objs <- c("X", "y", "beta_true")
missing <- required_objs[!vapply(required_objs, exists, logical(1), inherits = TRUE)]
if (length(missing) > 0) {
  stop("run_algorithm_unconstrained.R missing required objects in caller env: ",
       paste(missing, collapse = ", "))
}

n <- nrow(X)
p <- ncol(X)
d <- ncol(y)

stopifnot(nrow(beta_true) == p, ncol(beta_true) == d)

# -------------------------------
# Sampler settings
# -------------------------------
n_iter <- get_env_int("JASA_PROBIT_N_ITER", 20000)
log_every <- get_env_int("JASA_PROBIT_LOG_EVERY", 0)   # default: quiet
burn_in <- get_env_int("JASA_PROBIT_BURN_IN", 5000)
thin <- get_env_int("JASA_PROBIT_THIN", 10)

cat("\n---- unconstrained probit sampler (baseline) ----\n")
cat("n_iter=", n_iter, " burn_in=", burn_in, " thin=", thin, " log_every=", log_every, "\n", sep = "")
cat("-------------------------------------------------\n")

# -------------------------------
# Prior and precomputations
# -------------------------------
b0 <- rep(0, p)
B0 <- 10

V <- solve(diag(1 / B0, p) + t(X) %*% X)
L <- t(chol(V))  # Cholesky factor of V

# -------------------------------
# Storage
# -------------------------------
beta_samples <- array(NA_real_, dim = c(n_iter, p, d))
beta <- matrix(0, nrow = p, ncol = d)

# -------------------------------
# Gibbs sampler iterations
# -------------------------------
running_time_unconstrained <- system.time({
  for (iter in 1:n_iter) {
    if (log_every > 0 && (iter %% log_every == 0)) {
      cat("Iter:", iter, "\n")
    }

    # Step 1: Sample latent zeta given beta and y
    zeta <- matrix(NA_real_, n, d)
    for (j in 1:d) {
      mu_j <- X %*% beta[, j]
      lower <- ifelse(y[, j] == 1, 0, -Inf)
      upper <- ifelse(y[, j] == 1, Inf, 0)
      zeta[, j] <- rtruncnorm(n, a = lower, b = upper, mean = mu_j, sd = 1)
    }

    # Step 2: Sample beta
    mean_normal <- V %*% ((1 / B0) * matrix(b0, nrow = p, ncol = d) + t(X) %*% zeta)
    beta <- mean_normal + L %*% matrix(rnorm(p * d), p, d)

    beta_samples[iter, , ] <- beta
  }
})

cat(n_iter, "iterations (unconstrained) completed.\n")
print(running_time_unconstrained)

# -------------------------------
# Posterior summary
# -------------------------------
if (burn_in >= n_iter) {
  warning("burn_in >= n_iter; skipping posterior summary.")
} else {
  keep_idx <- seq(burn_in + 1, n_iter, by = thin)
  beta_post_mean <- apply(beta_samples[keep_idx, , , drop = FALSE], c(2, 3), mean)
  cat("Posterior mean (baseline method):\n")
  print(beta_post_mean)
}
