# ------------------------------------------------------------
# run_algorithm_constrained.R
# Augmented Gibbs sampler (constrained / ILP) for probit simulation
#
# EXPECTS in caller environment:
#   X (n x p), y (n x d), beta_true (p x d), A (m x d), b (length m)
#
# PRODUCES in caller environment:
#   beta_samples_augmented (n_iter x p x d)
#   beta_augmented         (p x d) final state
#   running_time_augmented (system.time output)
#   beta_post_mean_augmented (p x d)
#   rmse_augmented (scalar)
# ------------------------------------------------------------

# Compile Rcpp code (hit-and-run + feasibility / TUM utilities)
source("R/src/common/cpp_build.R")

suppressPackageStartupMessages({
  library(truncnorm)
  library(lpSolve)
  library(lintools)
})

# -------------------------------
# Helpers: env vars (lightweight)
# -------------------------------
get_env_int <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  as.integer(x)
}

# -------------------------------
# Validate required objects exist
# -------------------------------
required_objs <- c("X", "y", "A", "b", "beta_true")
missing <- required_objs[!vapply(required_objs, exists, logical(1), inherits = TRUE)]
if (length(missing) > 0) {
  stop("run_algorithm_constrained.R missing required objects in caller env: ",
       paste(missing, collapse = ", "))
}

n <- nrow(X)
p <- ncol(X)
d <- ncol(y)
m <- nrow(A)

stopifnot(nrow(beta_true) == p, ncol(beta_true) == d)
stopifnot(ncol(A) == d, length(b) == m)

# -------------------------------
# Sampler settings
# -------------------------------
n_iter <- get_env_int("JASA_PROBIT_N_ITER", 20000)
n_har <- get_env_int("JASA_PROBIT_N_HAR", 100)
log_every <- get_env_int("JASA_PROBIT_LOG_EVERY", 100)

burn_in <- get_env_int("JASA_PROBIT_BURN_IN", 5000)
thin <- get_env_int("JASA_PROBIT_THIN", 10)

cat("\n---- constrained probit sampler (augmented) ----\n")
cat("n_iter=", n_iter, " n_har=", n_har, " burn_in=", burn_in, " thin=", thin,
    " log_every=", log_every, "\n", sep = "")
cat("-----------------------------------------------\n")

# -------------------------------
# Initialize hit-and-run component
# -------------------------------
# U_free is n x m: indicates which constraints are active
U_free <- 1 * (t(A %*% t(y) == b))
U <- matrix(rexp(n * m), nrow = n, ncol = m) * U_free
UA <- U %*% A

# -------------------------------
# Prior and linear algebra prep
# -------------------------------
b0 <- rep(0, p)
B0 <- 10
V <- solve(diag(1 / B0, p) + t(X) %*% X)
L <- t(chol(V))

# -------------------------------
# Storage
# -------------------------------
beta_augmented <- matrix(0, p, d)
beta_samples_augmented <- array(NA_real_, dim = c(n_iter, p, d))
zeta <- matrix(NA_real_, n, d)

# Initialization of zeta
for (j in 1:d) {
  mu_j <- X %*% beta_augmented[, j]
  lower <- ifelse(y[, j] == 1, UA[, j], -Inf)
  upper <- ifelse(y[, j] == 1, Inf, UA[, j])
  zeta[, j] <- rtruncnorm(n, a = lower, b = upper, mean = mu_j, sd = 1)
}

# Safety for printing acceptance rate early
accept_or_not <- rep(1, n)

# -------------------------------
# Gibbs sampler loop
# -------------------------------
greaterequal <- 1 - y

running_time_augmented <- system.time({
  for (iter in 1:n_iter) {

    if (log_every > 0 && (iter %% log_every == 0)) {
      cat("Iter:", iter, "Acceptance Rate:", mean(accept_or_not), "\n")
    }

    # ---- Update U using current zeta ----
    U <- loop_hit_and_run(t(A), zeta, greaterequal, U, U_free, n_iter = n_har)
    UA <- U %*% A

    # ---- Propose new zeta ----
    zeta_new <- zeta
    for (j in 1:d) {
      mu_j <- X %*% beta_augmented[, j]
      lower <- ifelse(y[, j] == 1, UA[, j], -Inf)
      upper <- ifelse(y[, j] == 1, Inf, UA[, j])
      zeta_new[, j] <- rtruncnorm(n, a = lower, b = upper, mean = mu_j, sd = 1)
    }

    # Dual thresholding (same logic as your original)
    zeta_tilde <- ifelse(y > 0.5, pmax(zeta, zeta_new), pmin(zeta, zeta_new))

    # ---- Sample U_star and check feasibility ----
    U_star <- loop_hit_and_run(t(A), zeta_tilde, greaterequal, U, U_free, n_iter = n_har)
    accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)

    # ---- Accept/reject zeta update ----
    zeta <- zeta_new * accept_or_not + zeta * (1 - accept_or_not)

    # ---- Update beta ----
    mean_normal <- V %*% ((1 / B0) * matrix(b0, p, d) + t(X) %*% zeta)
    beta_augmented <- mean_normal + L %*% matrix(rnorm(p * d), p, d)

    beta_samples_augmented[iter, , ] <- beta_augmented
  }
})

cat(n_iter, "iterations (augmented) completed.\n")
print(running_time_augmented)

# -------------------------------
# Posterior summary (kept lightweight)
# -------------------------------
if (burn_in >= n_iter) {
  warning("burn_in >= n_iter; skipping posterior summary.")
} else {
  keep_idx <- seq(burn_in + 1, n_iter, by = thin)
  beta_post_mean_augmented <- apply(beta_samples_augmented[keep_idx, , , drop = FALSE], c(2, 3), mean)
  rmse_augmented <- sqrt(mean((beta_post_mean_augmented - beta_true)^2))

  cat("Posterior mean (augmented method):\n")
  print(beta_post_mean_augmented)
  cat("RMSE (augmented method):", rmse_augmented, "\n")
}
