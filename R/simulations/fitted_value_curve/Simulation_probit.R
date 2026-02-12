# ------------------------------------------------------------
# Simulation_probit.R
# Section 5.1: Fitted value curve / constrained vs unconstrained probit
# Repo scheme: run from repo root; write outputs to results/
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(truncnorm)
  library(lpSolve)
})

# -------------------------------
# Helpers: env vars
# -------------------------------
get_env_int <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  as.integer(x)
}
get_env_dbl <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  as.numeric(x)
}
get_env_bool <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  tolower(x) %in% c("1", "true", "t", "yes", "y")
}
get_env_str <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  x
}

# -------------------------------
# Reproducibility / outputs
# -------------------------------
SEED <- get_env_int("JASA_SEED", 1234)
set.seed(SEED)

RESULTS_ROOT <- file.path("results", "runs", "probit")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)

RUN_TAG <- get_env_str("JASA_RUN_TAG", "probit_default")
RUN_DIR <- file.path(RESULTS_ROOT, RUN_TAG)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

meta_path <- file.path(RUN_DIR, "run_metadata.txt")
cat(
  paste0("timestamp: ", Sys.time(), "\n"),
  paste0("run_tag: ", RUN_TAG, "\n"),
  paste0("seed: ", SEED, "\n"),
  paste0("R.version: ", R.version.string, "\n"),
  file = meta_path
)
capture.output(sessionInfo(), file = file.path(RUN_DIR, "sessionInfo.txt"))

# -------------------------------
# User-tunable settings
# -------------------------------
n <- get_env_int("JASA_N", 100)
p <- get_env_int("JASA_P", 2)
d <- get_env_int("JASA_D", 2)

# Optional: control whether to generate/saves trace plots
MAKE_PLOTS <- get_env_bool("JASA_PLOT", FALSE)

cat("---- probit simulation settings ----\n")
cat("n=", n, " p=", p, " d=", d, " seed=", SEED, " run_tag=", RUN_TAG, "\n", sep = "")
cat("------------------------------------\n\n")

# -------------------------------
# Simulate Data
# -------------------------------
# Design matrix X with intercept + one continuous covariate
X <- cbind(1, sort(rnorm(n))) # n x p, with p=2 default
stopifnot(ncol(X) == p)

# True regression coefficients (p x d)
beta_true <- matrix(rnorm(p * d), nrow = p, ncol = d)

# Latent response (Gaussian)
zeta_true <- X %*% beta_true + matrix(rnorm(n * d), nrow = n, ncol = d)

# ILP constraint for d=2: y_1 + y_2 <= 1
# (Kept for documentation; mapping implemented via closed form below)
A <- matrix(c(1, 1), ncol = 2)
b <- c(1)

# ILP mapping function (argmax c'y subject to Ay ≤ b, y ∈ {0,1}^2)
# For this simple constraint, the solution is:
# - choose y1=1 if c1 is the largest positive among {c1,c2,0}
# - choose y2=1 if c2 is the largest positive among {c1,c2,0}
ilp_map <- function(c) {
  ans <- c(0, 0)
  if (c[1] > max(c[2], 0)) ans[1] <- 1
  if (c[2] > max(c[1], 0)) ans[2] <- 1
  ans
}

y <- t(apply(zeta_true, 1, ilp_map))
stopifnot(nrow(y) == n, ncol(y) == d)

# Save inputs
saveRDS(X, file = file.path(RUN_DIR, "X.rds"))
saveRDS(y, file = file.path(RUN_DIR, "y.rds"))
saveRDS(beta_true, file = file.path(RUN_DIR, "beta_true.rds"))

# -------------------------------
# Run algorithms (sources must be within repo)
# -------------------------------
# NOTE: Place these files in the repo, e.g.:
#   R/simulations/fitted_value_curve/run_algorithm_unconstrained.R
#   R/simulations/fitted_value_curve/run_algorithm_constrained.R
#
# They are expected to create:
#   beta_samples              (unconstrained)
#   beta_samples_augmented    (constrained / ILP)
#
# If they currently depend on hard-coded paths, refactor them similarly.

source(file.path("R", "simulations", "fitted_value_curve", "run_algorithm_unconstrained.R"))
source(file.path("R", "simulations", "fitted_value_curve", "run_algorithm_constrained.R"))

# Basic sanity checks on outputs
if (!exists("beta_samples")) stop("Expected object 'beta_samples' not found after running unconstrained algorithm.")
if (!exists("beta_samples_augmented")) stop("Expected object 'beta_samples_augmented' not found after running constrained algorithm.")

# Save algorithm outputs
saveRDS(beta_samples, file = file.path(RUN_DIR, "beta_samples_unconstrained.rds"))
saveRDS(beta_samples_augmented, file = file.path(RUN_DIR, "beta_samples_constrained.rds"))

cat("Saved probit simulation outputs under: ", RUN_DIR, "\n", sep = "")



# -------------------------------
# Optional plots (saved to files)
# -------------------------------
if (MAKE_PLOTS) {
  fig_dir <- file.path(RUN_DIR, "figures")
  dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

  # Helper to save a trace plot for a specific (k,j)
  save_trace <- function(samples, k, j, main_title, out_path) {
    png(out_path, width = 1200, height = 800, res = 150)
    plot(samples[, k, j], type = "l",
         main = main_title,
         xlab = "Iteration", ylab = sprintf("beta[%d, %d]", k, j))
    abline(h = beta_true[k, j], col = "red")
    dev.off()
  }

  # Unconstrained traces
  for (j in 1:d) {
    for (k in 1:p) {
      save_trace(
        samples = beta_samples,
        k = k, j = j,
        main_title = sprintf("Trace: beta[%d,%d] (unconstrained)", k, j),
        out_path = file.path(fig_dir, sprintf("trace_unconstrained_beta_%d_%d.png", k, j))
      )
    }
  }

  # Constrained traces
  for (j in 1:d) {
    for (k in 1:p) {
      save_trace(
        samples = beta_samples_augmented,
        k = k, j = j,
        main_title = sprintf("Trace: beta[%d,%d] (constrained / ILP)", k, j),
        out_path = file.path(fig_dir, sprintf("trace_constrained_beta_%d_%d.png", k, j))
      )
    }
  }

  cat("Saved trace plots under: ", fig_dir, "\n", sep = "")
}
