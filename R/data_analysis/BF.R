# ------------------------------------------------------------
# BF.R
# Bayes factor comparing full vs reduced duck models using saved MCMC draws
# Repo scheme: run from repo root; read from results/runs; write to results/tables
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(splines)
})

set.seed(123)

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
get_env_str <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  x
}

# -------------------------------
# Paths
# -------------------------------
DATA_DIR <- Sys.getenv("JASA_DATA_DIR", unset = file.path("data", "waterfowl_matching"))
DUCKS_CSV <- file.path(DATA_DIR, "duck_data.csv")
Z_CSV <- file.path(DATA_DIR, "Z_matrix.csv")

RESULTS_RUNS_ROOT <- file.path("results", "runs", "data_analysis")
RESULTS_TABLES_ROOT <- file.path("results", "tables", "data_analysis")
dir.create(RESULTS_TABLES_ROOT, recursive = TRUE, showWarnings = FALSE)

# A run tag for *this BF computation* (separate from full/reduced tags)
RUN_TAG <- Sys.getenv("JASA_RUN_TAG", "bf_default")
RUN_DIR <- file.path(RESULTS_RUNS_ROOT, RUN_TAG)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

meta_path <- file.path(RUN_DIR, "run_metadata.txt")
cat(
  paste0("timestamp: ", Sys.time(), "\n"),
  paste0("run_tag: ", RUN_TAG, "\n"),
  paste0("DATA_DIR: ", DATA_DIR, "\n"),
  paste0("R.version: ", R.version.string, "\n"),
  "seed: 123\n",
  file = meta_path
)
capture.output(sessionInfo(), file = file.path(RUN_DIR, "sessionInfo.txt"))

# -------------------------------
# Inputs: which runs to compare
# -------------------------------
FULL_TAG <- get_env_str("JASA_FULL_TAG", "default")
REDUCED_TAG <- get_env_str("JASA_REDUCED_TAG", "reduced_default")

FULL_DIR <- file.path(RESULTS_RUNS_ROOT, FULL_TAG)
REDUCED_DIR <- file.path(RESULTS_RUNS_ROOT, REDUCED_TAG)

# Dimensions used in filenames (must match how you ran the samplers)
n_iter <- get_env_int("JASA_N_ITER", 50000)
kappa <- get_env_int("JASA_KAPPA", 5)

# data-implied n, d for building W_list and B
stopifnot(file.exists(DUCKS_CSV), file.exists(Z_CSV))
ducks <- read.csv(DUCKS_CSV)
y <- as.matrix(read.csv(Z_CSV, header = FALSE))

n <- nrow(y)
d <- ncol(y)

# B matrix for spline basis
B <- bs(seq(0, 1, length.out = n), df = kappa, intercept = TRUE)

# W_list (same as models)
W1 <- matrix(1, n, d)
W2 <- matrix(rep(ducks$duck_weight_male, each = n), nrow = n, ncol = d, byrow = FALSE)
W3 <- matrix(rep(ducks$duck_weight_female, each = n), nrow = n, ncol = d, byrow = FALSE)
W_list <- list(W1 = W1, W2 = W2, W3 = W3)

# Construct C utility
construct_C_from_ducks <- function(vec) {
  levels <- unique(vec)
  K_local <- length(levels)
  d_local <- length(vec)
  C <- matrix(0, nrow = K_local, ncol = d_local)
  for (j in 1:d_local) {
    k <- which(levels == vec[j])
    C[k, j] <- 1
  }
  list(C = C, s_list = rowSums(C), levels = levels)
}

# Full model: by species (K=7)
full_obj <- construct_C_from_ducks(ducks$duck_species)
C_full <- full_obj$C
s_list_full <- full_obj$s_list
K_full <- nrow(C_full)

# Reduced model: by grouped species (K=2)
ducks$duck_group <- ifelse(
  ducks$duck_species %in% c("American Black Duck", "Mallard", "Gadwall"),
  "Dabbling Duck",
  "Diving Duck"
)
reduced_obj <- construct_C_from_ducks(ducks$duck_group)
C_reduced <- reduced_obj$C
s_list_reduced <- reduced_obj$s_list
K_reduced <- nrow(C_reduced)

# Prior parameter used in BF computation
tau_rho <- get_env_dbl("JASA_TAU_RHO", 0.1)

# -------------------------------
# Load samples (full and reduced)
# -------------------------------
full_case_tag <- sprintf("duck_iter%d_n%d_kappa%d_K%d", n_iter, n, kappa, K_full)
reduced_case_tag <- sprintf("duck_reduced_iter%d_n%d_kappa%d_K%d", n_iter, n, kappa, K_reduced)

a_full_path <- file.path(FULL_DIR, paste0(full_case_tag, "_a_samples.rds"))
rho_full_path <- file.path(FULL_DIR, paste0(full_case_tag, "_rho_samples.rds"))
zeta_full_path <- file.path(FULL_DIR, paste0(full_case_tag, "_zeta_samples.rds"))

a_red_path <- file.path(REDUCED_DIR, paste0(reduced_case_tag, "_a_samples.rds"))
rho_red_path <- file.path(REDUCED_DIR, paste0(reduced_case_tag, "_rho_samples.rds"))
zeta_red_path <- file.path(REDUCED_DIR, paste0(reduced_case_tag, "_zeta_samples.rds"))

needed <- c(a_full_path, rho_full_path, zeta_full_path, a_red_path, rho_red_path, zeta_red_path)
missing <- needed[!file.exists(needed)]
if (length(missing) > 0) {
  cat("Missing required sample files:\n", paste0("  - ", missing, collapse = "\n"), "\n", sep = "")
  stop("BF.R requires a_samples, rho_samples, and zeta_samples from BOTH models. (Set JASA_SAVE_ZETA=true when running the samplers.)")
}

a_samples_full <- readRDS(a_full_path)
rho_samples_full <- readRDS(rho_full_path)
zeta_samples_full <- readRDS(zeta_full_path)

a_samples_reduced <- readRDS(a_red_path)
rho_samples_reduced <- readRDS(rho_red_path)
zeta_samples_reduced <- readRDS(zeta_red_path)

# -------------------------------
# Subsample indices for BF stability / speed
# -------------------------------
bf_start <- get_env_int("JASA_BF_START", 2000)
bf_end <- get_env_int("JASA_BF_END", n_iter)
bf_by <- get_env_int("JASA_BF_BY", 20)

idx <- seq(bf_start, bf_end, by = bf_by)
idx <- idx[idx >= 1 & idx <= nrow(a_samples_full)]  # safe clamp

a_samples_full <- a_samples_full[idx, , drop = FALSE]
a_samples_reduced <- a_samples_reduced[idx, , drop = FALSE]
rho_samples_full <- rho_samples_full[idx, , , drop = FALSE]
rho_samples_reduced <- rho_samples_reduced[idx, , , drop = FALSE]
zeta_samples_full <- zeta_samples_full[idx, , , drop = FALSE]
zeta_samples_reduced <- zeta_samples_reduced[idx, , , drop = FALSE]

cat(
  "\n---- BF settings ----\n",
  paste0("FULL_TAG: ", FULL_TAG, "\n"),
  paste0("REDUCED_TAG: ", REDUCED_TAG, "\n"),
  paste0("n_iter (filename): ", n_iter, "\n"),
  paste0("kappa: ", kappa, "\n"),
  paste0("K_full: ", K_full, "\n"),
  paste0("K_reduced: ", K_reduced, "\n"),
  paste0("bf_start: ", bf_start, "\n"),
  paste0("bf_end: ", bf_end, "\n"),
  paste0("bf_by: ", bf_by, "\n"),
  paste0("N_used: ", length(idx), "\n"),
  paste0("tau_rho: ", tau_rho, "\n"),
  file = meta_path,
  append = TRUE
)

# -------------------------------
# Core BF function (your existing logic, only wrapped)
# -------------------------------
log_BF_middle_calculator <- function(a_samples,
                                     rho_samples,
                                     zeta_samples,  # N x n x d
                                     s_list,        # length-K vector
                                     W_list,
                                     B, C, tau_rho) {
  N <- nrow(a_samples)
  K <- dim(rho_samples)[3]
  kappa <- dim(rho_samples)[2]
  d_rho <- kappa * K
  d_a <- ncol(a_samples)

  a_mean <- apply(a_samples, 2, mean)
  a_var <- apply(a_samples, 2, var) + 1e-8
  rho_mean <- apply(rho_samples, c(2, 3), mean)
  rho_var <- apply(rho_samples, c(2, 3), var) + 1e-8

  log_h <- numeric(N)
  log_posterior <- numeric(N)
  log_prior <- numeric(N)

  for (i in 1:N) {
    a_i <- a_samples[i, ]
    rho_i <- rho_samples[i, , ]   # kappa x K
    zeta_i <- zeta_samples[i, , ] # n x d

    mu_i <- a_i[1] * W_list[[1]] +
      a_i[2] * W_list[[2]] +
      a_i[3] * W_list[[3]] +
      B %*% rho_i %*% C

    # proposal density h (factorized normal approx)
    log_h[i] <- -0.5 * (
      d_a * log(2 * pi) + sum(log(a_var)) +
        d_rho * log(2 * pi) + sum(log(rho_var)) +
        sum((a_i - a_mean)^2 / a_var) +
        sum((rho_i - rho_mean)^2 / rho_var)
    )

    # likelihood term (up to constant): N(mu, I)
    log_posterior[i] <- -0.5 * sum((zeta_i - mu_i)^2)

    # prior for rho columns: N(0, (tau_rho * s_k)^{-1} I)
    log_prior_rho <- 0
    for (k in 1:K) {
      rho_k <- rho_i[, k]
      sk <- s_list[k]
      log_prior_rho <- log_prior_rho + (
        -0.5 * kappa * log(2 * pi) +
          0.5 * kappa * log(tau_rho * sk) -
          0.5 * tau_rho * sk * sum(rho_k^2)
      )
    }
    log_prior[i] <- log_prior_rho
  }

  # log-sum-exp trick
  log_summand <- log_h - log_posterior - log_prior
  max_log <- max(log_summand)
  max_log + log(sum(exp(log_summand - max_log)))
}

log_BF_num <- log_BF_middle_calculator(
  a_samples = a_samples_reduced,
  rho_samples = rho_samples_reduced,
  zeta_samples = zeta_samples_reduced,
  s_list = s_list_reduced,
  W_list = W_list,
  B = B,
  C = C_reduced,
  tau_rho = tau_rho
)

log_BF_den <- log_BF_middle_calculator(
  a_samples = a_samples_full,
  rho_samples = rho_samples_full,
  zeta_samples = zeta_samples_full,
  s_list = s_list_full,
  W_list = W_list,
  B = B,
  C = C_full,
  tau_rho = tau_rho
)

log_BF <- log_BF_den - log_BF_num
BF <- exp(log_BF)

cat("Log Bayes Factor:", round(log_BF, 3), "\n")
cat("Bayes Factor in favor of the reduced model:", round(BF, 3), "\n")

# Save a small table for reviewers
out_csv <- file.path(RESULTS_TABLES_ROOT, paste0("bayes_factor_", RUN_TAG, ".csv"))
out_rds <- file.path(RESULTS_TABLES_ROOT, paste0("bayes_factor_", RUN_TAG, ".rds"))

res <- data.frame(
  RUN_TAG = RUN_TAG,
  FULL_TAG = FULL_TAG,
  REDUCED_TAG = REDUCED_TAG,
  n_iter = n_iter,
  n = n,
  d = d,
  kappa = kappa,
  K_full = K_full,
  K_reduced = K_reduced,
  bf_start = bf_start,
  bf_end = bf_end,
  bf_by = bf_by,
  N_used = length(idx),
  tau_rho = tau_rho,
  log_BF = log_BF,
  BF = BF
)

write.csv(res, out_csv, row.names = FALSE)
saveRDS(res, out_rds)

cat("Saved BF results to:\n  ", out_csv, "\n  ", out_rds, "\n", sep = "")
