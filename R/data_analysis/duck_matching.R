# ------------------------------------------------------------
# duck_matching.R
# Repo scheme: run from repo root; read data from data/; write outputs to results/
# ------------------------------------------------------------

# Compile Rcpp dependencies (hit-and-run + TUM)
source("R/src/common/cpp_build.R")

suppressPackageStartupMessages({
  library(truncnorm)
  library(splines)
  library(lpSolve)
  library(lintools)
})

set.seed(123)

# -------------------------------
# Optional overrides via env vars
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


# -------------------------------
# Paths + outputs
# -------------------------------
DATA_DIR <- Sys.getenv("JASA_DATA_DIR", unset = file.path("data", "waterfowl_matching"))

# Use your current file names (no renaming required)
DUCKS_CSV <- file.path(DATA_DIR, "duck_data.csv")
A_CSV <- file.path(DATA_DIR, "A_tilde_matrix.csv")
Z_CSV <- file.path(DATA_DIR, "Z_matrix.csv")

# Output folder
RESULTS_ROOT <- file.path("results", "runs", "data_analysis")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)

RUN_TAG <- Sys.getenv("JASA_RUN_TAG", "default")
RUN_DIR <- file.path(RESULTS_ROOT, RUN_TAG)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

# Save metadata for reviewers
writeLines(c(
  paste0("timestamp: ", Sys.time()),
  paste0("run_tag: ", RUN_TAG),
  paste0("DATA_DIR: ", DATA_DIR),
  paste0("R.version: ", R.version.string),
  "seed: 123"
), con = file.path(RUN_DIR, "run_metadata.txt"))
capture.output(sessionInfo(), file = file.path(RUN_DIR, "sessionInfo.txt"))

# -------------------------------
# Load data
# -------------------------------
stopifnot(file.exists(DUCKS_CSV), file.exists(A_CSV), file.exists(Z_CSV))

ducks <- read.csv(DUCKS_CSV)
A <- as.matrix(read.csv(A_CSV, header = FALSE))
y <- as.matrix(read.csv(Z_CSV, header = FALSE))

b <- rep(1, nrow(A))


# -------------------------------
# Pre-MCMC
# -------------------------------
# Dimensions implied by data (more robust than hard-coding)
n <- nrow(y)         # Number of weeks (rows of Z)
d <- ncol(y)         # Number of edges (cols of Z)
m <- nrow(A)         # Number of constraints (rows of A)

# These depend on your covariate design; keep as before but allow override if needed
K <- get_env_int("JASA_K", 7)        # Number of groups/species
M <- 3                               # Length of alpha [DO NOT CHANGE]
kappa <- get_env_int("JASA_KAPPA", 5) # Degrees of freedom (basis functions)

# Construct C
construct_C_from_ducks <- function(species_vec) {
  species_levels <- unique(species_vec)  # preserve original order
  K_local <- length(species_levels)
  d_local <- length(species_vec)

  C <- matrix(0, nrow = K_local, ncol = d_local)
  for (j in 1:d_local) {
    k <- which(species_levels == species_vec[j])
    C[k, j] <- 1
  }

  list(C = C, s_list = rowSums(C), species_levels = species_levels)
}

result <- construct_C_from_ducks(ducks$duck_species)
C <- result$C
s_list <- result$s_list

# Sanity checks (fail early for reviewers)
stopifnot(nrow(C) == K, ncol(C) == d)
stopifnot(nrow(A) == m, ncol(A) == d)
stopifnot(length(b) == m)

# Construct W_list
W1 <- matrix(1, n, d)
W2 <- matrix(rep(ducks$duck_weight_male, each = n), nrow = n, ncol = d, byrow = FALSE)
W3 <- matrix(rep(ducks$duck_weight_female, each = n), nrow = n, ncol = d, byrow = FALSE)
W_list <- list(W1 = W1, W2 = W2, W3 = W3)

# Construct B (n x kappa)
B <- bs(seq(0, 1, length.out = n), df = kappa, intercept = TRUE)
write.csv(B, file = file.path(RUN_DIR, "B_matrix.csv"), row.names = FALSE) # save B for plotting in python

# Find whether U can be non-zero
U_free <- (t(A %*% t(y) == b)) * 1
U <- matrix(rexp(n * m), nrow = n, ncol = m) * U_free

# -------------------------------
# Specify prior parameters for alpha and rho
# -------------------------------
tau_a <- get_env_dbl("JASA_TAU_A", 0.1)
tau_rho <- get_env_dbl("JASA_TAU_RHO", 0.1)

# -------------------------------
# Precompute constant matrices for updates
# -------------------------------
B_star <- t(B) %*% B + diag(tau_rho, kappa)
inv_B_star <- solve(B_star)
inv_B_star_BT <- inv_B_star %*% t(B)

W_star <- matrix(0, M, M)
for (i in 1:M) {
  for (j in 1:M) {
    W_star[i, j] <- sum(W_list[[i]] * W_list[[j]])
  }
}
W_star <- W_star + diag(tau_a, M)
inv_W_star <- solve(W_star)

chol_inv_B_star <- t(chol(inv_B_star))
chol_inv_W_star <- t(chol(inv_W_star))

# -------------------------------
# Gibbs sampler settings (overridable)
# -------------------------------
n_iter <- get_env_int("JASA_N_ITER", 50000)
n_warmup <- get_env_int("JASA_N_WARMUP", 5000)  # if you use it later; harmless to define
thin <- get_env_int("JASA_N_THIN", 1)           # if you use thinning later

# Saving controls (important: zeta_samples is huge)
save_rho <- get_env_bool("JASA_SAVE_RHO", TRUE)
save_a <- get_env_bool("JASA_SAVE_A", TRUE)
save_zeta <- get_env_bool("JASA_SAVE_ZETA", FALSE) # default FALSE because it's enormous

# Pre-allocate draws (only what we intend to save)
rho_samples <- if (save_rho) array(NA, dim = c(n_iter, kappa, K)) else NULL
a_samples <- if (save_a) array(NA, dim = c(n_iter, M)) else NULL
zeta_samples <- if (save_zeta) array(NA, dim = c(n_iter, n, d)) else NULL

# Initial values
rho <- matrix(0, nrow = kappa, ncol = K)
a <- rnorm(M)

# Initialize zeta
zeta <- matrix(NA, n, d)
UA <- U %*% A
W_sum <- Reduce(`+`, lapply(1:M, function(mm) a[mm] * W_list[[mm]]))
B_rho_C <- B %*% rho %*% C
mu <- W_sum + B_rho_C

for (j in 1:d) {
  lower_bound <- ifelse(y[, j] == 1, UA[, j], -Inf)
  upper_bound <- ifelse(y[, j] == 1, Inf, UA[, j])
  zeta[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu[, j], sd = 1)
}

# Record effective settings for this run (append to your metadata file)
meta_path <- file.path(RUN_DIR, "run_metadata.txt")

cat(
  "\n---- duck_matching settings ----\n",
  paste0("n: ", n, "\n"),
  paste0("d: ", d, "\n"),
  paste0("m: ", m, "\n"),
  paste0("K: ", K, "\n"),
  paste0("kappa: ", kappa, "\n"),
  paste0("n_iter: ", n_iter, "\n"),
  paste0("n_warmup: ", n_warmup, "\n"),
  paste0("thin: ", thin, "\n"),
  paste0("tau_a: ", tau_a, "\n"),
  paste0("tau_rho: ", tau_rho, "\n"),
  paste0("save_rho: ", save_rho, "\n"),
  paste0("save_a: ", save_a, "\n"),
  paste0("save_zeta: ", save_zeta, "\n"),
  file = meta_path,
  append = TRUE
)



# -------------------------------
# Gibbs sampler iterations
# -------------------------------
for (iter in 1:n_iter) {
  # Keep track of the process
  if (iter %% 100 == 0){
    cat("MCMC iteration:", iter, "; current AR:", mean(accept_or_not), "\n")
  }

  greaterequal = 1-y # if y=1, (UA)< zeta, hence greaterequal == 1-y
  U <- loop_hit_and_run(t(A),zeta,
                        greaterequal,U,
                        U_free, n_iter = 100)
  UA = U%*%A

  # Step 1: Sample latent variables zeta given beta and y (vectorized over observations)
  zeta_new <- zeta # matrix(NA, n, d)

  # Random subset of [d]
  J = d %/% 4
  sampled_indices <- sample(1:d, J, replace = FALSE)
  # sampled_indices <- 1:d
  for (j in sampled_indices) {
    # Set vectorized lower and upper bounds based on observed y
    lower_bound <- ifelse(y[, j] == 1, UA[,j], -Inf)
    upper_bound <- ifelse(y[, j] == 1, Inf, UA[,j])
    # Sample all n latent variables for outcome j in one call
    zeta_new[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu[,j], sd = 1)
  }


  zeta_tilde <- (y>0.5) * pmax(zeta,zeta_new) + (y<0.5) * pmin(zeta,zeta_new)
  U_star <- loop_hit_and_run(t(A),zeta_tilde,
                             greaterequal,U,
                             U_free, n_iter = 100)

  accept_or_not <- check_feasible(t(A),zeta,
                                  greaterequal,U_star)

  # print(mean(accept_or_not))


  zeta <- zeta_new * accept_or_not + zeta * (1-accept_or_not)
  
  # sum(abs((zeta>=U%*%A)-y))
  
  # Step 2: Sample a and rho 
  
  # sample rho
  Q <- inv_B_star_BT %*% (zeta - W_sum)
  
  rho <- matrix(0, kappa, K)
  col_start <- 1
  for (k in 1:K) {
    col_end <- col_start + s_list[k] - 1
    Q_k <- rowMeans(Q[, col_start:col_end, drop = FALSE])
    noise <- chol_inv_B_star %*% rnorm(kappa) / sqrt(s_list[k])
    rho[, k] <- Q_k + noise
    col_start <- col_end + 1
  }
  
  # sample a
  diff <- zeta - B_rho_C
  gamma <- sapply(1:M, function(m) sum(W_list[[m]] * diff))
  a <- inv_W_star %*% gamma + chol_inv_W_star %*% rnorm(M)
  
  
  # Update W_sum, B_rho_C, and mu
  W_sum <- Reduce(`+`, lapply(1:M, function(m) a[m] * W_list[[m]]))
  B_rho_C <- B %*% rho %*% C
  mu <- W_sum + B_rho_C
  
  
  # Store samples
  if (save_rho) rho_samples[iter, , ] <- rho
  if (save_a) a_samples[iter, ] <- a
  if (save_zeta) zeta_samples[iter, , ] <- zeta

}


# -------------------------------
# Save results
# -------------------------------
case_tag <- sprintf("duck_iter%d_n%d_kappa%d_K%d", n_iter, n, kappa, K)

if (save_rho) {
  saveRDS(rho_samples, file = file.path(RUN_DIR, paste0(case_tag, "_rho_samples.rds")))
}
if (save_a) {
  saveRDS(a_samples, file = file.path(RUN_DIR, paste0(case_tag, "_a_samples.rds")))
}
if (save_zeta) {
  saveRDS(zeta_samples, file = file.path(RUN_DIR, paste0(case_tag, "_zeta_samples.rds")))
}

cat("Saved results under: ", RUN_DIR, "\n", sep = "")






# # -------------------------------
# # Post-processing: summary of beta samples
# # -------------------------------

# burn_in <- n_iter %/% 2

# # Posterior means
# rho_post_mean <- apply(rho_samples[(burn_in + 1):n_iter, , ], c(2, 3), mean)
# a_post_mean   <- colMeans(a_samples[(burn_in + 1):n_iter, ])

# # Print
# cat("Posterior mean of rho (kappa x K matrix):\n")
# print(rho_post_mean)


# cat("Posterior mean of a (length M vector):\n")
# print(a_post_mean)




# # Trace plots
# max_p <- min(p, 5)
# max_K <- min(K, 5)

# par(mfrow = c(max_p, max_K))
# for (j in 1:max_p) {
#   for (k in 1:max_K) {
#     plot(rho_samples[(burn_in + 1):n_iter, j, k], type = 'l',
#          main = paste0("Trace: rho[", j, ",", k, "]"),
#          xlab = "Iteration", ylab = paste0("rho[", j, ",", k, "]"))
#   }
# }


# par(mfrow = c(1, M))
# for (j in 1:M) {
#   plot(a_samples[(burn_in + 1):n_iter, j], type = 'l',
#        main = paste0("Trace: a[", j, "]"),
#        xlab = "Iteration", ylab = paste0("a[", j, "]"))
# }


# thin <- 50
# thinned_indices <- seq(from = burn_in + 1, to = n_iter, by = thin)

# # ACF for rho
# par(mfrow = c(max_p, max_K))
# for (j in 1:max_p) {
#   for (k in 1:max_K) {
#     acf(rho_samples[thinned_indices, j, k],
#         main = paste("ACF: rho[", j, ",", k, "]"))
#   }
# }

# # ACF for a
# par(mfrow = c(1, M))
# for (j in 1:M) {
#   acf(a_samples[thinned_indices, j],
#       main = paste("ACF: a[", j, "]"))
# }














