# ------------------------------------------------------------
# Production_Run.R
# Repo scheme: run from repo root; write outputs to results/
# ------------------------------------------------------------

# Compile Rcpp code (hit-and-run + TUM)
source("R/src/common/cpp_build.R")

suppressPackageStartupMessages({
  library(truncnorm)
  library(lpSolve)
  library(lintools)
})

# -------------------------------
# Reproducibility / paths
# -------------------------------
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

get_env_int_list <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  # comma or space separated
  parts <- unlist(strsplit(x, "[, ]+"))
  parts <- parts[parts != ""]
  as.integer(parts)
}

get_env_str_list <- function(name, default) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  parts <- unlist(strsplit(x, "[, ]+"))
  parts <- parts[parts != ""]
  parts
}



# Assume you run from repo root. All outputs go under results/
RESULTS_ROOT <- file.path("results", "runs", "mh_within_gibbs")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)

# If you want separate outputs per run type (e.g., smoke/full), set:
RUN_TAG <- Sys.getenv("JASA_RUN_TAG", "default")
RUN_DIR <- file.path(RESULTS_ROOT, RUN_TAG)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)

# -------------------------------
# Global
# -------------------------------
n <- get_env_int("JASA_N", 1000)
p <- get_env_int("JASA_P", 5)
d_list <- get_env_int_list("JASA_D_LIST", c(2, 5, 10, 20, 50, 100, 200, 500, 1000))
m_list <- get_env_int_list("JASA_M_LIST", c(1, 2, 5, 10, 20, 50, 100))
n_rep <- get_env_int("JASA_N_REP", 5)

method_list <- get_env_str_list("JASA_METHODS", c("exponential"))   # can be half_gaussian or exponential or both
cat("\n---- U-kernel configuration ----\n")
cat("Kernels used for u update: ", paste(method_list, collapse = ", "), "\n")

if (length(method_list) > 1) {
  cat("NOTE: Multiple kernels selected — this corresponds to supplementary comparison.\n")
} else {
  cat("NOTE: Single kernel selected — this corresponds to main paper results.\n")
}
cat("----------------------------------\n\n")


n_iter <- get_env_int("JASA_N_ITER", 50000)
n_warmup <- get_env_int("JASA_N_WARMUP", 5000)
n_thin <- get_env_int("JASA_N_THIN", 25)
n_iter_hit_and_run <- get_env_int("JASA_N_HAR", 100)

button_save <- get_env_bool("JASA_SAVE", FALSE)
button_plot <- get_env_bool("JASA_PLOT", FALSE)


# -------------------------------
# Utility functions
# -------------------------------
L2 <- function(v) sqrt(sum(v^2))

RMSE <- function(v, v_true) sqrt(mean((v - v_true)^2))

simulate_data <- function(n, p, d, m) {
  if ((d * m <= 50) || (m < 2)) {
    A <- round(matrix(runif(m * d, -1, 1), m, d))
    while ((!is_totally_unimodular(A)) || any(rowSums(abs(A)) == 0)) {
      A <- round(matrix(runif(m * d, -1, 1), m, d))
    }
    b <- rep(1, m)
  } else {
    # Generate A as a random incidence matrix
    A <- matrix(0, m, d)
    for (i in 1:m) {
      ind <- sample.int(d, 2, replace = FALSE)
      A[i, ind[1]] <- 1
      A[i, ind[2]] <- -1
    }
    b <- sample(0:1, m, replace = TRUE)
  }

  # X: n x p
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)

  beta_true <- matrix(rnorm(p * d), nrow = p, ncol = d)
  zeta_true <- X %*% beta_true + matrix(rnorm(n * d), nrow = n, ncol = d)

  ilp_map <- function(c) {
    constraint_directions <- rep("<=", length(b))
    lp_result <- lp(
      direction = "max",
      objective.in = c,
      const.mat = A,
      const.dir = constraint_directions,
      const.rhs = b,
      all.bin = TRUE
    )
    lp_result$solution
  }

  y <- t(apply(zeta_true, 1, ilp_map))
  U_free <- (t(A %*% t(y) == b)) * 1

  list(A = A, b = b, X = X, y = y, beta_true = beta_true, U_free = U_free)
}

# Plotting helpers (optional; base R)
trace_plot <- function(beta_samples, p, d, n_warmup, n_iter, beta_true) {
  row_index <- 1:min(p, 5)
  col_index <- 1:min(d, 5)
  par(mfrow = c(length(row_index), length(col_index)))
  for (k in row_index) {
    for (j in col_index) {
      plot(
        beta_samples[(n_warmup + 1):n_iter, k, j],
        type = "l",
        main = paste0("Trace for beta[", k, ",", j, "]"),
        xlab = "Iteration",
        ylab = paste0("beta[", k, ",", j, "]")
      )
      abline(h = beta_true[k, j], col = "red")
    }
  }
}

ACF_plot <- function(beta_samples, p, d, n_warmup, n_iter, thin = 25) {
  row_index <- 1:min(p, 5)
  col_index <- 1:min(d, 5)
  par(mfrow = c(length(row_index), length(col_index)))
  thinned_indices <- seq(from = n_warmup + 1, to = n_iter, by = thin)
  for (k in row_index) {
    for (j in col_index) {
      acf(
        beta_samples[thinned_indices, k, j],
        main = paste0("ACF for beta[", k, ",", j, "] (thin=", thin, ")")
      )
    }
  }
}

# -------------------------------
# Run all cases
# -------------------------------
RMSE_table_exponential <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))
RMSE_table_halfgaussian <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))
runtime_table_exponential <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))
runtime_table_halfgaussian <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))
cputime_table_exponential <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))
cputime_table_halfgaussian <- array(NA, dim = c(n_rep, max(m_list), max(d_list)))

for (d in d_list) {
  for (m in m_list) {
    if (m < d) {
      for (repetition in 1:n_rep) {
        cat(
          "=============== Running case (n,p,d,m)=(",
          n, ",", p, ",", d, ",", m,
          "); rep ", repetition,
          " ===============\n",
          sep = ""
        )

        sim_data <- simulate_data(n, p, d, m)
        A <- sim_data$A
        X <- sim_data$X
        y <- sim_data$y
        beta_true <- sim_data$beta_true
        U_free <- sim_data$U_free

        # Prior setup and pre-compute fixed variables
        B0 <- 1
        V_mat <- solve(diag(1 / B0, p) + t(X) %*% X)
        L_mat <- t(chol(V_mat))
        greaterequal <- 1 - y

        for (method in method_list) {
          # Initialization
          beta <- matrix(0, nrow = p, ncol = d)
          U <- matrix(rexp(n * m), nrow = n, ncol = m) * U_free
          UA <- U %*% A

          zeta <- matrix(NA, nrow = n, ncol = d)
          for (j in 1:d) {
            mu_j <- X %*% beta[, j]
            lower_bound <- ifelse(y[, j] == 1, UA[, j], -Inf)
            upper_bound <- ifelse(y[, j] == 1, Inf, UA[, j])
            zeta[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu_j, sd = 1)
          }

          if ((method == "exponential") || (method == "half_gaussian")) {
            cat("=== running method: ", method, " ===\n", sep = "")
            ker <- method

            beta_samples <- array(NA, dim = c(n_iter, p, d))

            running_time <- system.time({
              for (iter in 1:n_iter) {
                rmse_beta <- RMSE(beta, beta_true)

                if ((iter < 5) || (iter %% 1000 == 0)) {
                  if (iter > 1) {
                    cat(
                      "MCMC iteration ", iter,
                      ": RMSE(beta): ", rmse_beta,
                      "; current AR: ", mean(accept_or_not),
                      "\n",
                      sep = ""
                    )
                  } else {
                    cat(
                      "MCMC iteration ", iter,
                      ": RMSE(beta): ", rmse_beta,
                      "; U shape: (", nrow(U), ", ", ncol(U), ")\n",
                      sep = ""
                    )
                  }
                }

                # ----- Update zeta -----
                U <- loop_hit_and_run_multi_kernel(
                  t(A), zeta, greaterequal, U, U_free,
                  n_iter = n_iter_hit_and_run, kernel = ker
                )
                UA <- U %*% A

                zeta_new <- zeta
                J <- min(d, 100)
                sampled_indices <- sample.int(d, J, replace = FALSE)
                for (j in sampled_indices) {
                  mu_j <- X %*% beta[, j]
                  lower_bound <- ifelse(y[, j] == 1, UA[, j], -Inf)
                  upper_bound <- ifelse(y[, j] == 1, Inf, UA[, j])
                  zeta_new[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu_j, sd = 1)
                }

                zeta_tilde <- (y > 0.5) * pmax(zeta, zeta_new) + (y < 0.5) * pmin(zeta, zeta_new)
                U_star <- loop_hit_and_run_multi_kernel(
                  t(A), zeta_tilde, greaterequal, U, U_free,
                  n_iter = n_iter_hit_and_run, kernel = ker
                )

                accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)
                zeta <- zeta_new * accept_or_not + zeta * (1 - accept_or_not)

                # ----- Update beta -----
                mean_normal <- V_mat %*% t(X) %*% zeta
                beta <- mean_normal + L_mat %*% matrix(rnorm(p * d), nrow = p, ncol = d)

                beta_samples[iter, , ] <- beta
              }
            })

            cat(n_iter, "iterations of MCMC with ", n_warmup, " warm-ups completed.\n", sep = "")
            print(running_time)

            # ----- Post-processing -----
            beta_post_mean <- apply(beta_samples[(n_warmup + 1):n_iter, , ], c(2, 3), mean)
            rmse_beta <- RMSE(beta_post_mean, beta_true)
            cat("RMSE(beta): ", rmse_beta, "\n", sep = "")

            # Save per-case results into RUN_DIR (no setwd)
            if (button_save) {
              case_tag <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d", repetition, n_iter, n, d, p, m)

              if (method == "exponential") {
                RMSE_table_exponential[repetition, m, d] <- rmse_beta
                runtime_table_exponential[repetition, m, d] <- running_time[3]
                cputime_table_exponential[repetition, m, d] <- running_time[1]

                saveRDS(beta_samples, file = file.path(RUN_DIR, paste0(case_tag, "__kernelexponential__beta_samples.rds")))
                saveRDS(beta_true,   file = file.path(RUN_DIR, paste0(case_tag, "__kernelexponential__beta_true.rds")))
              } else if (method == "half_gaussian") {
                RMSE_table_halfgaussian[repetition, m, d] <- rmse_beta
                runtime_table_halfgaussian[repetition, m, d] <- running_time[3]
                cputime_table_halfgaussian[repetition, m, d] <- running_time[1]

                saveRDS(beta_samples, file = file.path(RUN_DIR, paste0(case_tag, "__kernelhalfgaussian__beta_samples.rds")))
                saveRDS(beta_true,   file = file.path(RUN_DIR, paste0(case_tag, "__kernelhalfgaussian__beta_true.rds")))
              }
            }

            if (button_plot) {
              trace_plot(beta_samples, p, d, n_warmup, n_iter, beta_true)
              ACF_plot(beta_samples, p, d, n_warmup, n_iter, thin = n_thin)
            }
          }
        }
      }
    }
  }
}

# Save summary tables (if button_save is TRUE, but saving tables is typically always useful)
# Here we always save tables to RUN_DIR to avoid losing results.
saveRDS(RMSE_table_exponential, file = file.path(RUN_DIR, "RMSE_table_exponential.rds"))
saveRDS(runtime_table_exponential, file = file.path(RUN_DIR, "runtime_table_exponential.rds"))
saveRDS(cputime_table_exponential, file = file.path(RUN_DIR, "cputime_table_exponential.rds"))

saveRDS(RMSE_table_halfgaussian, file = file.path(RUN_DIR, "RMSE_table_halfgaussian.rds"))
saveRDS(runtime_table_halfgaussian, file = file.path(RUN_DIR, "runtime_table_halfgaussian.rds"))
saveRDS(cputime_table_halfgaussian, file = file.path(RUN_DIR, "cputime_table_halfgaussian.rds"))

# Optional prints (kept as in original intent)
for (r in 1:n_rep) {
  print(RMSE_table_exponential[r, m_list, d_list])
  print(runtime_table_exponential[r, m_list, d_list])
  print(cputime_table_exponential[r, m_list, d_list])
}
for (r in 1:n_rep) {
  print(RMSE_table_halfgaussian[r, m_list, d_list])
  print(runtime_table_halfgaussian[r, m_list, d_list])
  print(cputime_table_halfgaussian[r, m_list, d_list])
}

cat("Done. Outputs saved under: ", RUN_DIR, "\n", sep = "")
