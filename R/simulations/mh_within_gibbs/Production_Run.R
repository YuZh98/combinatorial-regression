# ------------------------------------------------------------
# Production_Run.R
# Repo scheme: run from repo root; write outputs to results/
# ------------------------------------------------------------

# ============================================================
# SETUP
# ============================================================

# Compile Rcpp code (hit-and-run + TUM)
source("R/src/common/cpp_build.R")

suppressPackageStartupMessages({
  library(truncnorm)
  library(lpSolve)
  library(lintools)
})

set.seed(123)


# ============================================================
# ENVIRONMENT VARIABLE PARSING UTILITIES
# ============================================================

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


# ============================================================
# OUTPUT DIRECTORY SETUP
# ============================================================

RESULTS_ROOT <- file.path("results", "runs", "mh_within_gibbs")
dir.create(RESULTS_ROOT, recursive = TRUE, showWarnings = FALSE)

RUN_TAG <- Sys.getenv("JASA_RUN_TAG", "default")
RUN_DIR <- file.path(RESULTS_ROOT, RUN_TAG)
dir.create(RUN_DIR, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# CONFIGURATION
# ============================================================

config <- list(
  # Data dimensions
  n = get_env_int("JASA_N", 1000),
  p = get_env_int("JASA_P", 5),
  d_list = get_env_int_list("JASA_D_LIST", c(2, 5, 10, 20)),
  m_list = get_env_int_list("JASA_M_LIST", c(1, 2, 5, 10)),
  n_rep = get_env_int("JASA_N_REP", 1),
  
  # Kernel methods
  method_list = get_env_str_list("JASA_METHODS", c("exponential")),
  
  # MCMC parameters
  n_iter = get_env_int("JASA_N_ITER", 5000),
  n_warmup = get_env_int("JASA_N_WARMUP", 2000),
  n_thin = get_env_int("JASA_N_THIN", 25),
  n_iter_hit_and_run = get_env_int("JASA_N_HAR", 100),
  
  # Output options
  button_save = get_env_bool("JASA_SAVE", TRUE),
  button_plot = get_env_bool("JASA_PLOT", TRUE)
)


# ============================================================
# CONFIGURATION LOGGING
# ============================================================

print_configuration <- function(cfg) {
  cat("\n")
  cat("============================================================\n")
  cat("CONFIGURATION SUMMARY\n")
  cat("============================================================\n")
  cat("Data dimensions:\n")
  cat("  n =", cfg$n, "\n")
  cat("  p =", cfg$p, "\n")
  cat("  d_list =", paste(cfg$d_list, collapse = ", "), "\n")
  cat("  m_list =", paste(cfg$m_list, collapse = ", "), "\n")
  cat("  n_rep =", cfg$n_rep, "\n")
  cat("\n")
  cat("MCMC settings:\n")
  cat("  n_iter =", cfg$n_iter, "\n")
  cat("  n_warmup =", cfg$n_warmup, "\n")
  cat("  n_thin =", cfg$n_thin, "\n")
  cat("  n_iter_hit_and_run =", cfg$n_iter_hit_and_run, "\n")
  cat("\n")
  cat("U-kernel configuration:\n")
  cat("  Kernels used for u update:", paste(cfg$method_list, collapse = ", "), "\n")
  
  if (length(cfg$method_list) > 1) {
    cat("  NOTE: Multiple kernels selected — this corresponds to supplementary comparison.\n")
  } else {
    cat("  NOTE: Single kernel selected — this corresponds to main paper results.\n")
  }
  
  cat("\n")
  cat("Output options:\n")
  cat("  Save results:", cfg$button_save, "\n")
  cat("  Generate plots:", cfg$button_plot, "\n")
  cat("  Output directory:", RUN_DIR, "\n")
  cat("============================================================\n")
  cat("\n")
}

print_configuration(config)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

L2 <- function(v) sqrt(sum(v^2))

RMSE <- function(v, v_true) sqrt(mean((v - v_true)^2))


# ============================================================
# DATA SIMULATION
# ============================================================

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


# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

trace_plot <- function(beta_samples, p, d, n_warmup, n_iter, beta_true) {
  row_index <- 1:min(p, 2)
  col_index <- 1:min(d, 2)
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
  row_index <- 1:min(p, 2)
  col_index <- 1:min(d, 2)
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


# ============================================================
# MH-WITHIN-GIBBS SAMPLER
# ============================================================

run_mhwg_sampler <- function(sim_data, cfg, method) {
  # Alias
  if (method == "halfgaussian") method = "half_gaussian"

  # Extract simulation data
  A <- sim_data$A
  X <- sim_data$X
  y <- sim_data$y
  beta_true <- sim_data$beta_true
  U_free <- sim_data$U_free
  
  n <- nrow(X)
  p <- ncol(X)
  d <- ncol(y)
  m <- nrow(A)
  
  # Prior setup and pre-compute fixed variables
  B0 <- 1
  V_mat <- solve(diag(1 / B0, p) + t(X) %*% X)
  L_mat <- t(chol(V_mat))
  greaterequal <- 1 - y
  
  # Initialize state
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
  
  # Storage for samples
  beta_samples <- array(NA, dim = c(cfg$n_iter, p, d))
  
  # Run MCMC
  cat("=== Running method:", method, "===\n")
  
  running_time <- system.time({
    for (iter in 1:cfg$n_iter) {
      rmse_beta <- RMSE(beta, beta_true)
      
      # Logging
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
        n_iter = cfg$n_iter_hit_and_run, kernel = method
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
        n_iter = cfg$n_iter_hit_and_run, kernel = method
      )
      
      accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)
      zeta <- zeta_new * accept_or_not + zeta * (1 - accept_or_not)
      
      # ----- Update beta -----
      mean_normal <- V_mat %*% t(X) %*% zeta
      beta <- mean_normal + L_mat %*% matrix(rnorm(p * d), nrow = p, ncol = d)
      
      beta_samples[iter, , ] <- beta
    }
  })
  
  cat(cfg$n_iter, " iterations of MCMC with ", cfg$n_warmup, " warm-ups completed.\n", sep = "")
  print(running_time)
  
  # Post-processing
  beta_post_mean <- apply(beta_samples[(cfg$n_warmup + 1):cfg$n_iter, , ], c(2, 3), mean)
  rmse_beta <- RMSE(beta_post_mean, beta_true)
  cat("RMSE(beta): ", rmse_beta, "\n", sep = "")
  
  list(
    beta_samples = beta_samples,
    beta_true = beta_true,
    rmse = rmse_beta,
    elapsed_time = running_time[3],
    cpu_time = running_time[1]
  )
}


# ============================================================
# RESULTS SAVING FUNCTIONS
# ============================================================

save_case_results <- function(results, method, repetition, cfg, n, d, p, m) {
  case_tag <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d", 
                      repetition, cfg$n_iter, n, d, p, m)
  
  if (method == "exponential") {
    kernel_label <- "kernelexponential"
  } else if (method == "halfgaussian") {
    kernel_label <- "kernelhalfgaussian"
  } else {
    kernel_label <- paste0("kernel", method)
  }
  
  saveRDS(results$beta_samples, 
          file = file.path(RUN_DIR, paste0(case_tag, "__", kernel_label, "__beta_samples.rds")))
  saveRDS(results$beta_true, 
          file = file.path(RUN_DIR, paste0(case_tag, "__", kernel_label, "__beta_true.rds")))
}

initialize_result_tables <- function(cfg) {
  max_m <- max(cfg$m_list)
  max_d <- max(cfg$d_list)
  n_rep <- cfg$n_rep
  
  list(
    RMSE_table_exponential = array(NA, dim = c(n_rep, max_m, max_d)),
    RMSE_table_halfgaussian = array(NA, dim = c(n_rep, max_m, max_d)),
    runtime_table_exponential = array(NA, dim = c(n_rep, max_m, max_d)),
    runtime_table_halfgaussian = array(NA, dim = c(n_rep, max_m, max_d)),
    cputime_table_exponential = array(NA, dim = c(n_rep, max_m, max_d)),
    cputime_table_halfgaussian = array(NA, dim = c(n_rep, max_m, max_d))
  )
}

save_summary_tables <- function(tables) {
  saveRDS(tables$RMSE_table_exponential, 
          file = file.path(RUN_DIR, "RMSE_table_exponential.rds"))
  saveRDS(tables$runtime_table_exponential, 
          file = file.path(RUN_DIR, "runtime_table_exponential.rds"))
  saveRDS(tables$cputime_table_exponential, 
          file = file.path(RUN_DIR, "cputime_table_exponential.rds"))
  
  saveRDS(tables$RMSE_table_halfgaussian, 
          file = file.path(RUN_DIR, "RMSE_table_halfgaussian.rds"))
  saveRDS(tables$runtime_table_halfgaussian, 
          file = file.path(RUN_DIR, "runtime_table_halfgaussian.rds"))
  saveRDS(tables$cputime_table_halfgaussian, 
          file = file.path(RUN_DIR, "cputime_table_halfgaussian.rds"))
}


# ============================================================
# MAIN EXECUTION LOOP
# ============================================================

# Initialize result tables
result_tables <- initialize_result_tables(config)

# Extract convenience variables
n <- config$n
p <- config$p
d_list <- config$d_list
m_list <- config$m_list
n_rep <- config$n_rep
method_list <- config$method_list

# Main loop over all cases
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
        
        # Simulate data for this case
        sim_data <- simulate_data(n, p, d, m)
        
        # Run sampler for each method
        for (method in method_list) {
          if ((method == "exponential") || (method == "halfgaussian")) {
            # Run MH-within-Gibbs sampler
            results <- run_mhwg_sampler(sim_data, config, method)
            
            # Store results in tables
            if (method == "exponential") {
              result_tables$RMSE_table_exponential[repetition, m, d] <- results$rmse
              result_tables$runtime_table_exponential[repetition, m, d] <- results$elapsed_time
              result_tables$cputime_table_exponential[repetition, m, d] <- results$cpu_time
            } else if (method == "halfgaussian") {
              result_tables$RMSE_table_halfgaussian[repetition, m, d] <- results$rmse
              result_tables$runtime_table_halfgaussian[repetition, m, d] <- results$elapsed_time
              result_tables$cputime_table_halfgaussian[repetition, m, d] <- results$cpu_time
            }

            # ----------------------------------------
            # Save case-specific results
            # ----------------------------------------

            if (config$button_save) {
              tryCatch(
                {
                  save_case_results(results, method, repetition, config, n, d, p, m)
                },
                error = function(e) {
                  warning(sprintf(
                    "Saving failed (method=%s, rep=%d, d=%d, m=%d): %s",
                    method, repetition, d, m, conditionMessage(e)
                  ))
                }
              )
            }

            # ----------------------------------------
            # Generate plots
            # ----------------------------------------

            if (config$button_plot) {
              tryCatch(
                {
                  trace_plot(
                    results$beta_samples,
                    p, d,
                    config$n_warmup,
                    config$n_iter,
                    results$beta_true
                  )
                  
                  ACF_plot(
                    results$beta_samples,
                    p, d,
                    config$n_warmup,
                    config$n_iter,
                    thin = config$n_thin
                  )
                },
                error = function(e) {
                  warning(sprintf(
                    "Plotting failed (method=%s, rep=%d, d=%d, m=%d): %s",
                    method, repetition, d, m, conditionMessage(e)
                  ))
                }
              )
            }
          }
        }
      }
    }
  }
}


# ============================================================
# SAVE SUMMARY TABLES AND PRINT RESULTS
# ============================================================

save_summary_tables(result_tables)

# ----------------------------------------
# Print summary tables (clearly labeled)
# ----------------------------------------

for (method in config$method_list) {
  
  rmse_name    <- paste0("RMSE_table_", method)
  runtime_name <- paste0("runtime_table_", method)
  cputime_name <- paste0("cputime_table_", method)
  
  cat("\n####################################################\n")
  cat("Kernel:", method, "\n")
  for (r in 1:n_rep) {
    
    cat("\n--- Repetition", r, "---------------------------------\n")
    
    cat("\nRMSE(beta posterior mean vs truth)\n")
    cat("Rows: m in", paste(m_list, collapse = ", "), "\n")
    cat("Cols: d in", paste(d_list, collapse = ", "), "\n")
    print(result_tables[[rmse_name]][r, m_list, d_list])
    
    cat("\nWall-clock runtime (seconds)\n")
    print(result_tables[[runtime_name]][r, m_list, d_list])
    
    cat("\nCPU time (seconds)\n")
    print(result_tables[[cputime_name]][r, m_list, d_list])
  }
}


cat("Done. Outputs saved under: ", RUN_DIR, "\n", sep = "")