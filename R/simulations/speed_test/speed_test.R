# ------------------------------------------------------------
# speed_test.R
# Benchmark MH-within-Gibbs sampler: v1 (current) vs v2 (revised).
# Run from repo root: Rscript R/simulations/speed_test/speed_test.R
#
# Strategy:
#   1. Compile both Rcpp implementations.
#   2. Define run_mhwg_sampler(..., sampler_fn = ...) — identical outer loop,
#      sampler function injected.
#   3. For each (n, d, m, n_iter_har) cell, simulate ONE dataset, then run
#      both versions back-to-back with the same RNG seed for state init.
#      RNG paths inside samplers will diverge (different rejection logic),
#      so beta posterior means are compared statistically, not bit-wise.
#   4. Repeat each cell `n_rep` times. Report median wall + CPU time and
#      posterior-mean RMSE for both versions.
#
# Env vars:
#   ST_N_ITER       outer Gibbs iterations  (default 200)
#   ST_N_WARMUP     warmup                  (default 50)
#   ST_N_HAR        hit-and-run inner iters (default 100)
#   ST_N_REP        replicates per cell     (default 3)
#   ST_GRID         "small" | "medium"      (default "small")
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(Rcpp)
  library(truncnorm)
  library(lpSolve)
  library(lintools)
})

# ============================================================
# COMPILE BOTH IMPLEMENTATIONS
# ============================================================

cat("Compiling v1 (current)...\n")
sourceCpp("R/src/cpp/hit_and_run_augmented.cpp")

cat("Compiling v2 (revised)...\n")
sourceCpp("R/src/cpp/hit_and_run_augmented_v2.cpp")

sourceCpp("R/src/cpp/tum_check.cpp")


# ============================================================
# CONFIGURATION
# ============================================================

get_env <- function(name, default, parser = as.character) {
  x <- Sys.getenv(name, unset = "")
  if (x == "") return(default)
  parser(x)
}

N_ITER    <- get_env("ST_N_ITER",   200, as.integer)
N_WARMUP  <- get_env("ST_N_WARMUP", 50,  as.integer)
N_HAR     <- get_env("ST_N_HAR",    100, as.integer)
N_REP     <- get_env("ST_N_REP",    3,   as.integer)
GRID      <- get_env("ST_GRID",     "small")

if (GRID == "small") {
  cells <- list(
    list(n =  500, p = 5, d = 10, m = 2),
    list(n = 1000, p = 5, d = 10, m = 2),
    list(n = 1000, p = 5, d = 20, m = 5)
  )
} else if (GRID == "medium") {
  cells <- list(
    list(n = 1000, p = 5, d = 10, m = 2),
    list(n = 1000, p = 5, d = 20, m = 5),
    list(n = 2000, p = 5, d = 20, m = 5),
    list(n = 2000, p = 5, d = 40, m = 10)
  )
} else if (GRID == "large") {
  # Push (d, m) higher to find where wall_speedup approaches cpu_speedup.
  # m/d ratio held at 0.25 (matches existing grid). n fixed at 2000.
  cells <- list(
    list(n = 2000, p = 5, d =  60, m = 15),   # m*d =  900
    list(n = 2000, p = 5, d =  80, m = 20),   # m*d = 1600
    list(n = 2000, p = 5, d = 120, m = 30),   # m*d = 3600
    list(n = 2000, p = 5, d = 160, m = 40)    # m*d = 6400
  )
} else {
  stop(sprintf("Unknown ST_GRID: %s", GRID))
}

cat(sprintf("Config: n_iter=%d n_warmup=%d n_har=%d n_rep=%d grid=%s\n",
            N_ITER, N_WARMUP, N_HAR, N_REP, GRID))


# ============================================================
# DATA SIMULATION (matches Production_Run.R)
# ============================================================

simulate_data <- function(n, p, d, m) {
  if ((d * m <= 50) || (m < 2)) {
    A <- round(matrix(runif(m * d, -1, 1), m, d))
    while ((!is_totally_unimodular(A)) || any(rowSums(abs(A)) == 0)) {
      A <- round(matrix(runif(m * d, -1, 1), m, d))
    }
    b <- rep(1, m)
  } else {
    A <- matrix(0, m, d)
    for (i in 1:m) {
      ind <- sample.int(d, 2, replace = FALSE)
      A[i, ind[1]] <- 1
      A[i, ind[2]] <- -1
    }
    b <- sample(0:1, m, replace = TRUE)
  }

  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  beta_true <- matrix(rnorm(p * d), nrow = p, ncol = d)
  zeta_true <- X %*% beta_true + matrix(rnorm(n * d), nrow = n, ncol = d)

  ilp_map <- function(c) {
    lp_result <- lp(
      direction    = "max",
      objective.in = c,
      const.mat    = A,
      const.dir    = rep("<=", length(b)),
      const.rhs    = b,
      all.bin      = TRUE
    )
    lp_result$solution
  }

  y      <- t(apply(zeta_true, 1, ilp_map))
  U_free <- (t(A %*% t(y) == b)) * 1

  list(A = A, b = b, X = X, y = y, beta_true = beta_true, U_free = U_free)
}

RMSE <- function(v, v_true) sqrt(mean((v - v_true)^2))


# ============================================================
# SIMULATION CACHE
# ============================================================
# Sim data is deterministic per seed but ILP generation cost can be high at
# large d. Cache to disk keyed on (n, p, d, m, rep).

sim_cache_dir <- file.path("results", "runs", "speed_test", "sim_cache")
dir.create(sim_cache_dir, recursive = TRUE, showWarnings = FALSE)

cached_simulate <- function(n, p, d, m, rep) {
  key  <- sprintf("sim_n%d_p%d_d%d_m%d_rep%d.rds", n, p, d, m, rep)
  path <- file.path(sim_cache_dir, key)
  if (file.exists(path)) {
    cat(sprintf("  [cache hit] %s\n", basename(path)))
    return(readRDS(path))
  }
  set.seed(1000 + rep)
  t0  <- Sys.time()
  sim <- simulate_data(n, p, d, m)
  cat(sprintf("  [cache miss] generated %s in %.1fs\n",
              basename(path), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  saveRDS(sim, path)
  sim
}


# ============================================================
# PARAMETRIZED MH-WITHIN-GIBBS SAMPLER
# Same as Production_Run.R but with sampler_fn + feasible_fn injected.
# ============================================================

run_mhwg_sampler <- function(sim_data, n_iter, n_warmup, n_har, method,
                             sampler_fn, feasible_fn) {

  A         <- sim_data$A
  X         <- sim_data$X
  y         <- sim_data$y
  beta_true <- sim_data$beta_true
  U_free    <- sim_data$U_free

  n <- nrow(X); p <- ncol(X); d <- ncol(y); m <- nrow(A)

  B0    <- 1
  V_mat <- solve(diag(1 / B0, p) + t(X) %*% X)
  L_mat <- t(chol(V_mat))
  ge    <- 1 - y

  beta <- matrix(0, nrow = p, ncol = d)
  U    <- matrix(rexp(n * m), nrow = n, ncol = m) * U_free
  UA   <- U %*% A

  zeta <- matrix(NA, nrow = n, ncol = d)
  for (j in 1:d) {
    mu_j <- X %*% beta[, j]
    lo   <- ifelse(y[, j] == 1, UA[, j], -Inf)
    hi   <- ifelse(y[, j] == 1, Inf,    UA[, j])
    zeta[, j] <- rtruncnorm(n, a = lo, b = hi, mean = mu_j, sd = 1)
  }

  beta_samples <- array(NA, dim = c(n_iter, p, d))

  rt <- system.time({
    for (iter in 1:n_iter) {

      U  <- sampler_fn(t(A), zeta, ge, U, U_free,
                       n_iter = n_har, kernel = method)
      UA <- U %*% A

      zeta_new <- zeta
      J <- min(d, 100)
      idx <- sample.int(d, J, replace = FALSE)
      for (j in idx) {
        mu_j <- X %*% beta[, j]
        lo   <- ifelse(y[, j] == 1, UA[, j], -Inf)
        hi   <- ifelse(y[, j] == 1, Inf,    UA[, j])
        zeta_new[, j] <- rtruncnorm(n, a = lo, b = hi, mean = mu_j, sd = 1)
      }

      zeta_tilde <- (y > 0.5) * pmax(zeta, zeta_new) +
                    (y < 0.5) * pmin(zeta, zeta_new)
      U_star <- sampler_fn(t(A), zeta_tilde, ge, U, U_free,
                           n_iter = n_har, kernel = method)
      acc    <- feasible_fn(t(A), zeta, ge, U_star)
      zeta   <- zeta_new * acc + zeta * (1 - acc)

      mean_normal <- V_mat %*% t(X) %*% zeta
      beta <- mean_normal + L_mat %*% matrix(rnorm(p * d), nrow = p, ncol = d)
      beta_samples[iter, , ] <- beta
    }
  })

  post_mean <- apply(beta_samples[(n_warmup + 1):n_iter, , , drop = FALSE],
                     c(2, 3), mean)
  list(
    wall = as.numeric(rt[3]),
    cpu  = as.numeric(rt[1]),
    rmse = RMSE(post_mean, beta_true)
  )
}


# ============================================================
# BENCHMARK LOOP
# ============================================================

rows <- list()

for (cell in cells) {
  n <- cell$n; p <- cell$p; d <- cell$d; m <- cell$m
  cat(sprintf("\n=== Cell n=%d p=%d d=%d m=%d ===\n", n, p, d, m))

  for (rep in seq_len(N_REP)) {

    # Same data + same init RNG seed for both versions per replicate.
    sim <- cached_simulate(n, p, d, m, rep)

    set.seed(2000 + rep)
    r_v1 <- run_mhwg_sampler(sim, N_ITER, N_WARMUP, N_HAR, "exponential",
                             sampler_fn  = loop_hit_and_run_multi_kernel,
                             feasible_fn = check_feasible)

    set.seed(2000 + rep)
    r_v2 <- run_mhwg_sampler(sim, N_ITER, N_WARMUP, N_HAR, "exponential",
                             sampler_fn  = loop_hit_and_run_multi_kernel_v2,
                             feasible_fn = check_feasible_v2)

    sp_wall <- r_v1$wall / r_v2$wall
    sp_cpu  <- r_v1$cpu  / r_v2$cpu
    par_eff <- sp_wall / sp_cpu   # parallel efficiency of v2 vs v1

    rows[[length(rows) + 1]] <- data.frame(
      n = n, p = p, d = d, m = m, rep = rep,
      v1_wall = r_v1$wall, v2_wall = r_v2$wall,
      v1_cpu  = r_v1$cpu,  v2_cpu  = r_v2$cpu,
      v1_rmse = r_v1$rmse, v2_rmse = r_v2$rmse,
      speedup_wall        = sp_wall,
      speedup_cpu         = sp_cpu,
      parallel_efficiency = par_eff
    )

    cat(sprintf("  rep %d:  v1 wall=%.2fs cpu=%.2fs  |  v2 wall=%.2fs cpu=%.2fs  |  speedup=%.2fx (wall) %.2fx (cpu)  |  par_eff=%.2f  |  RMSE v1=%.4f v2=%.4f\n",
                rep, r_v1$wall, r_v1$cpu, r_v2$wall, r_v2$cpu,
                sp_wall, sp_cpu, par_eff,
                r_v1$rmse, r_v2$rmse))
  }
}

results <- do.call(rbind, rows)


# ============================================================
# SUMMARY
# ============================================================

cat("\n\n============================================================\n")
cat("SUMMARY (median across replicates)\n")
cat("============================================================\n")

agg <- aggregate(
  cbind(v1_wall, v2_wall, v1_cpu, v2_cpu, v1_rmse, v2_rmse,
        speedup_wall, speedup_cpu, parallel_efficiency) ~ n + p + d + m,
  data = results, FUN = median
)
print(agg)

cat("\nMAD across replicates (wall time):\n")
mad_tab <- aggregate(
  cbind(v1_wall, v2_wall) ~ n + p + d + m,
  data = results, FUN = mad
)
print(mad_tab)

# Correctness gate: posterior-mean RMSE must agree within MC tolerance.
delta <- abs(agg$v1_rmse - agg$v2_rmse)
tol   <- 0.10   # generous; tighten with more replicates / iters
cat(sprintf("\nMax |RMSE_v1 - RMSE_v2| across cells: %.4f (tolerance %.2f)\n",
            max(delta), tol))
if (any(delta > tol)) {
  warning("Posterior-mean RMSE diverges between v1 and v2 beyond tolerance — check correctness!")
}

# Persist.
out_dir <- file.path("results", "runs", "speed_test")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_path <- file.path(out_dir,
                      sprintf("speed_test_%s.rds",
                              format(Sys.time(), "%Y%m%d_%H%M%S")))
saveRDS(list(results = results, agg = agg), out_path)
cat(sprintf("\nSaved: %s\n", out_path))
