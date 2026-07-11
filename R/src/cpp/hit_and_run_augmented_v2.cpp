#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

// Revised hit-and-run sampler. Same semantics as hit_and_run_multi_kernel,
// but with the optimizations identified in the speed audit:
//   1. Drop -I augmentation; handle nonneg constraints per-coord (O(d) not O(m*d))
//   2. Replace m row-dots with a single Armadillo gemv per iteration
//   3. Maintain A * x0 incrementally (axpy) instead of recomputing
//   4. Sample direction only over active coordinates (skip masked-out RNG)
//   5. Exponential kernel: sign-flip when direction_sum < 0 instead of rejection
//   6. Reuse buffers across iterations (no per-iter heap allocs)
//   7. schedule(static) across chains (each chain has identical cost)

// =============================================================================
// HELPER: TRUNCATED NORMAL (same as v1)
// =============================================================================

static inline double sample_truncnorm_v2(double mean, double sd,
                                         double lower, double upper) {
  double a = (lower - mean) / sd;
  double b = (upper - mean) / sd;
  a = std::max(-37.0, std::min(37.0, a));
  b = std::max(-37.0, std::min(37.0, b));

  double Phi_a = R::pnorm(a, 0.0, 1.0, 1, 0);
  double Phi_b = R::pnorm(b, 0.0, 1.0, 1, 0);

  if (Phi_b - Phi_a < 1e-15) {
    return mean + sd * (a + b) / 2.0;
  }

  double u = R::runif(0, 1);
  double Phi_x = Phi_a + u * (Phi_b - Phi_a);
  double x_std = R::qnorm(Phi_x, 0.0, 1.0, 1, 0);
  return mean + sd * x_std;
}

// =============================================================================
// MAIN HIT-AND-RUN (REVISED)
// =============================================================================

//' Optimized Hit-and-Run Sampler (v2)
//'
//' Same semantics as hit_and_run_multi_kernel but with cache-friendly bound
//' computation and incremental Ax maintenance.
// [[Rcpp::export]]
arma::vec hit_and_run_multi_kernel_v2(
    const arma::mat& A,
    const arma::vec& z,
    const arma::vec& greater_equal,
    arma::vec x0,
    int n_iter,
    const arma::vec& kappa,
    double rho = 1.0,
    std::string kernel = "exponential",
    int max_dir_tries = 1000,
    double bound_truncation = 1e5
) {
  const int d = x0.n_elem;

  if ((int)kappa.n_elem != d)
    Rcpp::stop("Length of kappa must equal dimension of x0");
  if (arma::accu(kappa) < 1)
    Rcpp::stop("kappa must have at least one non-zero element");
  if (greater_equal.n_elem != A.n_rows)
    Rcpp::stop("Length of greater_equal must equal number of rows of A");
  if (kernel != "exponential" && kernel != "half_gaussian")
    Rcpp::stop("kernel must be 'exponential' or 'half_gaussian'");

  const int m = A.n_rows;

  // Transform constraints to A_trans * x <= z_trans form.
  arma::mat A_trans(m, d);
  arma::vec z_trans(m);
  for (int j = 0; j < m; ++j) {
    if (greater_equal[j] > 0.5) {
      A_trans.row(j) = -A.row(j);
      z_trans[j]     = -z[j];
    } else {
      A_trans.row(j) = A.row(j);
      z_trans[j]     = z[j];
    }
  }

  // Active coordinates.
  arma::uvec active_idx = arma::find(kappa > 0.5);
  const int n_active    = active_idx.n_elem;

  // Pre-allocate buffers (reused every iteration).
  arma::vec direction(d, arma::fill::zeros);
  arma::vec Ad(m);
  arma::vec Ax = A_trans * x0;   // maintained incrementally below
  arma::vec rand_active(n_active);

  for (int iter = 1; iter < n_iter; ++iter) {

    // -----------------------------------------------------------------
    // STEP 1: sample direction (active coords only)
    // -----------------------------------------------------------------
    bool direction_valid = false;
    double direction_sum = 0.0;

    for (int tries = 0; tries < max_dir_tries && !direction_valid; ++tries) {

      // Fill only the active slots; inactive stay zero.
      for (int i = 0; i < n_active; ++i) {
        rand_active[i] = R::norm_rand();
      }

      // Compute norm over active part.
      double norm2 = 0.0;
      for (int i = 0; i < n_active; ++i) norm2 += rand_active[i] * rand_active[i];
      if (norm2 < 1e-24) continue;
      double inv_norm = 1.0 / std::sqrt(norm2);

      // Scatter into full direction vector (inactive remain zero from init).
      // First zero out previously-set active slots in case mask differs across
      // iterations (it doesn't here, but cheap insurance).
      direction.zeros();
      for (int i = 0; i < n_active; ++i) {
        direction[active_idx[i]] = rand_active[i] * inv_norm;
      }

      // direction_sum = sum over active coords (== sum over all since others are 0).
      direction_sum = 0.0;
      for (int i = 0; i < n_active; ++i) {
        direction_sum += direction[active_idx[i]];
      }

      if (kernel == "exponential") {
        // Symmetric in direction: flip sign instead of rejecting.
        if (direction_sum < -1e-12) {
          direction      = -direction;
          direction_sum  = -direction_sum;
        }
        if (direction_sum > 1e-12) direction_valid = true;
      } else {
        direction_valid = true;
      }
    }

    if (!direction_valid)
      Rcpp::stop("Failed to sample valid direction after max attempts");

    // -----------------------------------------------------------------
    // STEP 2: feasible interval via single gemv + per-coord nonneg
    // -----------------------------------------------------------------
    Ad = A_trans * direction;   // one BLAS gemv

    double t_min = -std::numeric_limits<double>::infinity();
    double t_max =  std::numeric_limits<double>::infinity();

    // Linear constraints A_trans * x <= z_trans.
    for (int j = 0; j < m; ++j) {
      const double Ad_j = Ad[j];
      const double Ax_j = Ax[j];
      if (std::abs(Ad_j) > 1e-12) {
        const double t_cand = (z_trans[j] - Ax_j) / Ad_j;
        if (Ad_j > 0) t_max = std::min(t_max, t_cand);
        else          t_min = std::max(t_min, t_cand);
      } else if (Ax_j > z_trans[j] + 1e-12) {
        Rcpp::stop("Starting point not in feasible region");
      }
    }

    // Nonnegativity: x0[k] + alpha * direction[k] >= 0.
    // Equivalent constraint row in v1 was -e_k * u <= 0 (Ad = -direction[k]).
    for (int i = 0; i < n_active; ++i) {
      const arma::uword k  = active_idx[i];
      const double v_k     = direction[k];
      const double x_k     = x0[k];
      if (std::abs(v_k) > 1e-12) {
        const double t_cand = -x_k / v_k;
        if (v_k < 0) t_max = std::min(t_max, t_cand);  // alpha <= -x/v when v<0
        else         t_min = std::max(t_min, t_cand);  // alpha >= -x/v when v>0
      } else if (x_k < -1e-12) {
        Rcpp::stop("Starting point not in feasible region (nonneg)");
      }
    }

    if (t_min > t_max) continue;   // skip iter; no feasible step

    const double t_min_f = std::max(t_min, -bound_truncation);
    const double t_max_f = std::min(t_max,  bound_truncation);

    // -----------------------------------------------------------------
    // STEP 3: sample step size (kernel-specific)
    // -----------------------------------------------------------------
    double alpha = 0.0;

    if (kernel == "exponential") {
      const double lambda_rate = rho * direction_sum;
      if (lambda_rate <= 0)
        Rcpp::stop("Internal error: lambda_rate <= 0 for exponential kernel");

      const double c = -lambda_rate;
      const double a = c * t_min_f;
      const double b = c * t_max_f;

      double u_rand = R::runif(0, 1);
      u_rand = std::max(1e-15, std::min(1.0 - 1e-15, u_rand));

      if (std::abs(b - a) < 1e-12) {
        alpha = t_min_f + u_rand * (t_max_f - t_min_f);
      } else if (b - a > 700) {
        alpha = (1.0 / c) * (b + std::log(u_rand));
      } else if (b - a < -700) {
        alpha = (1.0 / c) * (a + std::log(1.0 - u_rand));
      } else {
        alpha = (1.0 / c) * (a + std::log(1.0 - u_rand * (1.0 - std::exp(b - a))));
      }
    } else {
      // half_gaussian
      double a_coeff = 0.0, b_coeff = 0.0;
      for (int i = 0; i < n_active; ++i) {
        const arma::uword k = active_idx[i];
        const double v_k = direction[k];
        const double w_k = x0[k];
        a_coeff += v_k * v_k;
        b_coeff += w_k * v_k;
      }
      a_coeff *= 0.5;

      if (a_coeff < 1e-15) {
        alpha = t_min_f + R::runif(0, 1) * (t_max_f - t_min_f);
      } else {
        const double mu_g = -b_coeff / (2.0 * a_coeff);
        const double sd_g = std::sqrt(1.0 / (2.0 * a_coeff));
        alpha = sample_truncnorm_v2(mu_g, sd_g, t_min_f, t_max_f);
      }
    }

    // -----------------------------------------------------------------
    // STEP 4: update position + Ax incrementally
    // -----------------------------------------------------------------
    x0 += alpha * direction;
    Ax += alpha * Ad;
  }

  return x0;
}

// =============================================================================
// LOOP OVER CHAINS
// =============================================================================

//' Optimized Loop Hit-and-Run (v2)
// [[Rcpp::export]]
arma::mat loop_hit_and_run_multi_kernel_v2(
    const arma::mat& A,
    const arma::mat& Z,
    const arma::mat& Greater_equal,
    const arma::mat& X0,
    const arma::mat& Kappa,
    int n_iter,
    double rho = 1.0,
    std::string kernel = "exponential",
    int max_dir_tries = 1000,
    double bound_truncation = 1e5
) {
  const int n = X0.n_rows;
  const int d = X0.n_cols;

  if ((int)Z.n_rows != n || (int)Greater_equal.n_rows != n || (int)Kappa.n_rows != n)
    Rcpp::stop("Z, Greater_equal, and Kappa must have same number of rows as X0");

  arma::mat final_samples(n, d);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    arma::vec z_i     = Z.row(i).t();
    arma::vec ge_i    = Greater_equal.row(i).t();
    arma::vec x0_i    = X0.row(i).t();
    arma::vec kappa_i = Kappa.row(i).t();

    arma::vec sample_i;
    if (arma::accu(kappa_i) == 0) {
      sample_i = x0_i;
    } else {
      sample_i = hit_and_run_multi_kernel_v2(
        A, z_i, ge_i, x0_i, n_iter, kappa_i,
        rho, kernel, max_dir_tries, bound_truncation
      );
    }
    final_samples.row(i) = sample_i.t();
  }

  return final_samples;
}

// =============================================================================
// FEASIBILITY CHECK (identical semantics, renamed to avoid symbol clash)
// =============================================================================

//' Check Feasibility (v2 — identical to v1)
// [[Rcpp::export]]
Rcpp::IntegerVector check_feasible_v2(
    const arma::mat& A,
    const arma::mat& Z,
    const arma::mat& Greater_equal,
    const arma::mat& X0
) {
  const int n = X0.n_rows;
  const int d = X0.n_cols;
  const int m = A.n_rows;

  Rcpp::IntegerVector feas(n);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    bool ok = true;
    arma::rowvec x0_i = X0.row(i);

    for (int k = 0; k < d; ++k) {
      if (x0_i[k] < 0) { ok = false; break; }
    }
    if (!ok) { feas[i] = 0; continue; }

    for (int j = 0; j < m; ++j) {
      const double dotVal = arma::dot(A.row(j), x0_i);
      const double z_val  = Z(i, j);
      const double ge_val = Greater_equal(i, j);
      if (ge_val > 0.5) {
        if (dotVal < z_val - 1e-12) { ok = false; break; }
      } else {
        if (dotVal > z_val + 1e-12) { ok = false; break; }
      }
    }
    feas[i] = ok ? 1 : 0;
  }
  return feas;
}
