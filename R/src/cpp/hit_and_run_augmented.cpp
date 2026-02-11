#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]

// =============================================================================
// HELPER FUNCTIONS FOR TRUNCATED DISTRIBUTIONS
// =============================================================================

//' Sample from Truncated Normal Distribution
 //' 
 //' Uses inverse CDF method with numerical stability.
 //' 
 //' @param mean Mean of untruncated normal
 //' @param sd Standard deviation of untruncated normal
 //' @param lower Lower bound
 //' @param upper Upper bound
 //' @return Sample from TN(mean, sd^2, [lower, upper])
 // [[Rcpp::export]]
 double sample_truncnorm(double mean, double sd, double lower, double upper) {
   // Standardize bounds
   double a = (lower - mean) / sd;
   double b = (upper - mean) / sd;
   
   // Clip to prevent numerical issues with pnorm
   // pnorm can handle roughly [-37, 37]
   a = std::max(-37.0, std::min(37.0, a));
   b = std::max(-37.0, std::min(37.0, b));
   
   // Sample using inverse CDF
   // CDF: F(x) = [Φ(x) - Φ(a)] / [Φ(b) - Φ(a)]
   // Sample u ~ Uniform(0,1), return F^{-1}(u)
   
   double Phi_a = R::pnorm(a, 0.0, 1.0, 1, 0);  // P(Z <= a)
   double Phi_b = R::pnorm(b, 0.0, 1.0, 1, 0);  // P(Z <= b)
   
   // Handle edge cases
   if (Phi_b - Phi_a < 1e-15) {
     // Interval has near-zero probability, return midpoint
     return mean + sd * (a + b) / 2.0;
   }
   
   double u = R::runif(0, 1);
   double Phi_x = Phi_a + u * (Phi_b - Phi_a);
   
   // Inverse CDF: qnorm
   double x_std = R::qnorm(Phi_x, 0.0, 1.0, 1, 0);
   
   return mean + sd * x_std;
 }

// =============================================================================
// MAIN HIT-AND-RUN FUNCTION WITH KERNEL SELECTION
// =============================================================================

//' Hit-and-Run Sampler with Multiple Kernel Options
 //' 
 //' Implements hit-and-run sampling with choice of kernel:
 //' - "exponential": exp(-rho * sum(u_k)) for u_k > 0
 //' - "half_gaussian": exp(-0.5 * sum(u_k^2)) for u_k > 0
 //' 
 //' @param A Constraint coefficient matrix (m × d)
 //' @param z Right-hand side vector (m)
 //' @param greater_equal Binary vector indicating inequality type (m)
 //' @param x0 Feasible starting point (d)
 //' @param n_iter Number of iterations
 //' @param kappa Binary mask for active coordinates (d)
 //' @param rho Rate parameter (only used for exponential kernel)
 //' @param kernel Kernel type: "exponential" or "half_gaussian"
 //' @param max_dir_tries Maximum direction resampling attempts
 //' @param bound_truncation Truncate infinite bounds to +/- this value
 //' @return Final sample (d)
 //' 
 //' @details
 //' For exponential kernel:
 //' - Enforces direction positivity constraint: sum(v_k) > 0 for k in K_i
 //' - Samples from truncated exponential along line
 //' 
 //' For half-Gaussian kernel:
 //' - No direction constraint (symmetric kernel)
 //' - Samples from truncated Gaussian along line
 //' 
 //' @examples
 //' A <- matrix(c(1, 0, 0, 1), nrow = 2)
 //' z <- c(2, 2)
 //' greater_equal <- c(0, 0)
 //' x0 <- c(1, 1)
 //' kappa <- c(1, 1)
 //' 
 //' # Exponential kernel
 //' sample_exp <- hit_and_run_multi_kernel(A, z, greater_equal, x0, 100, kappa, 
 //'                                        rho = 1.0, kernel = "exponential")
 //' 
 //' # Half-Gaussian kernel
 //' sample_gauss <- hit_and_run_multi_kernel(A, z, greater_equal, x0, 100, kappa,
 //'                                          kernel = "half_gaussian")
 // [[Rcpp::export]]
 arma::vec hit_and_run_multi_kernel(
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
   int d = x0.n_elem;
   
   // Validation
   if (kappa.n_elem != d) {
     Rcpp::stop("Length of kappa must equal dimension of x0");
   }
   if (arma::accu(kappa) < 1) {
     Rcpp::stop("kappa must have at least one non-zero element");
   }
   if (greater_equal.n_elem != A.n_rows) {
     Rcpp::stop("Length of greater_equal must equal number of rows of A");
   }
   if (kernel != "exponential" && kernel != "half_gaussian") {
     Rcpp::stop("kernel must be 'exponential' or 'half_gaussian'");
   }
   
   int m = A.n_rows;
   
   // Transform constraints to uniform <= form
   arma::mat A_trans(m, d);
   arma::vec z_trans(m);
   
   for (int j = 0; j < m; j++) {
     if (greater_equal[j] > 0.5) {
       A_trans.row(j) = -A.row(j);
       z_trans[j] = -z[j];
     } else {
       A_trans.row(j) = A.row(j);
       z_trans[j] = z[j];
     }
   }
   
   // Augment with nonnegativity: u >= 0 becomes -I*u <= 0
   arma::mat A_aug = arma::join_cols(A_trans, -arma::eye(d, d));
   arma::vec z_aug = arma::join_cols(z_trans, arma::zeros(d));
   
   // Identify active coordinates
   arma::uvec active_idx = arma::find(kappa > 0.5);
   int n_active = active_idx.n_elem;
   
   // Main hit-and-run loop
   for (int iter = 1; iter < n_iter; iter++) {
     
     // =========================================================================
     // STEP 1: Sample Direction
     // =========================================================================
     bool direction_valid = false;
     int dir_tries = 0;
     arma::vec direction(d);
     double direction_sum = 0.0;
     
     while (!direction_valid && dir_tries < max_dir_tries) {
       dir_tries++;
       
       // Sample random direction
       direction = arma::randn(d);
       
       // Enforce zero for inactive coordinates
       direction = direction % kappa;
       
       // Normalize
       double norm_dir = arma::norm(direction, 2);
       if (norm_dir < 1e-12) {
         continue;
       }
       direction = direction / norm_dir;
       
       // Compute sum over active coordinates
       direction_sum = 0.0;
       for (unsigned int i = 0; i < active_idx.n_elem; i++) {
         direction_sum += direction[active_idx[i]];
       }
       
       // Check kernel-specific constraints
       if (kernel == "exponential") {
         // Exponential: requires direction_sum > 0
         if (direction_sum > 1e-12) {
           direction_valid = true;
         }
       } else {
         // Half-Gaussian: no constraint
         direction_valid = true;
       }
     }
     
     if (!direction_valid) {
       Rcpp::stop("Failed to sample valid direction after max attempts");
     }
     
     // =========================================================================
     // STEP 2: Compute Feasible Interval [t_min, t_max]
     // =========================================================================
     double t_min = -std::numeric_limits<double>::infinity();
     double t_max = std::numeric_limits<double>::infinity();
     
     for (arma::uword j = 0; j < A_aug.n_rows; j++) {
       double Ad = arma::dot(A_aug.row(j), direction);
       double Ax = arma::dot(A_aug.row(j), x0);
       
       if (std::abs(Ad) > 1e-12) {
         double t_candidate = (z_aug[j] - Ax) / Ad;
         if (Ad > 0) {
           t_max = std::min(t_max, t_candidate);
         } else {
           t_min = std::max(t_min, t_candidate);
         }
       } else {
         if (Ax > z_aug[j] + 1e-12) {
           Rcpp::stop("Starting point not in feasible region");
         }
       }
     }
     
     if (t_min > t_max) {
       // Infeasible interval, skip this iteration
       continue;
     }
     
     // Truncate infinite bounds
     double t_min_finite = std::max(t_min, -bound_truncation);
     double t_max_finite = std::min(t_max, bound_truncation);
     
     // =========================================================================
     // STEP 3: Sample Step Size (Kernel-Specific)
     // =========================================================================
     double alpha = 0.0;
     
     if (kernel == "exponential") {
       // -------------------------------------------------------------------
       // EXPONENTIAL KERNEL: Sample from truncated exponential
       // -------------------------------------------------------------------
       // Rate: lambda = rho * direction_sum (always positive by construction)
       double lambda_rate = rho * direction_sum;
       
       if (lambda_rate <= 0) {
         Rcpp::stop("Internal error: lambda_rate <= 0 for exponential kernel");
       }
       
       // Sample from truncated exponential on [t_min_finite, t_max_finite]
       // Using stable log-space computation
       double c = -lambda_rate;
       double a = c * t_min_finite;
       double b = c * t_max_finite;
       
       double u_rand = R::runif(0, 1);
       u_rand = std::max(1e-15, std::min(1.0 - 1e-15, u_rand));
       
       if (std::abs(b - a) < 1e-12) {
         // Degenerate case
         alpha = t_min_finite + u_rand * (t_max_finite - t_min_finite);
       } else if (b - a > 700) {
         // Upper overflow case
         alpha = (1.0 / c) * (b + std::log(u_rand));
       } else if (b - a < -700) {
         // Lower overflow case
         alpha = (1.0 / c) * (a + std::log(1.0 - u_rand));
       } else {
         // Normal case
         alpha = (1.0 / c) * (a + std::log(1.0 - u_rand * (1.0 - std::exp(b - a))));
       }
       
     } else if (kernel == "half_gaussian") {
       // -------------------------------------------------------------------
       // HALF-GAUSSIAN KERNEL: Sample from truncated Gaussian
       // -------------------------------------------------------------------
       // Target along line: exp(-0.5 * sum((omega_k + alpha * v_k)^2))
       // This is Gaussian in alpha with:
       //   a = 0.5 * sum(v_k^2)
       //   b = sum(omega_k * v_k)
       //   mean = -b / (2a)
       //   variance = 1 / (2a)
       
       double a_coeff = 0.0;
       double b_coeff = 0.0;
       
       for (unsigned int i = 0; i < active_idx.n_elem; i++) {
         arma::uword k = active_idx[i];
         double v_k = direction[k];
         double omega_k = x0[k];
         a_coeff += v_k * v_k;
         b_coeff += omega_k * v_k;
       }
       a_coeff *= 0.5;
       
       if (a_coeff < 1e-15) {
         // Degenerate case: uniform sampling
         alpha = t_min_finite + R::runif(0, 1) * (t_max_finite - t_min_finite);
       } else {
         // Gaussian parameters
         double gaussian_mean = -b_coeff / (2.0 * a_coeff);
         double gaussian_sd = std::sqrt(1.0 / (2.0 * a_coeff));
         
         // Sample from truncated Gaussian
         alpha = sample_truncnorm(gaussian_mean, gaussian_sd, 
                                  t_min_finite, t_max_finite);
       }
     }
     
     // =========================================================================
     // STEP 4: Update Position
     // =========================================================================
     x0 = x0 + alpha * direction;
   }
   
   return x0;
 }

// =============================================================================
// LOOP FUNCTION FOR MULTIPLE CHAINS
// =============================================================================

//' Loop Hit-and-Run Over Multiple Chains with Kernel Selection
 //' 
 //' @param A Constraint matrix (m × d)
 //' @param Z Matrix of z vectors (n × d), one per chain
 //' @param Greater_equal Matrix of inequality indicators (n × d)
 //' @param X0 Matrix of starting points (n × d)
 //' @param Kappa Matrix of masks (n × d)
 //' @param n_iter Number of iterations
 //' @param rho Rate parameter for exponential kernel
 //' @param kernel Kernel type: "exponential" or "half_gaussian"
 //' @param max_dir_tries Maximum direction attempts
 //' @param bound_truncation Bound truncation value
 //' @return Matrix of final samples (n × d)
 //' 
 //' @examples
 //' A <- matrix(c(1, 0, 0, 1), nrow = 2)
 //' Z <- matrix(c(2, 2, 2, 2), nrow = 2, byrow = TRUE)
 //' Greater_equal <- matrix(c(0, 0, 0, 0), nrow = 2, byrow = TRUE)
 //' X0 <- matrix(c(1, 1, 0.5, 0.5), nrow = 2, byrow = TRUE)
 //' Kappa <- matrix(c(1, 1, 1, 1), nrow = 2, byrow = TRUE)
 //' 
 //' samples <- loop_hit_and_run_multi_kernel(A, Z, Greater_equal, X0, Kappa,
 //'                                          n_iter = 100, kernel = "half_gaussian")
 // [[Rcpp::export]]
 arma::mat loop_hit_and_run_multi_kernel(
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
   int n = X0.n_rows;
   int d = X0.n_cols;
   
   // Validation
   if (Z.n_rows != n || Greater_equal.n_rows != n || Kappa.n_rows != n) {
     Rcpp::stop("Z, Greater_equal, and Kappa must have same number of rows as X0");
   }
   
   arma::mat final_samples(n, d);
   
   // Parallelize over chains
#pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < n; i++) {
     arma::vec z_i = Z.row(i).t();
     arma::vec ge_i = Greater_equal.row(i).t();
     arma::vec x0_i = X0.row(i).t();
     arma::vec kappa_i = Kappa.row(i).t();
     
     arma::vec sample_i;
     
     if (arma::accu(kappa_i) == 0) {
       sample_i = x0_i;
     } else {
       sample_i = hit_and_run_multi_kernel(
         A, z_i, ge_i, x0_i, n_iter, kappa_i,
         rho, kernel, max_dir_tries, bound_truncation
       );
     }
     
     final_samples.row(i) = sample_i.t();
   }
   
   return final_samples;
 }

// =============================================================================
// FEASIBILITY CHECK (UNCHANGED)
// =============================================================================

//' Check Feasibility of Starting Points
 //' 
 //' @param A Constraint matrix (m × d)
 //' @param Z Matrix of z vectors (n × d)
 //' @param Greater_equal Matrix of inequality indicators (n × d)
 //' @param X0 Matrix of points to check (n × d)
 //' @return Integer vector (n) with 1 if feasible, 0 if not
 // [[Rcpp::export]]
 Rcpp::IntegerVector check_feasible(
     const arma::mat& A,
     const arma::mat& Z,
     const arma::mat& Greater_equal,
     const arma::mat& X0
 ) {
   int n = X0.n_rows;
   int d = X0.n_cols;
   int m = A.n_rows;
   
   Rcpp::IntegerVector feas(n);
   
#pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < n; i++) {
     bool isFeasible = true;
     arma::rowvec x0_i = X0.row(i);
     
     // Check nonnegativity
     for (int k = 0; k < d; k++) {
       if (x0_i[k] < 0) {
         isFeasible = false;
         break;
       }
     }
     
     if (!isFeasible) {
       feas[i] = 0;
       continue;
     }
     
     // Check linear constraints
     for (int j = 0; j < m; j++) {
       double dotVal = arma::dot(A.row(j), x0_i);
       double z_val = Z(i, j);
       double ge_val = Greater_equal(i, j);
       
       if (ge_val > 0.5) {
         if (dotVal < z_val - 1e-12) {
           isFeasible = false;
           break;
         }
       } else {
         if (dotVal > z_val + 1e-12) {
           isFeasible = false;
           break;
         }
       }
     }
     
     feas[i] = isFeasible ? 1 : 0;
   }
   
   return feas;
 }

// =============================================================================
// UTILITY: Get Kernel Information
// =============================================================================

//' Get Information About Available Kernels
 //' 
 //' @return List with kernel properties
 //' @export
 // [[Rcpp::export]]
 Rcpp::List get_kernel_info() {
   Rcpp::List info = Rcpp::List::create(
     Rcpp::Named("available_kernels") = Rcpp::CharacterVector::create(
       "exponential", "half_gaussian"
     ),
     Rcpp::Named("exponential") = Rcpp::List::create(
       Rcpp::Named("description") = "Exponential kernel: exp(-rho * sum(u_k))",
       Rcpp::Named("requires_direction_constraint") = true,
       Rcpp::Named("parameters") = Rcpp::CharacterVector::create("rho"),
       Rcpp::Named("theoretical_mean") = "1/rho (for each coordinate)",
       Rcpp::Named("theoretical_variance") = "1/rho^2 (for each coordinate)"
     ),
     Rcpp::Named("half_gaussian") = Rcpp::List::create(
       Rcpp::Named("description") = "Half-Gaussian kernel: exp(-0.5 * sum(u_k^2))",
       Rcpp::Named("requires_direction_constraint") = false,
       Rcpp::Named("parameters") = Rcpp::CharacterVector::create(),
       Rcpp::Named("theoretical_mean") = "sqrt(2/pi) ~ 0.798 (for each coordinate)",
       Rcpp::Named("theoretical_variance") = "1 - 2/pi ~ 0.363 (for each coordinate)"
     )
   );
   
   return info;
 }