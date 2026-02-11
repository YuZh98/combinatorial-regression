#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::plugins(openmp)]]
// [[Rcpp::depends(RcppArmadillo)]]


//' Hit-and-Run Sampler with Mixed Inequalities, Exponential Density, and Constrained Direction (Final Sample Only)
 //' 
 //' This function performs hit-and-run sampling from the polytope defined by the constraints:
 //' 
 //'   For j = 1,..., m (rows of A):
 //'     - If greater_equal[j] == 1, then A[j,]*u >= z[j],
 //'     - If greater_equal[j] == 0, then A[j,]*u <= z[j],
 //' 
 //' along with the nonnegativity constraint \(u \ge 0\).
 //' 
 //' The target density is proportional to \(\exp(-\rho \sum_k u_k)\). The algorithm
 //' constrains the random direction by the binary vector kappa (elementwise multiplication).
 //' Only the final sample (after n_iter iterations) is returned.
 //' 
 //' @param A A matrix of constraint coefficients.
 //' @param z A vector of right-hand sides for the constraints.
 //' @param greater_equal A binary vector indicating the inequality type for each row of A.
 //'                      If greater_equal[j] == 1, then the constraint is A[j,]*u >= z[j].
 //'                      If greater_equal[j] == 0, then the constraint is A[j,]*u <= z[j].
 //' @param x0 A feasible starting point inside the polytope.
 //' @param n_iter Number of iterations (steps) to perform.
 //' @param kappa A binary vector (0s and 1s) to constrain the movement direction elementwise.
 //' @param rho Parameter for the exponential density (default = 1).
 //' @return A vector corresponding to the final sample from the polytope.
 //' @examples
 //' A <- matrix(c(1, 2, 2, 1), nrow = 2, byrow = TRUE)
 //' z <- c(4, 3)
 //' greater_equal <- c(0, 1)
 //' x0 <- c(1, 1)
 //' kappa <- c(1, 0)
 //' final_sample <- hit_and_run_exp_kappa_last(A, z, greater_equal, x0, n_iter = 1000, kappa)
 // [[Rcpp::export]]
 arma::vec hit_and_run_exp_kappa_last(const arma::mat& A, const arma::vec& z, 
                                      const arma::vec& greater_equal,
                                      arma::vec x0, int n_iter, arma::vec kappa, 
                                      double rho = 1.0) {
   int d = x0.n_elem;
   
   if(kappa.n_elem != d) {
     Rcpp::stop("Length of kappa must equal the dimension of x0.");
   }
   
   if(arma::accu(kappa) < 1) {
     Rcpp::stop("kappa must have at least one non-zero element.");
   }
   
   // Check that greater_equal has the same number of elements as rows in A.
   if(greater_equal.n_elem != A.n_rows) {
     Rcpp::stop("Length of greater_equal must equal the number of rows of A.");
   }
   
   // Transform the constraints based on greater_equal.
   int m = A.n_rows;
   arma::mat A_trans(m, d);
   arma::vec z_trans(m);
   
   for (int j = 0; j < m; j++) {
     if (greater_equal[j] > 0.5) {
       // For a 'greater than or equal' constraint, transform:
       // A[j,]*u >= z[j] becomes -A[j,]*u <= -z[j]
       A_trans.row(j) = -A.row(j);
       z_trans[j] = -z[j];
     } else {
       // Otherwise, leave as is.
       A_trans.row(j) = A.row(j);
       z_trans[j] = z[j];
     }
   }
   
   // Augment with nonnegativity constraints: u >= 0 is -I*u <= 0.
   arma::mat A_aug = arma::join_cols(A_trans, -arma::eye(d, d));
   arma::vec z_aug = arma::join_cols(z_trans, arma::zeros(d));
   
   // Main hit-and-run loop.
   for (int i = 1; i < n_iter; i++) {
     // Sample a random direction from a multivariate normal distribution.
     arma::vec direction = arma::randn(d);
     
     // Constrain the direction by elementwise multiplying by kappa.
     direction = direction % kappa;
     
     // Re-normalize the direction. Resample if the constrained direction is nearly zero.
     double norm_dir = arma::norm(direction, 2);
     while (norm_dir < 1e-12) {
       direction = arma::randn(d);
       direction = direction % kappa;
       norm_dir = arma::norm(direction, 2);
     }
     direction = direction / norm_dir;
     
     // Determine the feasible step sizes along the direction.
     double t_min = -std::numeric_limits<double>::infinity();
     double t_max = std::numeric_limits<double>::infinity();
     
     for (size_t j = 0; j < A_aug.n_rows; j++) {
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
           Rcpp::stop("The starting point is not in the feasible region.");
         }
       }
     }
     
     if (t_min > t_max) {
       Rcpp::stop("No feasible step found in the current direction.");
     }
     
     // Compute the rate at which the target density changes along the line.
     double s = arma::sum(direction);
     double u_rand = R::runif(0, 1);
     double t;
     if (std::abs(s) < 1e-12) {
       t = t_min + u_rand * (t_max - t_min);
     } else {
       double c = -rho * s;
       t = (1.0 / c) * log(exp(c * t_min) + u_rand * (exp(c * t_max) - exp(c * t_min)));
     }
     
     // Update the current point.
     x0 = x0 + t * direction;
   }
   
   // Return the final sample.
   return x0;
 }

//-------------------------------------------------------------
// New function: loop_hit_and_run (Parallelized)
//-------------------------------------------------------------

//' Loop Hit-and-Run Sampler Over Multiple Chains (Parallelized)
 //' 
 //' For each chain \(i\), the \(i\)th row of the input matrices Z, Greater_equal, X0, and Kappa is extracted
 //' (converted to column vectors). If the sum of Kappa is zero for that chain, the current X0 is returned;
 //' otherwise, the hit-and-run sampler is run to obtain the final sample for that chain.
 //' The outer loop is parallelized using OpenMP.
 //' 
 //' @param A A matrix representing the constraint coefficients (common to all chains).
 //' @param Z A matrix where each row corresponds to the \(z\) vector for a chain.
 //' @param Greater_equal A matrix where each row corresponds to the greater_equal vector for a chain.
 //' @param X0 A matrix where each row is the starting point for a chain.
 //' @param Kappa A matrix where each row is the kappa vector for a chain.
 //' @param n_iter Number of iterations for the hit-and-run sampler.
 //' @param rho Parameter for the exponential density (default = 1).
 //' @return A matrix where each row is the final sample from the corresponding chain.
 //' @examples
 //' A <- matrix(c(1, 2, 2, 1), nrow = 2, byrow = TRUE)
 //' Z <- matrix(c(4, 3,
 //'               5, 2), nrow = 2, byrow = TRUE)
 //' Greater_equal <- matrix(c(0, 1,
 //'                           1, 0), nrow = 2, byrow = TRUE)
 //' X0 <- matrix(c(1, 1,
 //'                0.5, 1), nrow = 2, byrow = TRUE)
 //' Kappa <- matrix(c(1, 0,
 //'                   1, 1), nrow = 2, byrow = TRUE)
 //' final_samples <- loop_hit_and_run(A, Z, Greater_equal, X0, Kappa, n_iter = 1000)
 // [[Rcpp::export]]
 arma::mat loop_hit_and_run(const arma::mat& A, 
                            const arma::mat& Z,
                            const arma::mat& Greater_equal,
                            const arma::mat& X0,
                            const arma::mat& Kappa,
                            int n_iter, double rho = 1.0) {
   int n = X0.n_rows;       // Number of chains.
   int d = X0.n_cols;       // Dimension of each sample.
   
   // Check dimensions: Z, Greater_equal, and Kappa should have the same number of rows as X0.
   if (Z.n_rows != n || Greater_equal.n_rows != n || Kappa.n_rows != n) {
     Rcpp::stop("Z, Greater_equal, and Kappa must have the same number of rows as X0.");
   }
   
   // Container for final samples: each row will be the final sample for chain i.
   arma::mat final_samples(n, d);
   
   // Parallelize over chains using OpenMP.
#pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < n; i++) {
     // Extract the i-th row and convert to column vectors.
     arma::vec z_i     = Z.row(i).t();
     arma::vec ge_i    = Greater_equal.row(i).t();
     arma::vec x0_i    = X0.row(i).t();
     arma::vec kappa_i = Kappa.row(i).t();
     
     arma::vec sample_i;
     
     // If the kappa vector is all zeros, use the current x0_i.
     if (arma::accu(kappa_i) == 0) {
       sample_i = x0_i;
     } else {
       sample_i = hit_and_run_exp_kappa_last(A, z_i, ge_i, x0_i, n_iter, kappa_i, rho);
     }
     
     final_samples.row(i) = sample_i.t();
   }
   
   return final_samples;
 }


//' Check Feasibility of Starting Points for Multiple Chains
 //' 
 //' This function takes as input the same parameters as loop_hit_and_run, but instead of
 //' running the hit-and-run sampler, it checks if the supplied starting point (each row of X0)
 //' satisfies the constraints. For each chain i, the i-th row of Z, Greater_equal, and X0 is used.
 //' The constraints are defined as follows:
 //'   - For each constraint j (row of A):
 //'       * If Greater_equal[i,j] == 1, then the constraint is A[j,] * x0_i >= Z[i,j].
 //'       * Otherwise, the constraint is A[j,] * x0_i <= Z[i,j].
 //'   - Additionally, x0_i must satisfy the nonnegativity constraint: x0_i >= 0.
 //' 
 //' @param A A matrix of constraint coefficients (common to all chains).
 //' @param Z A matrix where each row corresponds to the z vector for a chain.
 //' @param Greater_equal A matrix where each row corresponds to the greater_equal vector for a chain.
 //' @param X0 A matrix where each row is the starting point for a chain.
 //' @param Kappa A matrix where each row is the kappa vector for a chain (not used in feasibility check).
 //' @param n_iter Number of iterations (ignored in this function).
 //' @param rho Parameter for the exponential density (ignored in this function).
 //' @return An integer vector of length n (number of chains), with 1 if the chain's x0 is feasible and 0 otherwise.
 //' @examples
 //' A <- matrix(c(1, 2,
 //'               2, 1), nrow = 2, byrow = TRUE)
 //' Z <- matrix(c(4, 3,
 //'               5, 2), nrow = 2, byrow = TRUE)
 //' Greater_equal <- matrix(c(0, 1,
 //'                           1, 0), nrow = 2, byrow = TRUE)
 //' X0 <- matrix(c(1, 1,
 //'                0.5, 1), nrow = 2, byrow = TRUE)
 //' Kappa <- matrix(c(1, 0,
 //'                   1, 1), nrow = 2, byrow = TRUE)
 //' feas <- check_feasible(A, Z, Greater_equal, X0, Kappa, 1000)
 // [[Rcpp::export]]
 Rcpp::IntegerVector check_feasible(const arma::mat& A, 
                                    const arma::mat& Z,
                                    const arma::mat& Greater_equal,
                                    const arma::mat& X0) {
   // Number of chains (rows in X0)
   int n = X0.n_rows;
   // Dimension of the variable
   int d = X0.n_cols;
   // Number of constraints (rows in A)
   int m = A.n_rows;
   
   // Container for feasibility results: 1 if feasible, 0 if not.
   Rcpp::IntegerVector feas(n);
   
   // Loop over chains
#pragma omp parallel for schedule(dynamic)
   for (int i = 0; i < n; i++) {
     bool isFeasible = true;
     
     // Extract the i-th starting point (as a row vector)
     arma::rowvec x0_i = X0.row(i);
     
     // Check nonnegativity: all components must be >= 0.
     for (int k = 0; k < d; k++) {
       if (x0_i[k] < 0) {
         isFeasible = false;
         break;
       }
     }
     
     // If nonnegative condition fails, mark as infeasible and continue.
     if (!isFeasible) {
       feas[i] = 0;
       continue;
     }
     
     // For each constraint j, check if x0_i satisfies it.
     // The i-th row of Z and Greater_equal provides the right-hand side and inequality type for this chain.
     for (int j = 0; j < m; j++) {
       double dotVal = arma::dot(A.row(j), x0_i);
       double z_val  = Z(i, j);
       double ge_val = Greater_equal(i, j);
       
       // If ge_val is 1 (or > 0.5), then require A.row(j)*x0_i >= z_val.
       if (ge_val > 0.5) {
         if (dotVal < z_val - 1e-12) {  // small tolerance for floating-point comparisons
           isFeasible = false;
           break;
         }
       } else { // Otherwise, require A.row(j)*x0_i <= z_val.
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
