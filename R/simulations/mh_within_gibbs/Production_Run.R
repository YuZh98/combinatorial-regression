rm(list=ls())
setwd('/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/r_code')
library(truncnorm)
library(lpSolve)
library(lintools)
library(Rcpp)
sourceCpp("hit_and_run.cpp") 
sourceCpp("hit_and_run_augmented.cpp")
sourceCpp("tum_check.cpp")  # check TUM

# Set seed for reproducibility
set.seed(123)
# -------------------------------
# Global
# -------------------------------
n <- 1000          # number of observations
p <- 5            # number of predictors
d_list <- c(1000)
m_list <- c(100)
n_rep <- 1
# For MCMC:
method_list <- c("exponential", "half_gaussian")
n_iter <- 10000     # Number of samples saved is n_iter
n_warmup <- 5000    # Only samples after n_warmup will be used to report RMSE(beta) and plotting
n_thin <- 25        # Only used in ACF plotting
n_iter_hit_and_run <- 100
# Post processing switches
button_save <- TRUE
button_plot <- FALSE

# -------------------------------
# Utilility functions
# -------------------------------
L2 <- function(v){
  sqrt(sum(v^2))
}

RMSE <- function(v, v_true){
  sqrt(mean((v-v_true)^2))
}

simulate_data <- function(n, p, d, m){
  if (d*m <= 50 | m < 2){
    A<- round(matrix(runif(m*d,-1,1),m,d))
    while( (!is_totally_unimodular(A)) || any(rowSums(abs(A))==0)){
      A<- round(matrix(runif(m*d,-1,1),m,d))
    }
    b<- rep(1,m)
  } else{
    # Generate A as a random incidence matrix
    A <- matrix(0, m, d)
    for (i in 1:m){
      ind <- sample(1:d, 2, replace = FALSE)
      A[i, ind[1]] <- 1
      A[i, ind[2]] <- -1
    }
    b <- sample(0:1, m, replace=TRUE)
  }
  X<- matrix( rnorm(n*p), n)
  beta_true <- matrix(rnorm(p * d), nrow = p, ncol = d)
  zeta_true <- X %*% beta_true + matrix(rnorm(n * d), n, d)
  ilp_map <- function(c){
    constraint_directions <- rep("<=", length(b))
    lp_result <- lp(direction = "max",
                    objective.in = c,
                    const.mat = A,
                    const.dir = constraint_directions,
                    const.rhs = b,
                    all.bin= TRUE)
    lp_result$solution
  }
  y <- t(apply(zeta_true, 1, ilp_map))
  U_free <- (t(A%*%t(y) == b))*1
  return(list(
    A=A, b=b, X=X, y=y, beta_true=beta_true, U_free=U_free
  ))
}


# Plotting functions
trace_plot <- function(beta_samples, p, d, n_warmup, n_iter){
  row_index <- c(1:min(p, 5))
  col_index <- c(1:min(d, 5))
  par(mfrow = c(length(row_index), length(col_index)))
  for (k in row_index){
    for (j in col_index){
      plot(beta_samples[(n_warmup+ 1):n_iter, k, j], type = 'l',
           main = paste("Trace for beta[", k, ",", j, "]", sep = ""),
           xlab = "Iteration", ylab = paste("beta[", k, ",", j, "]", sep = ""))
      abline(h=beta_true[k,j],col='red')
    }
  }
}

ACF_plot <- function(beta_samples, p, d, n_warmup, n_iter, thin=25){
  row_index <- c(1:min(p, 5))
  col_index <- c(1:min(d, 5))
  par(mfrow = c(length(row_index), length(col_index)))
  thinned_indices <- seq(from = n_warmup + 1, to = n_iter, by = thin)
  for (k in row_index) {
    for (j in col_index) {
      acf(beta_samples[thinned_indices, k, j],
          main = paste("ACF for beta[", k, ",", j, "] (thin=", thin, ")", sep = ""))
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
for (d in d_list){
  for (m in m_list){
    if (m < d){
      for (repetition in 1:n_rep){
        cat("=============== Running case (n,p,d,m)=(", n, ",", p, ",", d, ",", m, "); rep ", repetition, " ===============\n", sep="")
        sim_data <- simulate_data(n, p, d, m)
        A <- sim_data$A
        X <- sim_data$X
        y <- sim_data$y
        beta_true <- sim_data$beta_true
        U_free <- sim_data$U_free
        
        # Prior setup and pre-compute fixed variables
        B0 <- 1         # prior variance (scalar)
        V_mat <- solve(diag(1/B0, p) + t(X) %*% X)
        L_mat <- t(chol(V_mat))
        greaterequal = 1-y
        
        # Running all methods
        for (method in method_list){
          # Initialization
          beta <- matrix(0, nrow = p, ncol = d)
          U <- matrix(rexp(n*m),n,m) * U_free
          UA <- U%*%A
          zeta <- matrix(NA, n, d)
          for (j in 1:d) {
            mu_j <- X %*% beta[, j]
            lower_bound <- ifelse(y[, j] == 1, UA[,j], -Inf)
            upper_bound <- ifelse(y[, j] == 1, Inf, UA[,j])
            zeta[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu_j, sd = 1)
          }
          if (method == "exponential" | method =="half_gaussian"){
            cat("=== running method: ", method, " ===\n", sep="")
            ker <- method
            beta_samples <- array(NA, dim = c(n_iter, p, d))  # to store beta draws
            running_time <- system.time(
              # -------------------------------
              # Gibbs sampler iterations
              # -------------------------------
              for (iter in 1:n_iter) {
                rmse_beta <- RMSE(beta, beta_true)
                ##### Sanity check at the beginning and at some checkpoints to keep track of the process ######
                if (iter < 5 | (iter %% 1000 == 0)){
                  if (iter > 1) {
                    cat("MCMC iteration ", iter, ": RMSE(beta): ", rmse_beta, "; current AR: ", mean(accept_or_not), "\n", sep="")
                  } else {
                    cat("MCMC iteration ", iter, ": RMSE(beta): ", rmse_beta, "; U shape: (", nrow(U), ", ", ncol(U), ")\n", sep="")
                  }
                }
                
                ##### Update zeta #####
                U <- loop_hit_and_run_multi_kernel(t(A), zeta, greaterequal, U, U_free, n_iter=n_iter_hit_and_run, kernel=ker)
                UA <- U%*%A
                zeta_new <- zeta # matrix(NA, n, d)
                # Random subset of [d]
                J = min(d, 100)
                sampled_indices <- sample(1:d, J, replace = FALSE)
                for (j in sampled_indices) {
                  mu_j <- X %*% beta[, j]
                  lower_bound <- ifelse(y[, j] == 1, UA[,j], -Inf)
                  upper_bound <- ifelse(y[, j] == 1, Inf, UA[,j])
                  zeta_new[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu_j, sd = 1)
                }
                zeta_tilde <- (y>0.5) * pmax(zeta,zeta_new) + (y<0.5) * pmin(zeta,zeta_new)
                U_star <- loop_hit_and_run_multi_kernel(t(A), zeta_tilde, greaterequal, U, U_free, n_iter=n_iter_hit_and_run, kernel=ker)
                accept_or_not <- check_feasible(t(A), zeta, greaterequal, U_star)
                zeta <- zeta_new * accept_or_not + zeta * (1-accept_or_not)

                ##### Update beta #####
                mean_normal <- V_mat %*% t(X) %*% zeta
                beta <- mean_normal + L_mat %*% matrix(rnorm(p * d), p, d)
                
                ##### Save the beta draws for this iteration (after warm-ups) #####
                beta_samples[iter, , ] <- beta
              }
            )
            cat(n_iter, "iterations of MCMC with ", n_warmup, " warm-ups completed.\n", sep="")
            print(running_time)
            
            ##### Post-processing -------------------------------

            beta_post_mean <- apply(beta_samples[(n_warmup + 1):n_iter, , ], c(2, 3), mean)
            rmse_beta <- RMSE(beta_post_mean, beta_true)
            cat("RMSE(beta): ", rmse_beta, "\n", sep="")
            
            if (button_save){
              if (method == "exponential"){
                RMSE_table_exponential[repetition, m, d] = rmse_beta
                runtime_table_exponential[repetition, m, d] = running_time[3]
                cputime_table_exponential[repetition, m, d] = running_time[1]
                # Save beta_samples and beta_true
                setwd("/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Constraints/Results/Simulation_MHWG_exponential")
                beta_samples_filename <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d__kernelexponential__beta_samples.rds", repetition, n_iter, n, d, p, m)
                beta_true_filename <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d__kernelexponential__beta_true.rds", repetition, n_iter, n, d, p, m)
                saveRDS(beta_samples, file = beta_samples_filename)
                saveRDS(beta_true, file = beta_true_filename)
              } else if (method == "half_gaussian"){
                RMSE_table_halfgaussian[repetition, m, d] = rmse_beta
                runtime_table_halfgaussian[repetition, m, d] = running_time[3]
                cputime_table_halfgaussian[repetition, m, d] = running_time[1]
                # Save beta_samples and beta_true
                setwd("/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Constraints/Results/Simulation_MHWG_halfgaussian")
                beta_samples_filename <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d__kernelhalfgaussian__beta_samples.rds", repetition, n_iter, n, d, p, m)
                beta_true_filename <- sprintf("rep%d_iter%d_n%d_d%d_p%d_m%d__kernelhalfgaussian__beta_true.rds", repetition, n_iter, n, d, p, m)
                saveRDS(beta_samples, file = beta_samples_filename)
                saveRDS(beta_true, file = beta_true_filename)
              }
            }
            
            if (button_plot){
              trace_plot(beta_samples, p, d, n_warmup, n_iter)
              ACF_plot(beta_samples, p, d, n_warmup, n_iter, thin=n_thin)
            }
          }
        }
      }
    }
  }
}


setwd("/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Constraints/Results/Simulation_MHWG_exponential")
saveRDS(RMSE_table_exponential, file = "RMSE_table_exponential")
saveRDS(runtime_table_exponential, file = "runtime_table_exponential")
saveRDS(cputime_table_exponential, file = "cputime_table_exponential")

setwd("/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Constraints/Results/Simulation_MHWG_halfgaussian")
saveRDS(RMSE_table_halfgaussian, file = "RMSE_table_halfgaussian")
saveRDS(runtime_table_halfgaussian, file = "runtime_table_halfgaussian")
saveRDS(cputime_table_halfgaussian, file = "cputime_table_halfgaussian")


for (r in 1:n_rep){
  print(RMSE_table_exponential[r,m_list,d_list])
  print(runtime_table_exponential[r,m_list,d_list])
  print(cputime_table_exponential[r,m_list,d_list])
}
for (r in 1:n_rep){
  print(RMSE_table_halfgaussian[r,m_list,d_list])
  print(runtime_table_halfgaussian[r,m_list,d_list])
  print(cputime_table_halfgaussian[r,m_list,d_list])
}
