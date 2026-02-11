rm(list=ls())

# -------------------------------
# Load data
# -------------------------------
setwd('/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Data_Application')
# Read duck_data.csv
ducks <- read.csv("duck_data.csv")

# Read A and y as numeric matrices
A <- as.matrix(read.csv("A_tilde_matrix.csv", header = FALSE))
y <- as.matrix(read.csv("Z_matrix.csv", header = FALSE))

# Set b (constraint vector)
b <- rep(1, nrow(A))



# -------------------------------
# Load dependent codes
# -------------------------------
setwd('../r_code')

library(bridgesampling)
library(mvtnorm)
library(truncnorm)
library(splines)
library("lpSolve")
library("lintools")
# use our own customized hit_and_run
library(Rcpp)
sourceCpp("hit_and_run.cpp")  # make sure this file contains the code from above
sourceCpp("tum_check.cpp")  # check TUM




# -------------------------------
# Pre-MCMC
# -------------------------------
n <- 18  # Number of weeks
d <- 339  # Number of edges in the original graph
m <- 95  # Number of constraints
K <- 2 # Number of groups (species)
M <- 3  # Length of alpha [DO NOT CHANGE]
kappa <- 5  # Degrees of freedom (number of basis functions)


# Construct C
construct_C_from_ducks <- function(species_vec) {
  species_levels <- unique(species_vec)  # preserve original order
  K <- length(species_levels)
  d <- length(species_vec)
  
  C <- matrix(0, nrow = K, ncol = d)
  
  for (j in 1:d) {
    k <- which(species_levels == species_vec[j])
    C[k, j] <- 1
  }
  
  return(list(C = C, s_list = rowSums(C), species_levels = species_levels))
}


# Group American Black Duck, Mallard, and Gadwall into the same group
# Create a new column in ducks dataframe for group
ducks$duck_group <- ifelse(ducks$duck_species %in% c("American Black Duck", "Mallard", "Gadwall"), "Dabbling Duck", "Diving Duck")
result <- construct_C_from_ducks(ducks$duck_group) # This function can be found in duck_matching.R
C <- result$C
s_list <- result$s_list
unique(ducks$duck_species)


# Construct W_list
W1 = matrix(1, n, d)
W2 = matrix(rep(ducks$duck_weight_male, n), n, d, byrow=TRUE)
W3 = matrix(rep(ducks$duck_weight_female, n), n, d, byrow=TRUE)

W_list = list(W1=W1, W2=W2, W3=W3)

# Construct B
B <- bs(seq(0, 1, length.out = n), df = kappa, intercept = TRUE)  # n x kappa matrix



# Find whether U can be non-zero
U_free <- (t(A%*%t(y) == b))*1
U <- matrix(rexp(n*m),n,m) * U_free


# -------------------------------
# Specify prior parameters for alpha and rho
# -------------------------------
tau_a <- 0.1
tau_rho <- 0.1



# -------------------------------
# Precompute constant matrices for updates
# -------------------------------
B_star <- t(B) %*% B + diag(tau_rho, kappa)
inv_B_star <- solve(B_star)
inv_B_star_BT <- inv_B_star %*% t(B)

# W_star matrix (now M x M)
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
# Gibbs sampler settings
# -------------------------------
n_iter <- 50000     # number of Gibbs iterations

rho_samples <- array(NA, dim = c(n_iter, kappa, K))  # to store rho draws
a_samples <- array(NA, dim = c(n_iter, M))  # to store a draws
zeta_samples <- array(NA, dim = c(n_iter, n, d))  # to store zeta draws


rho <- matrix(0, nrow = kappa, ncol = K)   # initial value for rho
a <- rnorm(M)   # initial value for a



# Initialize zeta
zeta <- matrix(NA, n, d)
UA = U%*%A
W_sum <- Reduce(`+`, lapply(1:M, function(m) a[m] * W_list[[m]]))
B_rho_C <- B %*% rho %*% C
mu <- W_sum + B_rho_C
for (j in 1:d) {
  # Set vectorized lower and upper bounds based on observed y
  lower_bound <- ifelse(y[, j] == 1, UA[,j], -Inf)
  upper_bound <- ifelse(y[, j] == 1, Inf, UA[,j])
  # Sample all n latent variables for outcome j in one call
  zeta[, j] <- rtruncnorm(n, a = lower_bound, b = upper_bound, mean = mu[,j], sd = 1)
}


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
  rho_samples[iter, , ] <- rho
  a_samples[iter, ] <- a
  zeta_samples[iter, , ] <- zeta
}


rho_samples_reduced <- rho_samples
a_samples_reduced <- a_samples

# Save the samples
setwd('../Results/Duck_matching')
rho_samples_filename <- sprintf("run2_duck_iter%d_n%d_kappa%d_K%d_rho_samples_reduced.rds", n_iter, n, kappa, K)
a_samples_filename <- sprintf("run2_duck_iter%d_n%d_kappa%d_K%d_a_samples_reduced.rds", n_iter, n, kappa, K)
zeta_samples_filename <- sprintf("run2_duck_iter%d_n%d_kappa%d_K%d_zeta_samples_reduced.rds", n_iter, n, kappa, K)
saveRDS(rho_samples_reduced, file = rho_samples_filename)
saveRDS(a_samples_reduced, file = a_samples_filename)
saveRDS(zeta_samples, file = zeta_samples_filename)

# Read the saved samples from complete and reduced models for comparison
rho_samples_full <- readRDS(sprintf("duck_iter%d_n%d_kappa%d_K7_rho_samples.rds", n_iter, n, kappa))
a_samples_full <- readRDS(sprintf("duck_iter%d_n%d_kappa%d_K7_a_samples.rds", n_iter, n, kappa))



# Check the dimensions of the samples
dim(rho_samples_full)  # should be (n_iter, kappa, K)
dim(a_samples_full)    # should be (n_iter, M)
dim(rho_samples_reduced)  # should be (n_iter, kappa, K)
dim(a_samples_reduced)    # should be (n_iter, M)
























