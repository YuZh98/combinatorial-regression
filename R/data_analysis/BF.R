# Load the samples and construct necessary quantities
# (Assume the data frame 'ducks' is already loaded)
setwd("/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Results/Duck_matching")
a_samples_full <- readRDS("run2_duck_iter50000_n18_kappa5_K7_a_samples.rds")
rho_samples_full <- readRDS("run2_duck_iter50000_n18_kappa5_K7_rho_samples.rds")
zeta_samples_full <- readRDS("run2_duck_iter50000_n18_kappa5_K7_zeta_samples.rds")
a_samples_reduced <- readRDS("run2_duck_iter50000_n18_kappa5_K2_a_samples_reduced.rds")
rho_samples_reduced <- readRDS("run2_duck_iter50000_n18_kappa5_K2_rho_samples_reduced.rds")
zeta_samples_reduced <- readRDS("run2_duck_iter50000_n18_kappa5_K2_zeta_samples_reduced.rds")

sample_indeces <- seq(2000, 50000, by = 20)
a_samples_full <- a_samples_full[sample_indeces, ]
a_samples_reduced <- a_samples_reduced[sample_indeces, ]
rho_samples_full <- rho_samples_full[sample_indeces, , ]
rho_samples_reduced <- rho_samples_reduced[sample_indeces, , ]
zeta_samples_full <- zeta_samples_full[sample_indeces, , ]
zeta_samples_reduced <- zeta_samples_reduced[sample_indeces, , ]


W1 = matrix(1, n, d)
W2 = matrix(rep(ducks$duck_weight_male, n), n, d, byrow=TRUE)
W3 = matrix(rep(ducks$duck_weight_female, n), n, d, byrow=TRUE)
W_list = list(W1=W1, W2=W2, W3=W3)

write.csv(as.matrix(B), file = "B_matrix.csv", row.names = FALSE)

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

result <- construct_C_from_ducks(ducks$duck_species)
C_full <- result$C
s_list_full <- result$s_list

ducks$duck_group <- ifelse(ducks$duck_species %in% c("American Black Duck", "Mallard", "Gadwall"), 
                           "Dabbling Duck", 
                           "Diving Duck")
result <- construct_C_from_ducks(ducks$duck_group)
C_reduced <- result$C
s_list_reduced <- result$s_list

tau_rho <- 0.1






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
  a_var <- apply(a_samples, 2, var) + 1e-8  # regularize
  rho_mean <- apply(rho_samples, c(2, 3), mean)
  rho_var <- apply(rho_samples, c(2, 3), var) + 1e-8
  
  log_h <- numeric(N)
  log_posterior <- numeric(N)
  log_prior <- numeric(N)
  
  for (i in 1:N) {
    a_i <- a_samples[i, ]
    rho_i <- rho_samples[i, , ]         # kappa x K
    zeta_i <- zeta_samples[i, , ]       # n x d
    
    mu_i <- a_i[1] * W_list[[1]] +
      a_i[2] * W_list[[2]] +
      a_i[3] * W_list[[3]] +
      B %*% rho_i %*% C
    
    # log h: MVN(a_mean, diag(a_var)) x MVN(rho_mean, diag(rho_var))
    log_h[i] <- -0.5 * (
      d_a * log(2 * pi) + sum(log(a_var)) +
        d_rho * log(2 * pi) + sum(log(rho_var)) +
        sum((a_i - a_mean)^2 / a_var) +
        sum((rho_i - rho_mean)^2 / rho_var)
    )
    
    # log-likelihood (up to constant): N(mu, I)
    log_posterior[i] <- -0.5 * sum((zeta_i - mu_i)^2)
    
    # Correct prior: each column of rho has N(0, (tau_rho * s_k)^{-1} I)
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
  log_summand_i <- log_h - log_posterior - log_prior
  max_log <- max(log_summand_i)
  log_sum <- max_log + log(sum(exp(log_summand_i - max_log)))
  
  return(log_sum)
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

log_BF_dem <- log_BF_middle_calculator(
  a_samples = a_samples_full,
  rho_samples = rho_samples_full,
  zeta_samples = zeta_samples_full,
  s_list = s_list_full,
  W_list = W_list,
  B = B,
  C = C_full,
  tau_rho = tau_rho
)

log_BF <- log_BF_num - log_BF_dem
BF <- exp(log_BF)

cat("Log Bayes Factor:", round(log_BF, 3), "\n")
cat("Bayes Factor in favor of the full model:", round(BF, 3), "\n")


