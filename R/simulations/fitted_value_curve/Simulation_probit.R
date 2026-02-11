# ----------------------------------------
# Bayesian Multivariate Probit Regression with ILP Mapping
# ----------------------------------------

rm(list = ls())

# Load required packages
library(truncnorm)
library(lpSolve)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)  # for side-by-side plots

# Set seed for reproducibility
try = 1234
set.seed(try)

# ----------------------------------------
# Simulate Data
# ----------------------------------------
n <- 100  # number of observations
p <- 2     # number of predictors (including intercept)
d <- 2     # number of outcome dimensions

# Design matrix X with intercept and one continuous covariate
X <- matrix(c(rep(1, n), sort(rnorm(n))), nrow = n, ncol = p, byrow = FALSE)

# True regression coefficients (p × d)
beta_true <- matrix(rnorm(p*d), nrow = p, ncol = d)
cat("True beta:\n")
print(beta_true)

# Latent response
zeta_true <- X %*% beta_true + matrix(rnorm(n * d), nrow = n, ncol = d)

# ILP constraint: y must satisfy y_1 + y_2 <= 1
A <- matrix(c(1, 1), ncol = 2)
b <- c(1)

# ILP mapping function (argmax c'y subject to Ay ≤ b and y ∈ {0,1}^d)
ilp_map <- function(c) {
  ans = c(0, 0)
  if (c[1] > max(c[2], 0)){
    ans[1] = 1
  }
  if (c[2] > max(c[1], 0)){
    ans[2] = 1
  }
  ans
}

# Observed binary responses via ILP mapping
y <- t(apply(zeta_true, 1, ilp_map))


setwd('/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/r_code')
source("run_algorithm_unconstrained.R")
source("run_algorithm_constrained.R")


# Save X, y, beta_true, beta_samples, and beta_samples_augmented
setwd('/Users/zhengyu/Desktop/UF/Research/Intergral_Linear_Programming/Results/Probit')
saveRDS(X, sprintf("X_seed%d.rds", try))
saveRDS(y, sprintf("y_seed%d.rds", try))

saveRDS(beta_true, sprintf("beta_true_seed%d.rds", try))
saveRDS(beta_samples, sprintf("beta_samples_unconstrained_seed%d.rds", try))
saveRDS(beta_samples_augmented, sprintf("beta_samples_constrained_seed%d.rds", try))









# ----------------------------------------
# Trace Plots
# ----------------------------------------
par(mfrow = c(1, 1))
for (j in 1:d) {
  for (k in 1:p) {
    plot(beta_samples[, k, j], type = "l",
         main = sprintf("Trace for beta[%d, %d]", k, j),
         xlab = "Iteration", ylab = sprintf("beta[%d, %d]", k, j))
    abline(h = beta_true[k, j], col = "red")
  }
}

for (j in 1:d) {
  for (k in 1:p) {
    plot(beta_samples_augmented[, k, j], type = "l",
         main = sprintf("Trace for beta[%d, %d] (ILP)", k, j),
         xlab = "Iteration", ylab = sprintf("beta[%d, %d]", k, j))
    abline(h = beta_true[k, j], col = "red")
  }
}




