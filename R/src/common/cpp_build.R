# Compile Rcpp dependencies (run from repo root)
if (!requireNamespace("Rcpp", quietly = TRUE)) {
  stop("Package 'Rcpp' is required. Please install it in your R environment.")
}

Rcpp::sourceCpp("R/src/cpp/hit_and_run.cpp")
Rcpp::sourceCpp("R/src/cpp/hit_and_run_augmented.cpp")

# If/when you add TUM:
# Rcpp::sourceCpp("R/src/cpp/tum_check.cpp")
