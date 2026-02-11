#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace arma;
// [[Rcpp::depends(RcppArmadillo)]]

// Recursive function to compute the determinant of an integer matrix using Laplace expansion.
int detInt(const arma::imat& A) {
  int n = A.n_rows;
  if(n == 1)
    return A(0, 0);
  if(n == 2)
    return A(0, 0)*A(1, 1) - A(0, 1)*A(1, 0);
  
  int determinant = 0;
  for (int p = 0; p < n; p++) {
    // Build submatrix by excluding row 0 and column p
    arma::imat subMat(n - 1, n - 1);
    int subi = 0;
    for (int i = 1; i < n; i++) {
      int subj = 0;
      for (int j = 0; j < n; j++) {
        if (j == p) continue;
        subMat(subi, subj) = A(i, j);
        subj++;
      }
      subi++;
    }
    int sign = (p % 2 == 0) ? 1 : -1;
    determinant += sign * A(0, p) * detInt(subMat);
  }
  return determinant;
}

// Helper function to generate all combinations of indices of length k from a total of n elements.
void combinations(int offset, int k, std::vector<int>& current, 
                  std::vector< std::vector<int> >& result, int n) {
  if (k == 0) {
    result.push_back(current);
    return;
  }
  for (int i = offset; i <= n - k; i++) {
    current.push_back(i);
    combinations(i + 1, k - 1, current, result, n);
    current.pop_back();
  }
}

// [[Rcpp::export]]
List checkTotallyUnimodularArma(arma::imat M) {
  int n = M.n_rows;
  int m = M.n_cols;
  int maxSquare = std::min(n, m);
  
  // Iterate over all square submatrices of size k x k for k = 1 ... min(n, m)
  for (int k = 1; k <= maxSquare; k++) {
    std::vector< std::vector<int> > rowComb;
    std::vector<int> current;
    combinations(0, k, current, rowComb, n);
    
    std::vector< std::vector<int> > colComb;
    current.clear();
    combinations(0, k, current, colComb, m);
    
    // Iterate over all combinations of row and column indices.
    for (auto &rows : rowComb) {
      arma::uvec rowIdx = conv_to<arma::uvec>::from(rows);
      for (auto &cols : colComb) {
        arma::uvec colIdx = conv_to<arma::uvec>::from(cols);
        arma::imat subMat = M.submat(rowIdx, colIdx);
        int d = detInt(subMat);
        // If the determinant is not in {-1, 0, 1}, return the indices and the value.
        if (d != 0 && d != 1 && d != -1) {
          // Convert indices from 0-based to 1-based.
          std::vector<int> rows_R, cols_R;
          for (int r : rows)
            rows_R.push_back(r + 1);
          for (int c : cols)
            cols_R.push_back(c + 1);
          
          return List::create(Named("isTUM") = false,
                              Named("rows") = rows_R,
                              Named("cols") = cols_R,
                              Named("determinant") = d);
        }
      }
    }
  }
  
  // If no violation found, return that the matrix is totally unimodular.
  return List::create(Named("isTUM") = true,
                      Named("rows") = R_NilValue,
                      Named("cols") = R_NilValue,
                      Named("determinant") = R_NilValue);
}
