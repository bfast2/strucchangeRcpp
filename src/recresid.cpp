#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]
  
#include <iostream>
  
using namespace Rcpp;





//' Computation of recursive residuals in C++
//' @param X design matrix
//' @param y response vector
//' @param start integer (1-based) index of the first observation to compute recursive residuals
//' @param end integer (1-based) index of the last observation to compute recursive residuals
//' @param tol tolerance in the computation of recursive model coefficients
//' @return vector containing the recursive residuals 
//' @seealso \code{\link{recresid}} and \code{\link{recresid.default}}
arma::vec sc_cpp_recresid_arma(const arma::mat& X, const arma::vec& y,  unsigned int start, unsigned int end, const double& tol, const double& rcond_min) {
  if(!(start > X.n_cols && start <= X.n_rows)) stop("Invalid start");
  if(!(end >= start && end <= X.n_rows)) stop("Invalid end");
  --start;
  --end;
  int n=end; // n is not the number of rows but the last element index, i.e., nrows-1
  int q=start-1;
  int k = X.n_cols;
  arma::vec rval = arma::vec(n-q, arma::fill::zeros);
  
  // If the current submatrix of X has reciprocal conditioning 
  // number less then the following constant, the ordinary R 
  // implementation with column-pivoting QR decomposition 
  // is used. Otherwise, Armadillo functions solve() and 
  // inv_sympd() are used. This applies only to the first 
  // iterations as long as check is true. 

  arma::vec cur_y = y.subvec(0, q);
  arma::mat cur_X = X.submat(0, 0, q, k-1);
  arma::mat X1;
  arma::colvec cur_coef_full;
  
  Environment env_stats("package:stats");
  Function asNamespace("asNamespace");
  Environment env_sc = asNamespace("strucchangeRcpp");
  Function fXinv0 =  env_sc[".Xinv0"];
  Function fcoef0 =  env_sc[".coef0"];
  Function lmfit = env_stats["lm.fit"];
  
  
  // Decide whether to use Armadillo or R depending on the reciprocal conditioning number of cur_X
  if (1/arma::cond(cur_X) >= rcond_min) {
    arma::solve(cur_coef_full, cur_X, cur_y, arma::solve_opts::no_approx);
    arma::inv_sympd(X1, trans(cur_X) * cur_X);
  }
  else {
    List fitted = lmfit(Named("x", cur_X), Named("y", cur_y));
    NumericMatrix qrinv = fXinv0(fitted);
    X1 =  as<arma::mat>(qrinv);
    cur_coef_full  = as<arma::colvec>(fcoef0(fitted));
  }
  
  
  arma::colvec cur_coef = cur_coef_full; 
  arma::mat xr = X.row(q+1); 
  arma::vec fr = (1 + xr * X1  * trans(xr));
  rval(0) = as_scalar((y(q+1) - xr * cur_coef)/arma::sqrt(fr));
  bool check = true;
  if((q+1) < n)
  {
    for(int r=q+2; r<=n; ++r) {
      X1 -= (X1 * trans(xr) * xr * X1)/as_scalar(fr);
      cur_coef += X1 * trans(xr) * rval(r-q-2) * sqrt(fr); 
      
      if (check) {
        cur_y = y.subvec(0, r-1);
        cur_X = X.submat(0, 0, r-1, k-1);
        
        // Decide whether to use Armadillo or R depending in the reciprocal conditioning number of cur_X
        if (1/arma::cond(cur_X) >= rcond_min) {
          arma::solve(cur_coef_full, cur_X, cur_y, arma::solve_opts::no_approx);
          arma::inv_sympd(X1, trans(cur_X) * cur_X);
          bool nona = arma::is_finite( cur_coef) && arma::is_finite( cur_coef_full );
          if(nona && approx_equal(cur_coef_full,cur_coef, "absdiff", tol)) {
            check = false;
          } 
        }
        else {
          List fitted = lmfit(Named("x", cur_X), Named("y", cur_y));
          NumericMatrix qrinv = fXinv0(fitted);
          X1 =  as<arma::mat>(qrinv);
          arma::colvec coef1 = as<arma::colvec>(fitted["coefficients"]);
          bool nona = arma::is_finite( coef1 ) && arma::is_finite( cur_coef_full );
          if(nona && approx_equal(coef1,cur_coef, "absdiff", tol)) {
            check = false;
          } 
          cur_coef_full  = as<arma::colvec>(fcoef0(fitted));
        }
        
        cur_coef = cur_coef_full; 
      }
      xr = X.row(r); // This a a row 
      fr = (1 + xr * X1  * trans(xr));
      rval(r-q-1) = as_scalar((y(r) - xr * cur_coef)/arma::sqrt(fr));
    }
  }
  return rval;
}





//' Computation of recursive residuals in C++
//' @param X design matrix
//' @param y response vector
//' @param start integer (1-based) index of the first observation to compute recursive residuals
//' @param end integer (1-based) index of the last observation to compute recursive residuals
//' @param tol tolerance in the computation of recursive model coefficients
//' @return vector containing the recursive residuals 
//' @seealso \code{\link{recresid}} and \code{\link{recresid.default}}
// [[Rcpp::export(name = ".sc_cpp_recresid")]]
NumericVector sc_cpp_recresid(const arma::mat& X, const arma::vec& y,  unsigned int start, unsigned int end, const double& tol, const double& rcond_min) {
  arma::vec rval = sc_cpp_recresid_arma(X,y,start,end,tol,rcond_min);
  return  Rcpp::NumericVector(rval.begin(), rval.end());
}




