#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]

#include <iostream>

#include "rootmatrix.h"

using namespace Rcpp;


//' Computes the empirical fluctuation process according to moving OLS estimates (type ME) 
//' @param X design matrix
//' @param y response vector
//' @param rescale boolean argument defining whether or not the estimates will be standardized
//' @param h bandwith of the process
//' @return square root of X
// [[Rcpp::export(name = ".sc_cpp_efp_process_me")]]
List sc_cpp_efp_process_me(const arma::mat& X,const arma::vec& y, bool rescale, double h) {
  int n = X.n_rows;
  int k = X.n_cols;
  arma::colvec coef_hat;
  arma::solve(coef_hat, X, y, arma::solve_opts::no_approx);
  arma::colvec resid_hat = y - X*coef_hat; 
  double sigma = sqrt(arma::as_scalar(arma::trans(resid_hat)*resid_hat/(n-k)));
  int nh = floor(n*h);
  arma::mat process(k, n-nh+1, arma::fill::zeros);
  arma::mat Q12 = sc_cpp_rootmatrix_cross(X)/sqrt(n);

  for(int i=0; i <= n-nh; ++i) {
    arma::mat Xsub = X.submat(i, 0, i+nh-1, X.n_cols - 1);
    if (rescale) {
      arma::mat Qnh12 = sc_cpp_rootmatrix_cross(Xsub)/sqrt(nh);
      process.col(i) = Qnh12 * (arma::solve(Xsub, y.subvec(i,i+nh-1), arma::solve_opts::no_approx) - coef_hat);
    } 
    else {
      process.col(i) = Q12 * (arma::solve(Xsub, y.subvec(i,i+nh-1), arma::solve_opts::no_approx) - coef_hat);
    }
    
  }
  process = nh*trans(process)/(sqrt(n)*sigma); 
  
  return List::create(Named("process") = process,
                      Named("Q12")     = Q12);
}




//' Computes the empirical fluctuation process according to recursive OLS estimates (type RE) 
//' @param X design matrix
//' @param y response vector
//' @param rescale boolean argument defining whether or not the estimates will be standardized
//' @param h bandwith of the process
//' @return square root of X
// [[Rcpp::export(name = ".sc_cpp_efp_process_re")]]
List sc_cpp_efp_process_re(const arma::mat& X,const arma::vec& y, bool rescale) {
  int n = X.n_rows;
  int k = X.n_cols;
  arma::colvec coef_hat;
  arma::solve(coef_hat, X, y, arma::solve_opts::no_approx);
  arma::colvec resid_hat = y - X*coef_hat; 
  double sigma = sqrt(arma::as_scalar(arma::trans(resid_hat)*resid_hat/(n-k)));

  arma::mat process(k, n-k+2, arma::fill::zeros);
  arma::mat Q12 = sc_cpp_rootmatrix_cross(X)/sqrt(n);
  for(int i=k; i <= n-1; ++i) {
    arma::mat Xsub = X.submat(0, 0, i-1, X.n_cols - 1);
    if (rescale) {
      arma::mat Qi12 = sc_cpp_rootmatrix_cross(Xsub)/sqrt(i);
      process.col(i-k+1) = Qi12 * (arma::solve(Xsub, y.subvec(0,i-1), arma::solve_opts::no_approx) - coef_hat);
    } 
    else {
      process.col(i-k+1) = Q12 * (arma::solve(Xsub, y.subvec(0,i-1), arma::solve_opts::no_approx) - coef_hat);
    }
  }
  arma::vec v= arma::linspace<arma::vec>( k-1, n, n-(k-1)+1);
  arma::mat tp = trans(process);
  tp = (tp.each_col() % v)/(sqrt(n)*sigma);
  return List::create(Named("process") = tp,
                      Named("Q12")     = Q12);
}



