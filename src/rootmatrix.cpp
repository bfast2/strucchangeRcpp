#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]
  
#include <iostream>
#include "rootmatrix.h"
  
using namespace Rcpp;




//' Computes the square root of a symetric positive definite matrix
//' @param X symetric positive definite matrix
//' @return square root of X
// [[Rcpp::export(name = ".sc_cpp_rootmatrix")]]
arma::mat sc_cpp_rootmatrix(const arma::mat& X) {
  arma::mat SQRT_X = arma::sqrtmat_sympd(X) ;
  //bool success = arma::sqrtmat_sympd(SQRT_X,X);
  // if (!success) {
  //   Environment env_sc("package:strucchangeRcpp");
  //   Function rootmatrix =  env_sc[".root.matrix"];
  //   NumericMatrix out = rootmatrix(X);
  //   return as<arma::mat>(out);
  // }
  return SQRT_X;
}



//' Computes the square root of the Gramian matrix t(X) %*% X
//' @param X a matrix
//' @return square root of t(X) %*% X
// [[Rcpp::export(name = ".sc_cpp_rootmatrix_cross")]]
arma::mat sc_cpp_rootmatrix_cross(const arma::mat& X) {
  arma::mat XtX = arma::trans(X) * X;
  return sc_cpp_rootmatrix(XtX);
}


