#ifndef SC_RECRESID_H
#define SC_RECRESID_H


#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;



arma::vec sc_cpp_recresid_arma(const arma::mat& X, const arma::vec& y,  unsigned int start, unsigned int end, const double& tol);
  
NumericVector sc_cpp_recresid(const arma::mat& X, const arma::vec& y,  unsigned int start, unsigned int end, const double& tol);


#endif 