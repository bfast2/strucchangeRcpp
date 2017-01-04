#ifndef SC_ROOTMATRIX_H
#define SC_ROOTMATRIX_H

#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]

arma::mat sc_cpp_rootmatrix(const arma::mat& X);

arma::mat sc_cpp_rootmatrix_cross(const arma::mat& X);



#endif 