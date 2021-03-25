#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]

#include <iostream>
#include <vector>
#include "recresid.h"
using namespace Rcpp;





// [[Rcpp::export(name = ".sc_cpp_rssi")]]
arma::vec sc_cpp_rssi(const arma::vec& y, const arma::mat& X, int n, int i, const bool intercept_only, const double& tol, const double& rcond_min) {
  int k = X.n_cols;
  arma::vec ssr(n-i-1);
  arma::vec ysub = y.subvec(i-1, n-1);
  if (intercept_only) { 
    
    arma::vec v123 = arma::cumsum(arma::ones<arma::vec>(n-i+1));
    arma::vec A = ysub - arma::cumsum(ysub) / v123;
    ssr =  A.subvec(1,n-i) % arma::sqrt(1 + 1 / (v123.subvec(0,n-i-1)));
  }
  else {
    ssr = sc_cpp_recresid_arma(X.submat(i-1, 0, n-1, k-1 ), ysub, k+1, n-i+1, tol, rcond_min);
  }
  arma::vec out(n-i+1);
  out.fill(NA_REAL);
  out.subvec(k,n-i) = arma::cumsum(arma::square(ssr));
  return(out);
}



arma::mat sc_cpp_rssi_triang(const arma::vec& y, const arma::mat& X, int n, int h, const bool intercept_only, const double& tol, const double& rcond_min) {
  arma::mat out(n,n-h+1);
  out.fill(NA_REAL);
  for (int i=1; i<=n-h+1; ++i) {
    out.submat(0,i-1,n-i,i-1) = sc_cpp_rssi(y,X,n,i,intercept_only,tol, rcond_min);
  }
  return out;
}


// [[Rcpp::export(name = ".sc_cpp_rss")]]
double sc_cpp_rss(const arma::mat& rss_triang, const int i, const int j) {
   return rss_triang(j-i,i-1);
 }


// [[Rcpp::export(name = ".sc_cpp_extend_rss_table")]]
arma::mat sc_cpp_extend_rss_table(arma::mat& rss_table, const arma::mat& rss_triang, int n, int h, int breaks) {
  // extend table
  int n_rows = rss_table.n_rows;
  if (2*breaks <=  int(rss_table.n_cols)) {
    return rss_table;
  }
  arma::mat na(n_rows,2); 
  na.fill(NA_REAL);
  
  for (int m=rss_table.n_cols / 2 + 1; m <= breaks; ++m) {
    arma::mat my_rss_table = rss_table.submat(0, 2*(m-2), n_rows-1, 2*(m-2)+1);
    my_rss_table.insert_cols(2, na);
    for (int i=m*h; i<=n-h; ++i) {
      arma::vec break_rss_i( (i-h) - ((m-1)*h) + 1);
      for (int j=(m-1)*h; j<=i-h; ++j) {
        break_rss_i(j-((m-1)*h)) = my_rss_table(j-h, 1) + sc_cpp_rss(rss_triang,j+1,i);
      }
      int opt = arma::index_min(break_rss_i);
      my_rss_table(i-h, 2) = (m-1)*h + opt;
      my_rss_table(i-h, 3) = break_rss_i(opt);
      
    }
    rss_table.insert_cols(rss_table.n_cols, my_rss_table.submat(0,2, n_rows-1, 3));
  }
  return rss_table;
}


//' ..
//' @param 
//' @param 
//' @param
//' @param 
//' @return 
// [[Rcpp::export(name = ".sc_cpp_construct_rss_table")]]
List sc_cpp_construct_rss_table(const arma::vec& y, const arma::mat& X, int n, int h, int breaks, const bool intercept_only, const double& tol, const double& rcond_min) {
  
  int n_rows = n-h-h+1;
  arma::mat rss_table(n_rows, 2);
  arma::mat rss_triang = sc_cpp_rssi_triang(y,X,n,h,intercept_only,tol, rcond_min);
  rss_table.col(0) = arma::linspace<arma::vec>(h,n-h,n_rows);
  for (int i=0; i<n_rows; ++i) {  
    rss_table(i,1) = sc_cpp_rss(rss_triang, 1,i+h);
  }
  // set col and rownames in R
  return List::create(Rcpp::Named("RSS.table") = sc_cpp_extend_rss_table(rss_table, rss_triang, n, h, breaks),
                      Rcpp::Named("RSS.triang") = rss_triang);
  
}


















