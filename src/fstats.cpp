#include <RcppArmadillo.h>
#include <Rcpp.h>
//[[Rcpp::depends(RcppArmadillo)]]

#include <iostream>


using namespace Rcpp;


//' Compute F statistics for a data window
//' @param X design matrix
//' @param y response vector
//' @param istart window start index (1-based, as in R)
//' @param iend window end index (1-based, as in R)
//' @param rcond_min minimum reciprocal conditioning number to use armadillo::solve
//' @return list with elements stats and sume2 where stats is a vector with F statistics and sume2 is the sum of squared residuals of the model using all data
// [[Rcpp::export(name = ".sc_cpp_fstats")]]
List sc_cpp_fstats(const arma::mat& X,const arma::vec& y,int istart, int iend, double& rcond_min) {
  Environment env_stats("package:stats");
  Function lmfit = env_stats["lm.fit"];
  
  size_t n = X.n_rows;
  size_t k = X.n_cols;
  size_t np = iend - istart + 1;
  arma::vec rval = arma::vec(np, arma::fill::zeros);
  
  // initial fit on the whole dataset
  double rss_all = 0.0;
  if (1/arma::cond(X) >= rcond_min) {
    arma::vec coef = arma::solve(X, y, arma::solve_opts::no_approx + arma::solve_opts::fast);
    rss_all = arma::as_scalar(arma::sum(arma::square(y - X*coef))); 
 }
 else {
   List fm = lmfit(Named("x", X), Named("y", y));
   rss_all  = arma::as_scalar(arma::sum(arma::square(as<arma::vec>(fm["residuals"]))));

   // arma::mat Q,R;
   // arma::qr(Q,R,X);
   // arma::vec d = arma::trans(Q) * y;
   // arma::vec coef = arma::solve(R,d, arma::solve_opts::no_approx);
   // Rcout << "Using QR: coefficients: ";
   // coef.print(Rcout);
   // rss_all = arma::as_scalar(arma::sum(arma::square(y - X*coef)));
 }
  
  uint32_t count_X2_arma = 0;
  
  for (int i = istart - 1; i <= iend - 1; ++i) {
    arma::mat X1 = X.submat(0,0,i,k-1);
    arma::mat X2 = X.submat(i+1,0,n-1,k-1);
    
    arma::colvec y1 = y.subvec(0,i);
    arma::colvec y2 = y.subvec(i+1,n-1);
    
    arma::vec coef1;
    arma::vec coef2;
    
    double rss1, rss2;
    if (1/arma::cond(X1) >= rcond_min) {
      arma::solve(coef1, X1, y1, arma::solve_opts::no_approx + arma::solve_opts::fast);
      rss1 = arma::as_scalar(arma::sum(arma::square(y1 - X1*coef1)));
    }
    else {
      List fm = lmfit(Named("x", X1), Named("y", y1));
      rss1  = arma::as_scalar(arma::sum(arma::square(as<arma::vec>(fm["residuals"]))));
      // arma::mat Q,R;
      // arma::qr(Q,R,X1);
      // arma::vec d = arma::trans(Q) * y1;
      // arma::solve(coef1,R,d, arma::solve_opts::no_approx);
      // rss1  = arma::as_scalar(arma::sum(arma::square(y1 - X1*coef1)));
    }
    if (1/arma::cond(X2) >= rcond_min) {
      arma::solve(coef2, X2, y2, arma::solve_opts::no_approx + arma::solve_opts::fast);
      rss2 = arma::as_scalar(arma::sum(arma::square(y2 - X2*coef2))); 
      count_X2_arma++;
    }
    else {
     List fm = lmfit(Named("x", X2), Named("y", y2));
     rss2  = arma::as_scalar(arma::sum(arma::square(as<arma::vec>(fm["residuals"]))));

     // arma::mat Q,R;
     // arma::qr(Q,R,X2);
     // arma::vec d = arma::trans(Q) * y2;
     // arma::solve(coef2,R,d, arma::solve_opts::no_approx);
     // rss2  = arma::as_scalar(arma::sum(arma::square(y2 - X2*coef2)));
    }
    double rss = rss1 + rss2;
    double s2 = rss / (n-2*k);
    rval(i - (istart - 1)) = (rss_all - rss) / s2;
  }

  return List::create(Named("stats")     = Rcpp::NumericVector(rval.begin(), rval.end()),
                      Named("sume2")     = rss_all);
  
}




