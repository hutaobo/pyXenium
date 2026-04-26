#include <RcppEigen.h>
#include <Rcpp.h>
using namespace Eigen;
using namespace std;

//[[Rcpp::export]]
SEXP eigen_inv(Eigen::MatrixXd A){
  Eigen::MatrixXd C = A.inverse();
  return Rcpp::wrap(C);
}
