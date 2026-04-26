#include <RcppEigen.h>
#include <Rcpp.h>
using namespace Eigen;
using namespace std;

//[[Rcpp::export]]
SEXP eigen_matmul(Eigen::MatrixXd A, Eigen::MatrixXd B){
  Eigen::MatrixXd C = A * B;

  return Rcpp::wrap(C);
}



