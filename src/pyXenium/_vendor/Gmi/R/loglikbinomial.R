#' Binomial Log-Likelihood Function
#'
#' @param X A numeric matrix of predictors, with dimensions \code{n x p}.
#' @param y A binary response vector of length \code{n}, with values in \code{0} or \code{1}.
#' @param beta A numeric coefficient vector of length \code{p}.
#'
#' @returns A single numeric value representing the binomial log-likelihood
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(100), nrow = 20)
#' beta <- runif(ncol(X))
#' eta <- X %*% beta
#' prob <- Sigmoid(eta)
#' y <- rbinom(20, 1, prob)
#' loglikbinomial(X, y, beta)
#' @export
loglikbinomial <- function(X, y, beta) {
  link <- as.vector(X %*% beta)
  return(2 * sum(log(1 + exp(link)) - y * link))
}
