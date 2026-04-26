#' Generate Simulated Data
#'
#' @param seed An integer seed value
#' for reproducibility of the response variable generation.
#' @param n Integer. Number of observations to generate.
#' @param p Integer. Number of predictors.
#' @param alpha A numeric vector of length \code{p}, representing the coefficients for the main effects.
#' @param gamma A numeric vector representing the coefficients
#'  for all two-way interaction terms. The number of interactions is \code{choose(p, 2)}.
#' @param beta A numeric scalar. Intercept term in the logistic model.
#' @param rho Numeric value between -1 and 1. Correlation parameter for predictors.
#'
#' @returns A list containing:
#' \describe{
#'   \item{\code{x}}{An \code{n x p} matrix of standardized covariates.}
#'   \item{\code{xx}}{An \code{n x choose(p, 2)} matrix of all two-way interaction terms.}
#'   \item{\code{Y}}{A binary response vector of length \code{n} generated from a logistic model.}
#' }
#' @export
gendata <- function(seed, n, p, alpha, gamma, beta, rho) {
  set.seed(1)
  Sigma <- rho^(abs(outer(1:p, 1:p, "-")))
  x <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  x <- scale(x, center = TRUE, scale = TRUE)
  # generate Y
  # generate all possible two-way interaction term (matrix)
  xx <- t(apply(x, 1, combn, 2, prod))
  # generate theta:intercept
  theta <- beta + x %*% alpha + xx %*% gamma
  set.seed(seed)
  Y <- rbinom(n, 1, Sigmoid(theta))
  return(list(x = x, xx = xx, Y = Y))
}
