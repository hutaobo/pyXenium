#' Sigmoid Activation Function
#'
#' @param z A numeric vector.
#'
#' @returns A numeric vector with the sigmoid transformation applied.
#' @export
#'
#' @examples
#' Sigmoid(0)
Sigmoid <- function(z) {
  prob_mat <- 1 / (1 + exp(-z))
  prob_min <- 1e-05
  prob_max <- 1 - prob_min
  prob_mat <- pmin(pmax(prob_mat, prob_min), prob_max)
  return(c(prob_mat))
}
