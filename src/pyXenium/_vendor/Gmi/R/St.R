#' Soft Thresholding Function
#'
#' @param x A numeric vector to be thresholded.
#' @param lambda non-negative threshold parameter
#'
#' @returns A numeric vector or scalar of the same shape as `x`, where each element has been soft-thresholded.
#' @export
#' @examples
#' St(c(-3, -1, 0, 1), 1)
St <- function(x, lambda) {
  # soft threshholding
  val <- sign(x) * pmax(abs(x) - lambda, 0)
  return(val)
}
