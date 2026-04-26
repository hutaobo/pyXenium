#' Group Similar Values by Threshold
#'
#' @param x A numeric vector to be grouped.
#' @param threshold A non-negative numeric value. Elements in `x` within
#'
#' @returns An integer vector of the same length as `x`, where each element
#'          indicates the group index that the corresponding value in `x` belongs to.
#' @examples
#' x <- c(0.01, 0.02, 0.5, 0.52, 0.51, 1.5)
#' Indexgroup(x, threshold = 0.05)
#' @export
Indexgroup <- function(x, threshold) {
  if (length(x) == 1) {
    return(1)
  }
  group <- rep(0, length(x))
  x0 <- x
  t <- 1
  while (length(x) != 0) {
    diff <- x[1] - x
    loc <- which(abs(diff) <= threshold)
    set <- which(x0 %in% unique(x[loc]))
    group[set] <- t
    t <- t + 1
    x <- x[-loc]
  }
  return(group)
}
