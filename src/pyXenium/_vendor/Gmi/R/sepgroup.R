#' Group Similar Elements in a Numeric Vector
#'
#' This function partitions a numeric vector into groups where each group contains values that are
#' within a specified threshold of each other. Two elements belong to the same group if their
#' absolute difference is less than or equal to `threshold`.
#' @param x A numeric vector.
#' @param threshold A non-negative numeric value specifying the maximum distance allowed between elements to be considered in the same group.
#'
#' @returns A list of integer vectors. Each element of the list contains the indices of the original vector `x` that form a group.
#' @examples
#' Sepgroup(c(1.0, 1.1, 2.5, 2.6, 5.0), threshold = 0.2)
#' @export
Sepgroup <- function(x, threshold) {
  if (length(x) == 1) {
    return(x)
  }
  group <- c()
  x0 <- x
  i <- 1
  while (i <= length(x)) {
    diff <- x[i] - x
    loc <- which(abs(diff) <= threshold, x)
    set <- which(x0 %in% unique(x[loc]))
    group <- append(group, list(set))
    x <- x[-loc]
    if (i == 0) {
      break
    }
  }
  return(group)
}
