#' Convert Interaction Term Names to Corresponding Main Effect Indices
#'
#' convert an interaction name list into the corresponding main effect for enforcingstrong hierachy
#' @param interNameList A list of interaction term names.
#' @param p Integer. The total number of variables (dimensionality) in the design matrix.
#'
#' @returns An integer vector giving the indices of main effects involved in the interaction terms.
#' @examples
#' interNameList <- list("X1X3", "X2X4", "X1X2")
#' p <- 5
#' intertomain(interNameList, p)
#' @export
intertomain <- function(interNameList, p) {
  mainInd <- rep(0, p)
  for (i in 1:length(interNameList)) {
    interName <- interNameList[[i]]
    pair <- as.numeric(strsplit(interName, "X")[[1]][2:3])
    mainInd[pair[1]] <- 1
    mainInd[pair[2]] <- 1
  }
  return(which(mainInd == 1))
}
