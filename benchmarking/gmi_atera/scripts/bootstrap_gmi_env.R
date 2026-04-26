repos <- c(CRAN = "https://cloud.r-project.org")
cpcg_archive <- "https://cran.r-project.org/src/contrib/Archive/cPCG/cPCG_1.0.tar.gz"

missing_packages <- function(packages) {
  packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
}

cran_packages <- c("MASS", "Rcpp", "RcppArmadillo", "RcppEigen")
missing <- missing_packages(cran_packages)
if (length(missing) > 0) {
  install.packages(missing, repos = repos)
}

if (!requireNamespace("cPCG", quietly = TRUE)) {
  install.packages("cPCG", repos = repos)
}
if (!requireNamespace("cPCG", quietly = TRUE)) {
  install.packages(cpcg_archive, repos = NULL, type = "source")
}

missing <- missing_packages(c(cran_packages, "cPCG"))
if (length(missing) > 0) {
  stop("Missing required R packages after bootstrap: ", paste(missing, collapse = ", "))
}

vendor <- normalizePath(file.path("src", "pyXenium", "_vendor", "Gmi"), mustWork = TRUE)
install.packages(vendor, repos = NULL, type = "source")
if (!requireNamespace("Gmi", quietly = TRUE)) {
  stop("Vendored Gmi package did not install from local path: ", vendor)
}
