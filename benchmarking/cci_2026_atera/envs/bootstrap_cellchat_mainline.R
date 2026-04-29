args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected positional argument: %s", key), call. = FALSE)
    }
    name <- substring(key, 3)
    if (i == length(args) || startsWith(args[[i + 1]], "--")) {
      parsed[[name]] <- TRUE
      i <- i + 1
    } else {
      parsed[[name]] <- args[[i + 1]]
      i <- i + 2
    }
  }
  parsed
}

ensure_cran <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

ensure_cran_min_version <- function(pkg, version) {
  if (!requireNamespace(pkg, quietly = TRUE) || utils::packageVersion(pkg) < version) {
    install.packages(pkg)
  }
}

ensure_bioc <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
}

opts <- parse_args(args)
output_json <- opts[["output-json"]]

options(repos = c(CRAN = "https://cloud.r-project.org"))

ensure_cran("jsonlite")
ensure_cran("BiocManager")
ensure_cran("remotes")
ensure_cran_min_version("NMF", "0.23.0")

bioc_pkgs <- c("BiocGenerics", "Biobase", "BiocNeighbors", "ComplexHeatmap")
for (pkg in bioc_pkgs) {
  ensure_bioc(pkg)
}

if (!requireNamespace("CellChat", quietly = TRUE)) {
  remotes::install_github("jinworks/CellChat", upgrade = "never")
}

status <- list(
  method = "cellchat",
  status = if (requireNamespace("CellChat", quietly = TRUE)) "ready" else "failed",
  package_versions = list(
    R = R.version.string,
    CellChat = if (requireNamespace("CellChat", quietly = TRUE)) as.character(utils::packageVersion("CellChat")) else NULL,
    jsonlite = as.character(utils::packageVersion("jsonlite")),
    BiocManager = as.character(utils::packageVersion("BiocManager")),
    remotes = as.character(utils::packageVersion("remotes"))
  )
)

if (!is.null(output_json) && nzchar(output_json)) {
  dir.create(dirname(output_json), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(status, output_json, pretty = TRUE, auto_unbox = TRUE, null = "null")
}

cat(jsonlite::toJSON(status, pretty = TRUE, auto_unbox = TRUE, null = "null"))

if (!identical(status$status, "ready")) {
  quit(status = 2, save = "no")
}
