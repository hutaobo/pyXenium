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

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || is.na(x)) y else x
}

write_json <- function(path, payload) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(payload, path, pretty = TRUE, auto_unbox = TRUE, null = "null")
}

write_method_card <- function(output_dir, payload) {
  lines <- c(
    "# Method Card: CellChat",
    "",
    sprintf("- Status: `%s`", payload$status %||% "unknown"),
    sprintf("- Reason: `%s`", payload$reason %||% payload$error %||% "not recorded"),
    sprintf("- Reproduce: `%s`", payload$reproduce %||% "See run_summary.json and logs."),
    ""
  )
  writeLines(lines, file.path(output_dir, "method_card.md"))
}

fail_run <- function(output_dir, message, code = 2) {
  payload <- list(method = "cellchat", status = "failed", reason = message)
  write_json(file.path(output_dir, "run_summary.json"), payload)
  write_method_card(output_dir, payload)
  quit(status = code, save = "no")
}

standardize_cellchat <- function(raw, database_mode, artifact_path) {
  columns <- c(
    "method", "database_mode", "ligand", "receptor", "sender", "receiver",
    "score_raw", "score_std", "rank_within_method", "rank_fraction",
    "fdr_or_pvalue", "resolution", "spatial_support_type", "artifact_path"
  )
  if (is.null(raw) || nrow(raw) == 0) {
    empty <- as.data.frame(matrix(nrow = 0, ncol = length(columns)))
    colnames(empty) <- columns
    return(empty)
  }
  required <- c("source", "target", "ligand", "receptor")
  missing <- setdiff(required, colnames(raw))
  if (length(missing) > 0) {
    stop(sprintf("CellChat raw output is missing columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
  score_col <- if ("prob" %in% colnames(raw)) "prob" else if ("score" %in% colnames(raw)) "score" else NA
  if (is.na(score_col)) {
    stop("CellChat raw output must contain either 'prob' or 'score'.", call. = FALSE)
  }
  pvalue <- if ("pval" %in% colnames(raw)) suppressWarnings(as.numeric(raw$pval)) else rep(NA_real_, nrow(raw))
  score <- suppressWarnings(as.numeric(raw[[score_col]]))
  score[is.na(score)] <- min(score, na.rm = TRUE)
  score[is.na(score)] <- 0
  order_idx <- order(-score, pvalue, na.last = TRUE)
  rank <- rep(NA_real_, length(score))
  rank[order_idx] <- seq_along(score)
  rank_fraction <- 1 - ((rank - 1) / length(score))
  standardized <- data.frame(
    method = "cellchat",
    database_mode = database_mode,
    ligand = as.character(raw$ligand),
    receptor = as.character(raw$receptor),
    sender = as.character(raw$source),
    receiver = as.character(raw$target),
    score_raw = score,
    score_std = rank_fraction,
    rank_within_method = rank,
    rank_fraction = rank_fraction,
    fdr_or_pvalue = pvalue,
    resolution = "celltype_pair",
    spatial_support_type = "spatial_cellchat",
    artifact_path = artifact_path,
    stringsAsFactors = FALSE
  )
  standardized[order(standardized$rank_within_method), columns]
}

create_common_db <- function(lr_db) {
  data(CellChatDB.human, package = "CellChat", envir = environment())
  db <- CellChatDB.human
  template <- db$interaction[0, , drop = FALSE]
  interaction <- as.data.frame(matrix(NA, nrow = nrow(lr_db), ncol = ncol(template)), stringsAsFactors = FALSE)
  colnames(interaction) <- colnames(template)
  interaction$interaction_name <- paste(lr_db$ligand, lr_db$receptor, sep = "_")
  if ("interaction_name_2" %in% colnames(interaction)) {
    interaction$interaction_name_2 <- paste(lr_db$ligand, lr_db$receptor, sep = " - ")
  }
  if ("pathway_name" %in% colnames(interaction)) {
    interaction$pathway_name <- lr_db$pathway %||% "custom"
  }
  interaction$ligand <- lr_db$ligand
  interaction$receptor <- lr_db$receptor
  if ("annotation" %in% colnames(interaction)) {
    interaction$annotation <- "Secreted Signaling"
  }
  if ("evidence" %in% colnames(interaction)) {
    interaction$evidence <- "pyXenium common-db"
  }
  db$interaction <- interaction
  db
}

opts <- parse_args(args)
output_dir <- opts[["output-dir"]] %||% stop("--output-dir is required", call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
raw_dir <- file.path(output_dir, "raw")
dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("jsonlite", quietly = TRUE)) fail_run(output_dir, "R package jsonlite is required.")
if (!requireNamespace("Matrix", quietly = TRUE)) fail_run(output_dir, "R package Matrix is required.")
if (!requireNamespace("CellChat", quietly = TRUE)) fail_run(output_dir, "R package CellChat/SpatialCellChat is required.")

input_manifest <- opts[["input-manifest"]] %||% fail_run(output_dir, "--input-manifest is required.")
database_mode <- opts[["database-mode"]] %||% "common-db"
phase <- opts[["phase"]] %||% "smoke"
max_lr_pairs <- opts[["max-lr-pairs"]]
manifest <- jsonlite::fromJSON(input_manifest, simplifyVector = FALSE)
if (phase == "full") {
  if (length(manifest$full_bundle) == 0) {
    fail_run(output_dir, "Phase 'full' was requested, but the input manifest does not contain a full sparse bundle.")
  }
  bundle <- manifest$full_bundle
} else {
  bundle <- manifest$smoke_bundle
}
if (is.null(bundle$counts_symbol_mtx)) fail_run(output_dir, "Sparse counts bundle is missing counts_symbol_mtx.")

counts <- Matrix::readMM(bundle$counts_symbol_mtx)
barcodes <- readLines(bundle$barcodes_tsv)
genes <- read.delim(bundle$genes_tsv, sep = "\t", stringsAsFactors = FALSE)
meta <- read.delim(bundle$meta_tsv, sep = "\t", stringsAsFactors = FALSE)
coords <- read.delim(bundle$coords_tsv, sep = "\t", stringsAsFactors = FALSE)

gene_symbols <- if ("gene_symbol" %in% colnames(genes)) genes$gene_symbol else genes[[1]]
rownames(counts) <- make.unique(as.character(gene_symbols))
colnames(counts) <- as.character(barcodes)
rownames(meta) <- as.character(meta$cell_id)
meta <- meta[colnames(counts), , drop = FALSE]
coords <- coords[match(colnames(counts), coords$cell_id), , drop = FALSE]
coordinates <- as.matrix(coords[, c("x", "y")])
rownames(coordinates) <- colnames(counts)

suppressPackageStartupMessages(library(CellChat))
cellchat <- tryCatch(
  createCellChat(object = counts, meta = meta, group.by = "cell_type", datatype = "spatial", coordinates = coordinates),
  error = function(e) createCellChat(object = counts, meta = meta, group.by = "cell_type")
)

if (database_mode %in% c("common", "common-db", "smoke-panel")) {
  db_path <- if (database_mode == "smoke-panel") manifest$atera_smoke_panel_tsv else manifest$lr_db_common_tsv
  lr_db <- read.delim(db_path, sep = "\t", stringsAsFactors = FALSE)
  if (!"pathway" %in% colnames(lr_db)) lr_db$pathway <- "custom"
  if (!is.null(max_lr_pairs)) lr_db <- head(lr_db, as.integer(max_lr_pairs))
  cellchat@DB <- create_common_db(lr_db)
} else {
  data(CellChatDB.human, package = "CellChat", envir = environment())
  cellchat@DB <- CellChatDB.human
}

started <- Sys.time()
spatial_fallback <- NULL
result <- tryCatch({
  cellchat <- subsetData(cellchat)
  cellchat <- identifyOverExpressedGenes(cellchat)
  cellchat <- identifyOverExpressedInteractions(cellchat)
  cellchat <- tryCatch(
    computeCommunProb(cellchat, type = "truncatedMean", distance.use = TRUE),
    error = function(e) {
      spatial_fallback <<- conditionMessage(e)
      computeCommunProb(cellchat, type = "truncatedMean", distance.use = FALSE)
    }
  )
  cellchat <- filterCommunication(cellchat, min.cells = 10)
  cellchat <- computeCommunProbPathway(cellchat)
  raw <- subsetCommunication(cellchat)
  list(cellchat = cellchat, raw = raw)
}, error = function(e) {
  fail_run(output_dir, conditionMessage(e))
})

raw_path <- file.path(raw_dir, "cellchat_communication.tsv")
write.table(result$raw, raw_path, sep = "\t", quote = FALSE, row.names = FALSE)
saveRDS(result$cellchat, file.path(raw_dir, "cellchat_object.rds"))
standardized <- standardize_cellchat(result$raw, database_mode = database_mode, artifact_path = raw_dir)
standardized_path <- file.path(output_dir, "cellchat_standardized.tsv")
write.table(standardized, standardized_path, sep = "\t", quote = FALSE, row.names = FALSE)

summary <- list(
  method = "cellchat",
  status = "success",
  database_mode = database_mode,
  phase = phase,
  raw_tsv = raw_path,
  standardized_tsv = standardized_path,
  n_rows = nrow(standardized),
  elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
  spatial_fallback = spatial_fallback,
  package_versions = list(
    R = R.version.string,
    CellChat = as.character(utils::packageVersion("CellChat")),
    Matrix = as.character(utils::packageVersion("Matrix")),
    jsonlite = as.character(utils::packageVersion("jsonlite"))
  )
)
write_json(file.path(output_dir, "run_summary.json"), summary)
cat(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE))
