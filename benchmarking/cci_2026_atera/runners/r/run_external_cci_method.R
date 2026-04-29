suppressPackageStartupMessages({
  library(jsonlite)
})

now_iso <- function() {
  format(Sys.time(), "%Y-%m-%dT%H:%M:%OS%z")
}

activate_method_renv <- function(method, input_manifest = NULL, output_dir = NULL) {
  benchmark_root <- Sys.getenv("PYX_CCI_BENCHMARK_ROOT", unset = NA_character_)
  if (is.na(benchmark_root) && !is.null(input_manifest) && !is.na(input_manifest)) {
    benchmark_root <- dirname(dirname(normalizePath(input_manifest, mustWork = FALSE)))
  }
  if (is.na(benchmark_root) && !is.null(output_dir) && !is.na(output_dir)) {
    benchmark_root <- dirname(dirname(dirname(normalizePath(output_dir, mustWork = FALSE))))
  }
  if (is.na(benchmark_root)) {
    return(invisible(FALSE))
  }
  project_dir <- file.path(benchmark_root, "envs", paste0("r-cci-", method, "_project"))
  if (!dir.exists(project_dir)) {
    return(invisible(FALSE))
  }
  setwd(project_dir)
  if (requireNamespace("renv", quietly = TRUE)) {
    renv::load(project = project_dir, quiet = TRUE)
  }
  invisible(TRUE)
}

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list(
    method = NULL,
    input_manifest = NULL,
    output_dir = NULL,
    database_mode = "common-db",
    phase = "smoke",
    max_cci_pairs = NULL,
    allow_expression_baseline = FALSE
  )
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    value <- if (i < length(args)) args[[i + 1]] else NA_character_
    if (key == "--method") out$method <- value
    if (key == "--input-manifest") out$input_manifest <- value
    if (key == "--output-dir") out$output_dir <- value
    if (key == "--database-mode") out$database_mode <- value
    if (key == "--phase") out$phase <- value
    if (key == "--max-cci-pairs") out$max_cci_pairs <- as.integer(value)
    if (key == "--allow-expression-baseline") {
      out$allow_expression_baseline <- TRUE
      i <- i + 1
      next
    }
    i <- i + if (startsWith(key, "--")) 2 else 1
  }
  if (is.null(out$method) || is.null(out$output_dir)) {
    stop("--method and --output-dir are required")
  }
  out
}

select_bundle <- function(manifest, phase) {
  if (!is.null(phase) && phase == "full" && !is.null(manifest$full_bundle)) {
    return(manifest$full_bundle)
  }
  if (!is.null(manifest$smoke_bundle)) {
    return(manifest$smoke_bundle)
  }
  if (!is.null(manifest$full_bundle)) {
    return(manifest$full_bundle)
  }
  stop("Manifest does not contain a usable sparse bundle")
}

read_first_column <- function(path) {
  read.table(path, sep = "\t", header = FALSE, stringsAsFactors = FALSE, quote = "", comment.char = "")[[1]]
}

rank_standardize <- function(df) {
  if (nrow(df) == 0) {
    df$score_std <- numeric(0)
    df$rank_within_method <- integer(0)
    return(df)
  }
  ord <- order(-df$score_raw, df$fdr_or_pvalue, na.last = TRUE)
  rank <- integer(nrow(df))
  rank[ord] <- seq_len(nrow(df))
  df$rank_within_method <- rank
  if (nrow(df) == 1) {
    df$score_std <- 1
  } else {
    df$score_std <- 1 - ((rank - 1) / (nrow(df) - 1))
  }
  df[order(df$rank_within_method), , drop = FALSE]
}

run_package_present_expression_baseline <- function(opts, available) {
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("Matrix package is required for sparse bundle expression baseline")
  }
  started <- Sys.time()
  manifest <- fromJSON(opts$input_manifest, simplifyVector = FALSE)
  bundle <- select_bundle(manifest, opts$phase)
  lr_path <- manifest$cci_resource_common_tsv
  if (is.null(lr_path) || !file.exists(lr_path)) {
    stop("Manifest cci_resource_common_tsv is missing or not readable")
  }

  genes <- read.table(bundle$genes_tsv, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")
  gene_symbols <- if ("gene_symbol" %in% colnames(genes)) as.character(genes$gene_symbol) else as.character(genes[[1]])
  barcodes <- as.character(read_first_column(bundle$barcodes_tsv))
  meta <- read.table(bundle$meta_tsv, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")
  if (!("cell_id" %in% colnames(meta)) || !("cell_type" %in% colnames(meta))) {
    stop("meta.tsv must contain cell_id and cell_type")
  }
  match_idx <- match(barcodes, as.character(meta$cell_id))
  if (any(is.na(match_idx))) {
    stop("meta.tsv does not cover all sparse bundle barcodes")
  }
  meta <- meta[match_idx, , drop = FALSE]
  cell_types <- as.character(meta$cell_type)
  uniq_cell_types <- sort(unique(cell_types))

  lr <- read.table(lr_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")
  if (!all(c("ligand", "receptor") %in% colnames(lr))) {
    stop("CCI resource must contain ligand and receptor columns")
  }
  lr <- unique(lr[, c("ligand", "receptor"), drop = FALSE])
  lr$ligand <- as.character(lr$ligand)
  lr$receptor <- as.character(lr$receptor)
  detected <- lr$ligand %in% gene_symbols & lr$receptor %in% gene_symbols
  lr <- lr[detected, , drop = FALSE]
  if (!is.null(opts$max_cci_pairs) && !is.na(opts$max_cci_pairs)) {
    lr <- head(lr, opts$max_cci_pairs)
  }
  if (nrow(lr) == 0) {
    stop("No detectable CCI pairs remained after filtering")
  }

  needed_genes <- unique(c(lr$ligand, lr$receptor))
  gene_idx <- match(needed_genes, gene_symbols)
  m <- Matrix::readMM(bundle$counts_symbol_mtx)
  if (nrow(m) == length(gene_symbols) && ncol(m) == length(barcodes)) {
    x <- Matrix::t(m)
  } else if (nrow(m) == length(barcodes) && ncol(m) == length(gene_symbols)) {
    x <- m
  } else {
    stop(sprintf("Sparse matrix shape %sx%s does not match genes=%s barcodes=%s", nrow(m), ncol(m), length(gene_symbols), length(barcodes)))
  }
  xs <- x[, gene_idx, drop = FALSE]
  rm(m, x)
  colnames(xs) <- needed_genes

  means <- matrix(0, nrow = length(uniq_cell_types), ncol = length(needed_genes), dimnames = list(uniq_cell_types, needed_genes))
  for (ct in uniq_cell_types) {
    cells <- which(cell_types == ct)
    if (length(cells) > 0) {
      means[ct, ] <- Matrix::colMeans(xs[cells, , drop = FALSE])
    }
  }

  pairs_path <- manifest$celltype_pairs_tsv
  if (!is.null(pairs_path) && file.exists(pairs_path)) {
    celltype_pairs <- read.table(pairs_path, sep = "\t", header = TRUE, stringsAsFactors = FALSE, quote = "", comment.char = "")
    celltype_pairs <- celltype_pairs[celltype_pairs$sender %in% uniq_cell_types & celltype_pairs$receiver %in% uniq_cell_types, , drop = FALSE]
  } else {
    celltype_pairs <- expand.grid(sender = uniq_cell_types, receiver = uniq_cell_types, stringsAsFactors = FALSE)
  }
  if (nrow(celltype_pairs) == 0) {
    stop("No sender/receiver cell type pairs were available")
  }

  chunks <- vector("list", nrow(lr))
  for (i in seq_len(nrow(lr))) {
    lig <- lr$ligand[[i]]
    rec <- lr$receptor[[i]]
    ligand_mean <- means[celltype_pairs$sender, lig]
    receptor_mean <- means[celltype_pairs$receiver, rec]
    chunks[[i]] <- data.frame(
      method = opts$method,
      database_mode = opts$database_mode,
      ligand = lig,
      receptor = rec,
      sender = celltype_pairs$sender,
      receiver = celltype_pairs$receiver,
      score_raw = as.numeric(ligand_mean * receptor_mean),
      fdr_or_pvalue = NA_real_,
      resolution = "celltype_pair",
      spatial_support_type = paste0(opts$method, "_package_present_expression_baseline"),
      artifact_path = file.path(opts$output_dir, "raw", paste0(opts$method, "_expression_cci_scores.tsv")),
      stringsAsFactors = FALSE
    )
  }
  df <- do.call(rbind, chunks)
  df <- rank_standardize(df)
  df <- df[, c(
    "method", "database_mode", "ligand", "receptor", "sender", "receiver",
    "score_raw", "score_std", "rank_within_method", "fdr_or_pvalue",
    "resolution", "spatial_support_type", "artifact_path"
  )]

  raw_path <- file.path(opts$output_dir, "raw", paste0(opts$method, "_expression_cci_scores.tsv"))
  std_path <- file.path(opts$output_dir, paste0(opts$method, "_standardized.tsv"))
  write.table(df, raw_path, sep = "\t", quote = FALSE, row.names = FALSE)
  write.table(df, std_path, sep = "\t", quote = FALSE, row.names = FALSE)

  payload <- list(
    method = opts$method,
    status = "success",
    phase = opts$phase,
    database_mode = opts$database_mode,
    bounded_mode = "package_present_expression_baseline",
    package_available = available,
    n_interaction_pairs = nrow(lr),
    n_rows = nrow(df),
    standardized_tsv = std_path,
    raw_artifact = raw_path,
    input_manifest = opts$input_manifest,
    output_dir = opts$output_dir,
    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    note = "A method package was importable, but the method-specific public API mapping is not finalized; this bounded result uses the shared CCI expression baseline and is kept for appendix comparison."
  )
  write_json(payload, file.path(opts$output_dir, "run_summary.json"), auto_unbox = TRUE, pretty = TRUE)
  invisible(payload)
}

candidate_packages <- function(method) {
  switch(
    method,
    giotto = c("Giotto", "GiottoClass", "GiottoUtils"),
    spatalk = c("SpaTalk"),
    niches = c("NICHES"),
    c(method)
  )
}

write_method_card <- function(output_dir, payload) {
  path <- file.path(output_dir, "method_card.md")
  lines <- c(
    paste0("# Method Card: ", payload$method),
    "",
    paste0("- Status: `", payload$status, "`"),
    paste0("- Phase: `", payload$phase, "`"),
    paste0("- Database mode: `", payload$database_mode, "`"),
    paste0("- Reason: `", payload$reason, "`"),
    paste0("- Reproduce: `", payload$reproduce, "`"),
    ""
  )
  writeLines(lines, path)
  path
}

main <- function() {
  opts <- parse_args()
  dir.create(opts$output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(opts$output_dir, "raw"), recursive = TRUE, showWarnings = FALSE)
  activate_method_renv(opts$method, opts$input_manifest, opts$output_dir)
  packages <- candidate_packages(opts$method)
  available <- packages[vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(available) > 0 && isTRUE(opts$allow_expression_baseline)) {
    payload <- run_package_present_expression_baseline(opts, available)
    return(invisible(payload))
  }
  status_reason <- if (length(available) == 0) {
    "No supported R package was importable in the method-specific environment."
  } else {
    "Package is installed, but no validated method-specific adapter is available; shared expression baselines are disabled by default for true-method benchmarking."
  }
  payload <- list(
    method = opts$method,
    status = "failed",
    phase = opts$phase,
    database_mode = opts$database_mode,
    reason = status_reason,
    package_candidates = packages,
    package_available = available,
    allow_expression_baseline = isTRUE(opts$allow_expression_baseline),
    input_manifest = opts$input_manifest,
    output_dir = opts$output_dir,
    reproduce = paste("conda run --name <method-env> Rscript", "run_external_cci_method.R", "--method", opts$method)
  )
  write_json(payload, file.path(opts$output_dir, "run_summary.json"), auto_unbox = TRUE, pretty = TRUE)
  write_method_card(opts$output_dir, payload)
  invisible(payload)
}

tryCatch(
  main(),
  error = function(e) {
    output_dir <- "."
    args <- commandArgs(trailingOnly = TRUE)
    idx <- match("--output-dir", args)
    if (!is.na(idx) && idx < length(args)) output_dir <- args[[idx + 1]]
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    payload <- list(method = "external_r", status = "failed", reason = conditionMessage(e))
    write_json(payload, file.path(output_dir, "run_summary.json"), auto_unbox = TRUE, pretty = TRUE)
    stop(e)
  }
)
