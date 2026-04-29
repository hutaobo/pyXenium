`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0) {
    return(y)
  }
  if (is.atomic(x) && length(x) == 1 && is.na(x)) {
    return(y)
  }
  x
}

parse_cci_args <- function(args = commandArgs(trailingOnly = TRUE)) {
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

write_json <- function(path, payload) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  jsonlite::write_json(payload, path, pretty = TRUE, auto_unbox = TRUE, null = "null")
}

write_method_card <- function(output_dir, method, payload) {
  lines <- c(
    sprintf("# Method Card: %s", method),
    "",
    sprintf("- Status: `%s`", payload$status %||% "unknown"),
    sprintf("- Reason: `%s`", payload$reason %||% payload$error %||% "not recorded"),
    sprintf("- Reproduce: `%s`", payload$reproduce %||% "See run_summary.json and logs."),
    ""
  )
  writeLines(lines, file.path(output_dir, "method_card.md"))
}

fail_run <- function(output_dir, method, message, code = 2, extra = list()) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  payload <- c(list(method = method, status = "failed", reason = message), extra)
  write_json(file.path(output_dir, "run_summary.json"), payload)
  write_method_card(output_dir, method, payload)
  quit(status = code, save = "no")
}

select_bundle <- function(manifest, phase) {
  if (identical(phase, "full")) {
    if (is.null(manifest$full_bundle) || length(manifest$full_bundle) == 0) {
      stop("Phase 'full' was requested, but the input manifest has no full_bundle.", call. = FALSE)
    }
    return(manifest$full_bundle)
  }
  if (!is.null(manifest$smoke_bundle) && length(manifest$smoke_bundle) > 0) {
    return(manifest$smoke_bundle)
  }
  if (!is.null(manifest$full_bundle) && length(manifest$full_bundle) > 0) {
    return(manifest$full_bundle)
  }
  stop("Input manifest does not contain a usable sparse bundle.", call. = FALSE)
}

select_cell_subset <- function(barcodes, meta, max_cells = NULL, seed = 1) {
  if (is.null(max_cells) || identical(max_cells, TRUE)) {
    return(seq_along(barcodes))
  }
  max_cells <- as.integer(max_cells)
  if (is.na(max_cells) || max_cells <= 0 || length(barcodes) <= max_cells) {
    return(seq_along(barcodes))
  }
  max_cells_n <- as.numeric(max_cells)
  total_n <- as.numeric(length(barcodes))
  set.seed(as.integer(seed %||% 1))
  groups <- split(seq_along(barcodes), as.character(meta$cell_type))
  picked <- unlist(lapply(groups, function(idx) {
    target <- max(1, round(max_cells_n * as.numeric(length(idx)) / total_n))
    sample(idx, min(length(idx), target))
  }), use.names = FALSE)
  picked <- unique(picked)
  if (length(picked) > max_cells) {
    picked <- sample(picked, max_cells)
  } else if (length(picked) < max_cells) {
    remaining <- setdiff(seq_along(barcodes), picked)
    picked <- c(picked, sample(remaining, min(length(remaining), max_cells - length(picked))))
  }
  sort(picked)
}

read_sparse_bundle <- function(manifest, phase, max_cells = NULL, seed = 1) {
  bundle <- select_bundle(manifest, phase)
  required <- c("counts_symbol_mtx", "barcodes_tsv", "genes_tsv", "meta_tsv", "coords_tsv")
  missing <- required[vapply(bundle[required], is.null, logical(1))]
  if (length(missing) > 0) {
    stop(sprintf("Sparse bundle is missing fields: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
  counts <- Matrix::readMM(bundle$counts_symbol_mtx)
  barcodes <- readLines(bundle$barcodes_tsv)
  genes <- read.delim(bundle$genes_tsv, sep = "\t", stringsAsFactors = FALSE)
  meta <- read.delim(bundle$meta_tsv, sep = "\t", stringsAsFactors = FALSE)
  coords <- read.delim(bundle$coords_tsv, sep = "\t", stringsAsFactors = FALSE)
  gene_symbols <- if ("gene_symbol" %in% colnames(genes)) genes$gene_symbol else genes[[1]]
  gene_symbols <- make.unique(as.character(gene_symbols))
  barcodes <- as.character(barcodes)
  if (nrow(counts) == length(gene_symbols) && ncol(counts) == length(barcodes)) {
    counts_genes_by_cells <- as(counts, "dgCMatrix")
  } else if (nrow(counts) == length(barcodes) && ncol(counts) == length(gene_symbols)) {
    counts_genes_by_cells <- as(Matrix::t(counts), "dgCMatrix")
  } else {
    stop(sprintf(
      "Sparse matrix shape %sx%s does not match genes=%s barcodes=%s",
      nrow(counts), ncol(counts), length(gene_symbols), length(barcodes)
    ), call. = FALSE)
  }
  rownames(counts_genes_by_cells) <- gene_symbols
  colnames(counts_genes_by_cells) <- barcodes
  if (!all(c("cell_id", "cell_type") %in% colnames(meta))) {
    stop("meta.tsv must contain cell_id and cell_type.", call. = FALSE)
  }
  meta <- meta[match(barcodes, as.character(meta$cell_id)), , drop = FALSE]
  coords <- coords[match(barcodes, as.character(coords$cell_id)), , drop = FALSE]
  if (any(is.na(meta$cell_id)) || any(is.na(coords$cell_id))) {
    stop("meta.tsv/coords.tsv do not cover all sparse bundle barcodes.", call. = FALSE)
  }
  keep <- select_cell_subset(barcodes, meta, max_cells = max_cells, seed = seed)
  if (length(keep) < length(barcodes)) {
    counts_genes_by_cells <- counts_genes_by_cells[, keep, drop = FALSE]
    barcodes <- barcodes[keep]
    meta <- meta[keep, , drop = FALSE]
    coords <- coords[keep, , drop = FALSE]
  }
  list(
    counts = counts_genes_by_cells,
    barcodes = barcodes,
    gene_symbols = gene_symbols,
    meta = meta,
    coords = coords,
    bundle = bundle
  )
}

read_cci_resource <- function(manifest, mode, max_cci_pairs = NULL, chunk_id = NULL, chunk_size = NULL) {
  db_path <- if (mode %in% c("smoke-panel", "smoke_panel")) manifest$atera_smoke_panel_tsv else manifest$cci_resource_common_tsv
  if (is.null(db_path) || !file.exists(db_path)) {
    stop("Manifest CCI resource path is missing or unreadable.", call. = FALSE)
  }
  lr <- read.delim(db_path, sep = "\t", stringsAsFactors = FALSE)
  if (!all(c("ligand", "receptor") %in% colnames(lr))) {
    stop("CCI resource must contain ligand and receptor columns.", call. = FALSE)
  }
  lr$ligand <- as.character(lr$ligand)
  lr$receptor <- as.character(lr$receptor)
  if (!"pathway" %in% colnames(lr)) lr$pathway <- "custom"
  lr <- unique(lr[, intersect(c("ligand", "receptor", "pathway"), colnames(lr)), drop = FALSE])
  if (!is.null(chunk_id) && !is.null(chunk_size)) {
    chunk_id <- as.integer(chunk_id)
    chunk_size <- as.integer(chunk_size)
    start <- (chunk_id * chunk_size) + 1
    end <- min(nrow(lr), start + chunk_size - 1)
    if (start > nrow(lr)) lr <- lr[0, , drop = FALSE] else lr <- lr[start:end, , drop = FALSE]
  }
  if (!is.null(max_cci_pairs)) {
    lr <- head(lr, as.integer(max_cci_pairs))
  }
  lr
}

rank_standardize <- function(df) {
  if (nrow(df) == 0) {
    df$score_std <- numeric(0)
    df$rank_within_method <- numeric(0)
    df$rank_fraction <- numeric(0)
    return(df)
  }
  score <- suppressWarnings(as.numeric(df$score_raw))
  score[!is.finite(score)] <- 0
  pvalue <- suppressWarnings(as.numeric(df$fdr_or_pvalue))
  ord <- order(-score, pvalue, na.last = TRUE)
  rank <- rep(NA_real_, length(score))
  rank[ord] <- seq_along(score)
  rank_fraction <- if (length(score) == 1) 1 else 1 - ((rank - 1) / (length(score) - 1))
  df$score_raw <- score
  df$rank_within_method <- rank
  df$rank_fraction <- rank_fraction
  df$score_std <- rank_fraction
  df[order(df$rank_within_method), , drop = FALSE]
}

standardized_columns <- function() {
  c(
    "method", "database_mode", "ligand", "receptor", "sender", "receiver",
    "score_raw", "score_std", "rank_within_method", "rank_fraction",
    "fdr_or_pvalue", "resolution", "spatial_support_type", "artifact_path"
  )
}

write_standardized <- function(df, output_dir, method) {
  df <- rank_standardize(df)
  cols <- standardized_columns()
  for (col in setdiff(cols, colnames(df))) df[[col]] <- NA
  df <- df[, cols, drop = FALSE]
  path <- file.path(output_dir, paste0(method, "_standardized.tsv.gz"))
  con <- gzfile(path, open = "wt")
  on.exit(close(con), add = TRUE)
  write.table(df, con, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

write_raw_tsv <- function(df, output_dir, method, name = "raw_output.tsv") {
  raw_dir <- file.path(output_dir, "raw")
  dir.create(raw_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(raw_dir, name)
  write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
  path
}

write_params <- function(output_dir, params) {
  write_json(file.path(output_dir, "params.json"), params)
}
