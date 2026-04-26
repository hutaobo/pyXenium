args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "run_r_lr_utils.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
})

opts <- parse_lr_args()
method <- "niches"
output_dir <- opts[["output-dir"]] %||% stop("--output-dir is required", call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "raw"), recursive = TRUE, showWarnings = FALSE)

for (pkg in c("NICHES", "Seurat", "SeuratObject")) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    fail_run(output_dir, method, sprintf("R package %s is required.", pkg))
  }
}

input_manifest <- opts[["input-manifest"]] %||% fail_run(output_dir, method, "--input-manifest is required.")
database_mode <- opts[["database-mode"]] %||% "common"
phase <- opts[["phase"]] %||% "smoke"
max_lr_pairs <- opts[["max-lr-pairs"]]
max_cells <- opts[["max-cells"]]
seed <- as.integer(opts[["seed"]] %||% 1)
k <- as.integer(opts[["k"]] %||% 4)
chunk_id <- opts[["chunk-id"]]
chunk_size <- opts[["chunk-size"]]

started <- Sys.time()
params <- list(
  method = method,
  database_mode = database_mode,
  phase = phase,
  input_manifest = input_manifest,
  output_dir = output_dir,
  max_lr_pairs = max_lr_pairs,
  max_cells = max_cells,
  seed = seed,
  k = k,
  chunk_id = chunk_id,
  chunk_size = chunk_size,
  runner = "run_niches.R"
)
write_params(output_dir, params)

split_lr_name <- function(x) {
  dash <- intToUtf8(8212)
  parts <- strsplit(as.character(x), dash, fixed = TRUE)[[1]]
  if (length(parts) < 2) parts <- strsplit(as.character(x), "-", fixed = TRUE)[[1]]
  if (length(parts) < 2) return(c(as.character(x), NA_character_))
  c(parts[[1]], paste(parts[-1], collapse = "-"))
}

aggregate_niches_matrix <- function(mat, metadata, artifact_path, database_mode) {
  if (nrow(mat) == 0 || ncol(mat) == 0) {
    return(data.frame())
  }
  metadata$edge_col <- seq_len(nrow(metadata))
  vector_type <- as.character(metadata$VectorType)
  type_levels <- sort(unique(vector_type))
  type_index <- match(vector_type, type_levels)
  denom <- tabulate(type_index, nbins = length(type_levels))
  type_parts <- strsplit(type_levels, intToUtf8(8212), fixed = TRUE)
  senders <- vapply(type_parts, `[`, character(1), 1)
  receivers <- vapply(type_parts, function(x) if (length(x) >= 2) paste(x[-1], collapse = intToUtf8(8212)) else NA_character_, character(1))

  sm <- Matrix::summary(mat)
  if (nrow(sm) > 0) {
    agg <- aggregate(sm$x, by = list(interaction = sm$i, type = type_index[sm$j]), FUN = sum)
    colnames(agg)[3] <- "sum_score"
  } else {
    agg <- data.frame(interaction = integer(0), type = integer(0), sum_score = numeric(0))
  }
  grid <- expand.grid(interaction = seq_len(nrow(mat)), type = seq_along(type_levels))
  merged <- merge(grid, agg, by = c("interaction", "type"), all.x = TRUE, sort = FALSE)
  merged$sum_score[is.na(merged$sum_score)] <- 0
  merged$score_raw <- merged$sum_score / denom[merged$type]

  lr_parts <- t(vapply(rownames(mat)[merged$interaction], split_lr_name, character(2)))
  data.frame(
    method = method,
    database_mode = database_mode,
    ligand = lr_parts[, 1],
    receptor = lr_parts[, 2],
    sender = senders[merged$type],
    receiver = receivers[merged$type],
    score_raw = merged$score_raw,
    fdr_or_pvalue = NA_real_,
    resolution = "cell_pair_aggregated_to_celltype_pair",
    spatial_support_type = "niches_RunNICHES_CellToCellSpatial",
    artifact_path = artifact_path,
    stringsAsFactors = FALSE
  )
}

tryCatch({
  manifest <- jsonlite::fromJSON(input_manifest, simplifyVector = FALSE)
  bundle <- read_sparse_bundle(manifest, phase, max_cells = max_cells, seed = seed)
  counts <- bundle$counts
  lr <- read_lr_database(
    manifest,
    database_mode,
    max_lr_pairs = max_lr_pairs,
    chunk_id = chunk_id,
    chunk_size = chunk_size
  )
  lr <- lr[lr$ligand %in% rownames(counts) & lr$receptor %in% rownames(counts), , drop = FALSE]
  if (nrow(lr) == 0) stop("No common LR pairs were detectable in the sparse bundle.")
  custom_lr <- unique(data.frame(ligand = lr$ligand, receptor = lr$receptor, stringsAsFactors = FALSE))

  suppressPackageStartupMessages({
    library(Seurat)
    library(NICHES)
  })
  seu <- Seurat::CreateSeuratObject(counts = counts, assay = "RNA", meta.data = data.frame(row.names = bundle$barcodes))
  seu$cell_type <- as.character(bundle$meta$cell_type)
  seu$x <- as.numeric(bundle$coords$x)
  seu$y <- as.numeric(bundle$coords$y)
  Seurat::Idents(seu) <- seu$cell_type
  seu <- Seurat::NormalizeData(seu, assay = "RNA", verbose = FALSE)
  niches_out <- NICHES::RunNICHES(
    object = seu,
    assay = "RNA",
    LR.database = "custom",
    species = "human",
    custom_LR_database = custom_lr,
    meta.data.to.map = c("cell_type"),
    position.x = "x",
    position.y = "y",
    k = k,
    CellToCell = FALSE,
    CellToSystem = FALSE,
    SystemToCell = FALSE,
    CellToCellSpatial = TRUE,
    CellToNeighborhood = FALSE,
    NeighborhoodToCell = FALSE,
    output_format = "raw"
  )
  c2c <- if (!is.null(niches_out$CellToCellSpatial)) niches_out$CellToCellSpatial else niches_out[[1]]
  mat <- c2c$CellToCellSpatialMatrix
  metadata <- c2c$metadata
  if (is.null(mat) || is.null(metadata)) stop("NICHES output did not contain CellToCellSpatialMatrix and metadata.")
  raw_rds <- file.path(output_dir, "raw", "niches_cell_to_cell_spatial_raw.rds")
  saveRDS(c2c, raw_rds)
  raw_meta <- write_raw_tsv(metadata, output_dir, method, "niches_cell_to_cell_spatial_metadata.tsv")
  std <- aggregate_niches_matrix(mat, metadata, dirname(raw_meta), database_mode)
  if (nrow(std) == 0) stop("NICHES aggregation returned no rows.")
  std_path <- write_standardized(std, output_dir, method)
  summary <- list(
    method = method,
    status = "success",
    database_mode = database_mode,
    phase = phase,
    raw_rds = raw_rds,
    metadata_tsv = raw_meta,
    standardized_tsv_gz = std_path,
    n_lr_pairs = nrow(lr),
    n_rows = nrow(std),
    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    package_versions = list(
      R = R.version.string,
      NICHES = as.character(utils::packageVersion("NICHES")),
      Seurat = as.character(utils::packageVersion("Seurat")),
      Matrix = as.character(utils::packageVersion("Matrix")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    )
  )
  write_json(file.path(output_dir, "run_summary.json"), summary)
  cat(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE))
}, error = function(e) {
  fail_run(output_dir, method, conditionMessage(e), extra = list(params = params))
})
