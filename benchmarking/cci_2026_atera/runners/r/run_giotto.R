args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "run_r_cci_utils.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
})

opts <- parse_cci_args()
method <- "giotto"
output_dir <- opts[["output-dir"]] %||% stop("--output-dir is required", call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "raw"), recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("Giotto", quietly = TRUE)) {
  fail_run(output_dir, method, "R package Giotto is required.")
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  fail_run(output_dir, method, "R package data.table is required.")
}

input_manifest <- opts[["input-manifest"]] %||% fail_run(output_dir, method, "--input-manifest is required.")
database_mode <- opts[["database-mode"]] %||% "common"
phase <- opts[["phase"]] %||% "smoke"
n_cores <- as.integer(opts[["n-cores"]] %||% 8)
k <- as.integer(opts[["k"]] %||% 4)
random_iter <- as.integer(opts[["random-iter"]] %||% 100)
max_cci_pairs <- opts[["max-cci-pairs"]]
max_cells <- opts[["max-cells"]]
seed <- as.integer(opts[["seed"]] %||% 1)
do_parallel <- isTRUE(opts[["do-parallel"]])

started <- Sys.time()
params <- list(
  method = method,
  database_mode = database_mode,
  phase = phase,
  input_manifest = input_manifest,
  output_dir = output_dir,
  n_cores = n_cores,
  k = k,
  random_iter = random_iter,
  max_cci_pairs = max_cci_pairs,
  max_cells = max_cells,
  seed = seed,
  do_parallel = do_parallel,
  runner = "run_giotto.R"
)
write_params(output_dir, params)

tryCatch({
  manifest <- jsonlite::fromJSON(input_manifest, simplifyVector = FALSE)
  bundle <- read_sparse_bundle(manifest, phase, max_cells = max_cells, seed = seed)
  counts <- bundle$counts
  lr <- read_cci_resource(manifest, database_mode, max_cci_pairs = max_cci_pairs)
  lr <- lr[lr$ligand %in% rownames(counts) & lr$receptor %in% rownames(counts), , drop = FALSE]
  if (nrow(lr) == 0) stop("No common CCI pairs were detectable in the sparse bundle.")

  cell_metadata <- data.frame(
    cell_ID = bundle$barcodes,
    cell_types = as.character(bundle$meta$cell_type),
    stringsAsFactors = FALSE
  )
  rownames(cell_metadata) <- cell_metadata$cell_ID
  spatial_locs <- data.frame(
    cell_ID = bundle$barcodes,
    sdimx = as.numeric(bundle$coords$x),
    sdimy = as.numeric(bundle$coords$y),
    stringsAsFactors = FALSE
  )
  rownames(spatial_locs) <- spatial_locs$cell_ID

  suppressPackageStartupMessages(library(Giotto))
  gobject <- Giotto::createGiottoObject(
    raw_exprs = counts,
    spatial_locs = spatial_locs,
    cell_metadata = cell_metadata,
    cores = n_cores
  )
  gobject <- Giotto::normalizeGiotto(
    gobject = gobject,
    scalefactor = 6000,
    verbose = FALSE
  )
  gobject <- Giotto::createSpatialKNNnetwork(
    gobject = gobject,
    name = "knn_network",
    k = k,
    verbose = FALSE
  )
  raw <- Giotto::spatCellCellcom(
    gobject = gobject,
    spatial_network_name = "knn_network",
    cluster_column = "cell_types",
    random_iter = random_iter,
    gene_set_1 = lr$ligand,
    gene_set_2 = lr$receptor,
    min_observations = 2,
    detailed = FALSE,
    adjust_method = "fdr",
    adjust_target = "genes",
    do_parallel = do_parallel,
    cores = n_cores,
    set_seed = TRUE,
    seed_number = 1234,
    verbose = "none"
  )
  raw <- as.data.frame(raw)
  if (nrow(raw) == 0) stop("Giotto spatCellCellcom returned no rows.")
  raw_path <- write_raw_tsv(raw, output_dir, method, "giotto_spatCellCellcom.tsv")

  if (!all(c("ligand", "receptor", "lig_cell_type", "rec_cell_type") %in% colnames(raw))) {
    if ("LR_comb" %in% colnames(raw)) {
      parts <- strsplit(as.character(raw$LR_comb), "-", fixed = TRUE)
      raw$ligand <- vapply(parts, `[`, character(1), 1)
      raw$receptor <- vapply(parts, function(x) paste(x[-1], collapse = "-"), character(1))
    }
  }
  score_col <- if ("PI" %in% colnames(raw)) "PI" else if ("log2fc" %in% colnames(raw)) "log2fc" else "LR_expr"
  p_col <- if ("p.adj" %in% colnames(raw)) "p.adj" else if ("pvalue" %in% colnames(raw)) "pvalue" else NA
  std <- data.frame(
    method = method,
    database_mode = database_mode,
    ligand = as.character(raw$ligand),
    receptor = as.character(raw$receptor),
    sender = as.character(raw$lig_cell_type),
    receiver = as.character(raw$rec_cell_type),
    score_raw = suppressWarnings(as.numeric(raw[[score_col]])),
    fdr_or_pvalue = if (is.na(p_col)) NA_real_ else suppressWarnings(as.numeric(raw[[p_col]])),
    resolution = "celltype_pair",
    spatial_support_type = "giotto_spatCellCellcom_knn",
    artifact_path = dirname(raw_path),
    stringsAsFactors = FALSE
  )
  std_path <- write_standardized(std, output_dir, method)
  summary <- list(
    method = method,
    status = "success",
    database_mode = database_mode,
    phase = phase,
    raw_tsv = raw_path,
    standardized_tsv_gz = std_path,
    n_interaction_pairs = nrow(lr),
    n_rows = nrow(std),
    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    package_versions = list(
      R = R.version.string,
      Giotto = as.character(utils::packageVersion("Giotto")),
      Matrix = as.character(utils::packageVersion("Matrix")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    )
  )
  write_json(file.path(output_dir, "run_summary.json"), summary)
  cat(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE))
}, error = function(e) {
  fail_run(output_dir, method, conditionMessage(e), extra = list(params = params))
})
