args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "run_r_lr_utils.R"))

suppressPackageStartupMessages({
  library(jsonlite)
  library(Matrix)
})

get_spatalk_data <- function(name) {
  envs <- list(as.environment("package:SpaTalk"), asNamespace("SpaTalk"))
  for (env in envs) {
    if (exists(name, envir = env, inherits = FALSE)) {
      value <- get(name, envir = env, inherits = FALSE)
      if (is.function(value)) value <- value()
      return(as.data.frame(value, stringsAsFactors = FALSE))
    }
  }
  stop(sprintf("SpaTalk object '%s' was not found.", name), call. = FALSE)
}

opts <- parse_lr_args()
method <- "spatalk"
output_dir <- opts[["output-dir"]] %||% stop("--output-dir is required", call. = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(output_dir, "raw"), recursive = TRUE, showWarnings = FALSE)

if (!requireNamespace("SpaTalk", quietly = TRUE)) {
  fail_run(output_dir, method, "R package SpaTalk is required.")
}

input_manifest <- opts[["input-manifest"]] %||% fail_run(output_dir, method, "--input-manifest is required.")
database_mode <- opts[["database-mode"]] %||% "common"
phase <- opts[["phase"]] %||% "smoke"
n_cores <- as.integer(opts[["n-cores"]] %||% 8)
max_lr_pairs <- opts[["max-lr-pairs"]]
max_cells <- opts[["max-cells"]]
seed <- as.integer(opts[["seed"]] %||% 1)
n_neighbor <- as.integer(opts[["n-neighbor"]] %||% 10)
min_pairs <- as.integer(opts[["min-pairs"]] %||% 5)
per_num <- as.integer(opts[["per-num"]] %||% 100)
pvalue <- as.numeric(opts[["pvalue"]] %||% 0.05)
co_exp_ratio <- as.numeric(opts[["co-exp-ratio"]] %||% 0.1)

started <- Sys.time()
params <- list(
  method = method,
  database_mode = database_mode,
  phase = phase,
  input_manifest = input_manifest,
  output_dir = output_dir,
  n_cores = n_cores,
  max_lr_pairs = max_lr_pairs,
  max_cells = max_cells,
  seed = seed,
  n_neighbor = n_neighbor,
  min_pairs = min_pairs,
  per_num = per_num,
  pvalue = pvalue,
  co_exp_ratio = co_exp_ratio,
  runner = "run_spatalk.R"
)
write_params(output_dir, params)

tryCatch({
  manifest <- jsonlite::fromJSON(input_manifest, simplifyVector = FALSE)
  bundle <- read_sparse_bundle(manifest, phase, max_cells = max_cells, seed = seed)
  counts <- bundle$counts
  lr <- read_lr_database(manifest, database_mode, max_lr_pairs = max_lr_pairs)
  lr <- lr[lr$ligand %in% rownames(counts) & lr$receptor %in% rownames(counts), , drop = FALSE]
  if (nrow(lr) == 0) stop("No common LR pairs were detectable in the sparse bundle.")
  lrpairs <- data.frame(ligand = lr$ligand, receptor = lr$receptor, species = "Human", stringsAsFactors = FALSE)

  suppressPackageStartupMessages(library(SpaTalk))
  pathways <- get_spatalk_data("pathways")
  st_meta <- data.frame(
    cell = bundle$barcodes,
    x = as.numeric(bundle$coords$x),
    y = as.numeric(bundle$coords$y),
    stringsAsFactors = FALSE
  )
  celltype <- as.character(bundle$meta$cell_type)
  object <- SpaTalk::createSpaTalk(
    st_data = counts,
    st_meta = st_meta,
    species = "Human",
    if_st_is_sc = TRUE,
    spot_max_cell = 1,
    celltype = celltype
  )
  object <- SpaTalk::find_lr_path(
    object = object,
    lrpairs = lrpairs,
    pathways = pathways,
    if_doParallel = n_cores > 1,
    use_n_cores = n_cores
  )
  object <- SpaTalk::dec_cci_all(
    object = object,
    n_neighbor = n_neighbor,
    min_pairs = min_pairs,
    min_pairs_ratio = 0,
    per_num = per_num,
    pvalue = pvalue,
    co_exp_ratio = co_exp_ratio,
    if_doParallel = n_cores > 1,
    use_n_cores = n_cores
  )
  raw <- as.data.frame(object@lrpair)
  if (nrow(raw) == 0) stop("SpaTalk dec_cci_all returned no LR rows.")
  raw_path <- write_raw_tsv(raw, output_dir, method, "spatalk_dec_cci_all.tsv")
  saveRDS(object, file.path(output_dir, "raw", "spatalk_object.rds"))

  sender_col <- if ("celltype_sender" %in% colnames(raw)) "celltype_sender" else "sender"
  receiver_col <- if ("celltype_receiver" %in% colnames(raw)) "celltype_receiver" else "receiver"
  score_col <- if ("score" %in% colnames(raw)) "score" else if ("lr_co_ratio" %in% colnames(raw)) "lr_co_ratio" else NA
  if (is.na(score_col)) stop("SpaTalk output did not contain score or lr_co_ratio.")
  p_col <- if ("lr_co_ratio_pvalue" %in% colnames(raw)) "lr_co_ratio_pvalue" else NA
  std <- data.frame(
    method = method,
    database_mode = database_mode,
    ligand = as.character(raw$ligand),
    receptor = as.character(raw$receptor),
    sender = as.character(raw[[sender_col]]),
    receiver = as.character(raw[[receiver_col]]),
    score_raw = suppressWarnings(as.numeric(raw[[score_col]])),
    fdr_or_pvalue = if (is.na(p_col)) NA_real_ else suppressWarnings(as.numeric(raw[[p_col]])),
    resolution = "celltype_pair",
    spatial_support_type = "spatalk_dec_cci_all",
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
    n_lr_pairs = nrow(lr),
    n_rows = nrow(std),
    elapsed_seconds = as.numeric(difftime(Sys.time(), started, units = "secs")),
    package_versions = list(
      R = R.version.string,
      SpaTalk = as.character(utils::packageVersion("SpaTalk")),
      Matrix = as.character(utils::packageVersion("Matrix")),
      jsonlite = as.character(utils::packageVersion("jsonlite"))
    )
  )
  write_json(file.path(output_dir, "run_summary.json"), summary)
  cat(jsonlite::toJSON(summary, pretty = TRUE, auto_unbox = TRUE))
}, error = function(e) {
  fail_run(output_dir, method, conditionMessage(e), extra = list(params = params))
})
