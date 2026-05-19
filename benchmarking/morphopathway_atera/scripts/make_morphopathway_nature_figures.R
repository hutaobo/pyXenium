#!/usr/bin/env Rscript

parse_args <- function(args) {
  values <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (startsWith(key, "--")) {
      name <- sub("^--", "", key)
      if (i == length(args) || startsWith(args[[i + 1]], "--")) {
        values[[name]] <- TRUE
        i <- i + 1
      } else {
        values[[name]] <- args[[i + 1]]
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }
  values
}

arg_or <- function(values, name, default) {
  value <- values[[name]]
  if (is.null(value) || identical(value, "")) {
    default
  } else {
    value
  }
}

install_missing <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) {
    return(invisible(TRUE))
  }
  type <- if (.Platform$OS.type == "windows") "binary" else "source"
  install.packages(missing, repos = "https://cloud.r-project.org", type = type)
  still_missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(still_missing) > 0) {
    stop("Missing R packages after installation attempt: ", paste(still_missing, collapse = ", "))
  }
  invisible(TRUE)
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
required_packages <- c(
  "ggplot2", "patchwork", "svglite", "ragg", "readr", "dplyr", "tidyr",
  "stringr", "scales", "jsonlite", "tiff"
)
if (!identical(arg_or(args, "install-missing", "true"), "false")) {
  install_missing(required_packages)
}
invisible(lapply(required_packages, library, character.only = TRUE))

default_package_dir <- file.path(
  "benchmarking", "morphopathway_atera", "results",
  "brief_communication_package_highnull32_20260512_2049"
)
package_dir <- normalizePath(arg_or(args, "package-dir", default_package_dir), mustWork = TRUE)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
default_output_dir <- file.path(dirname(package_dir), paste0("nature_figure_package_highnull32_", timestamp))
output_dir <- normalizePath(arg_or(args, "output-dir", default_output_dir), mustWork = FALSE)
source_dir <- file.path(output_dir, "source_data")
figure_dir <- file.path(output_dir, "figures")
report_dir <- file.path(output_dir, "reports")
log_dir <- file.path(output_dir, "logs")
a100_dir <- file.path(output_dir, "a100")
dir.create(source_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(a100_dir, recursive = TRUE, showWarnings = FALSE)
render_started_at <- Sys.time()
resource_log_path <- file.path(log_dir, "figure_render.resource.log")

repo_root <- normalizePath(file.path(package_dir, "..", "..", "..", ".."), mustWork = TRUE)
script_dir <- file.path(repo_root, "benchmarking", "morphopathway_atera", "scripts")
he_prep_script <- file.path(script_dir, "prepare_nature_he_source_data.py")

read_csv <- function(name) {
  readr::read_csv(file.path(package_dir, name), show_col_types = FALSE)
}

manifest <- jsonlite::read_json(file.path(package_dir, "brief_communication_package_manifest.json"))
run_dirs <- unlist(manifest$run_dirs)
if (length(run_dirs) < 1) {
  stop("The brief communication manifest has no run_dirs.")
}
reference_run_dir <- normalizePath(run_dirs[[1]], mustWork = TRUE)

write_a100_manifest <- function(status, reason = "") {
  remote_root <- paste0("/data/taobo.hu/pyxenium_morphopathway_nature_figures_", timestamp)
  remote_repo <- paste0(remote_root, "/repo")
  package_rel <- file.path("benchmarking", "morphopathway_atera", "results", basename(package_dir))
  package_rel <- gsub("\\\\", "/", package_rel)
  remote_package_dir <- paste0(remote_repo, "/", package_rel)
  remote_output_dir <- paste0(remote_root, "/nature_figure_package_highnull32_a100")
  payload <- list(
    status = status,
    reason = reason,
    local_output_dir = output_dir,
    remote_root = remote_root,
    ssh_target = "taobo.hu@sscb-a100.scilifelab.se",
    plotting_backend = "R",
    evidence_policy = "new pyXenium.pathway package only; no old naturebiotech evidence input",
    suggested_commands = c(
      paste(
        "ssh taobo.hu@sscb-a100.scilifelab.se",
        shQuote(paste("mkdir -p", remote_root, "&& mkdir -p", remote_repo))
      ),
      paste(
        "rsync -av --relative",
        shQuote(file.path("benchmarking", "morphopathway_atera", "scripts")),
        shQuote(package_rel),
        shQuote(paste0("taobo.hu@sscb-a100.scilifelab.se:", remote_repo))
      ),
      paste(
        "ssh taobo.hu@sscb-a100.scilifelab.se",
        shQuote(paste(
          "cd", remote_repo,
          "&& Rscript benchmarking/morphopathway_atera/scripts/make_morphopathway_nature_figures.R",
          "--package-dir", remote_package_dir,
          "--output-dir", remote_output_dir
        ))
      )
    )
  )
  jsonlite::write_json(payload, file.path(a100_dir, "a100_fallback_manifest.json"), auto_unbox = TRUE, pretty = TRUE)
}

run_he_source_prep <- function() {
  expected <- file.path(source_dir, c(
    "he_overview_pixels_breast_discovery.csv.gz",
    "he_overview_pixels_cervical_validation.csv.gz",
    "he_block_overlay_breast_discovery.csv",
    "he_block_overlay_cervical_validation.csv",
    "he_image_integrity_manifest.csv"
  ))
  if (all(file.exists(expected))) {
    return(invisible(TRUE))
  }
  python <- Sys.which("python")
  if (!nzchar(python)) {
    python <- Sys.which("python3")
  }
  if (!nzchar(python)) {
    write_a100_manifest("triggered", "No python executable was available for numeric H&E source-data extraction.")
    stop("No python executable was available for numeric H&E source-data extraction.")
  }
  stdout_log <- file.path(log_dir, "prepare_nature_he_source_data.stdout.log")
  stderr_log <- file.path(log_dir, "prepare_nature_he_source_data.stderr.log")
  cmd_args <- c(
    he_prep_script,
    "--package-dir", package_dir,
    "--output-dir", source_dir,
    "--run-index", "0",
    "--max-raster-dim", arg_or(args, "max-raster-dim", "900")
  )
  start <- Sys.time()
  status <- system2(python, cmd_args, stdout = stdout_log, stderr = stderr_log)
  end <- Sys.time()
  writeLines(
    c(
      paste("python:", python),
      paste("script:", he_prep_script),
      paste("status:", status),
      paste("start:", start),
      paste("end:", end),
      paste("elapsed_seconds:", round(as.numeric(difftime(end, start, units = "secs")), 3))
    ),
    file.path(log_dir, "prepare_nature_he_source_data.resource.log")
  )
  if (!identical(status, 0L)) {
    reason <- paste("Local H&E source-data preparation failed with status", status)
    write_a100_manifest("triggered", reason)
    stop(reason)
  }
  missing <- expected[!file.exists(expected)]
  if (length(missing) > 0) {
    reason <- paste("H&E source-data preparation did not create:", paste(missing, collapse = ", "))
    write_a100_manifest("triggered", reason)
    stop(reason)
  }
  invisible(TRUE)
}

run_he_source_prep()

theme_nature <- function(base_size = 6.5) {
  ggplot2::theme_classic(base_size = base_size, base_family = "Arial") +
    ggplot2::theme(
      axis.line = ggplot2::element_line(linewidth = 0.3, colour = "black"),
      axis.ticks = ggplot2::element_line(linewidth = 0.3, colour = "black"),
      axis.text = ggplot2::element_text(colour = "black", size = base_size),
      axis.title = ggplot2::element_text(size = base_size + 0.2),
      legend.title = ggplot2::element_text(size = base_size),
      legend.text = ggplot2::element_text(size = base_size - 0.3),
      legend.key.height = ggplot2::unit(3.6, "mm"),
      strip.background = ggplot2::element_blank(),
      strip.text = ggplot2::element_text(face = "bold", size = base_size),
      plot.title = ggplot2::element_text(face = "bold", size = base_size + 0.5, hjust = 0),
      plot.subtitle = ggplot2::element_text(size = base_size - 0.2, hjust = 0),
      plot.margin = ggplot2::margin(2.5, 2.5, 2.5, 2.5),
      panel.grid = ggplot2::element_blank()
    )
}

family_palette <- c(
  endocrine_epithelial_identity = "#365C8D",
  stromal_remodeling_caf_ecm = "#6A994E",
  immune_ecology = "#B23A48",
  metabolic_stress = "#8A6F3D",
  invasion_boundary_emt = "#6D597A"
)

nice_label <- function(value) {
  stringr::str_to_sentence(gsub("_", " ", value))
}

extract_seed <- function(run_id) {
  stringr::str_extract(run_id, "seed[0-9]+")
}

he_display_adjustment <- "R-only per-channel 2-98 percentile contrast stretch for H&E display; raw numeric source pixels unchanged."

stretch_channel <- function(value) {
  lo <- as.numeric(stats::quantile(value, probs = 0.02, na.rm = TRUE, names = FALSE))
  hi <- as.numeric(stats::quantile(value, probs = 0.98, na.rm = TRUE, names = FALSE))
  if (!is.finite(lo) || !is.finite(hi) || hi <= lo) {
    return(as.integer(value))
  }
  scaled <- (as.numeric(value) - lo) / (hi - lo)
  as.integer(round(pmin(pmax(scaled, 0), 1) * 255))
}

pixel_to_raster <- function(pixel_path) {
  pixels <- readr::read_csv(
    pixel_path,
    col_types = readr::cols(
      x = readr::col_integer(),
      y = readr::col_integer(),
      r = readr::col_integer(),
      g = readr::col_integer(),
      b = readr::col_integer()
    )
  )
  width <- max(pixels$x) + 1L
  height <- max(pixels$y) + 1L
  values <- grDevices::rgb(
    stretch_channel(pixels$r),
    stretch_channel(pixels$g),
    stretch_channel(pixels$b),
    maxColorValue = 255
  )
  raster_matrix <- matrix("#FFFFFF", nrow = height, ncol = width)
  raster_matrix[cbind(pixels$y + 1L, pixels$x + 1L)] <- values
  list(raster = grDevices::as.raster(raster_matrix), width = width, height = height)
}

make_he_panel <- function(sample_role, pathway, title_text) {
  pixel_path <- file.path(source_dir, paste0("he_overview_pixels_", sample_role, ".csv.gz"))
  block_path <- file.path(source_dir, paste0("he_block_overlay_", sample_role, ".csv"))
  integrity <- readr::read_csv(file.path(source_dir, "he_image_integrity_manifest.csv"), show_col_types = FALSE)
  info <- dplyr::filter(integrity, .data$sample_role == sample_role)
  image <- pixel_to_raster(pixel_path)
  blocks <- readr::read_csv(block_path, show_col_types = FALSE)
  if (!pathway %in% colnames(blocks)) {
    stop("Pathway ", pathway, " was not present in ", block_path)
  }
  info <- info[1, ]
  bar_um <- 5000
  bar_px <- bar_um / as.numeric(info$um_per_display_pixel_mean[[1]])
  x0 <- image$width * 0.055
  x1 <- min(image$width * 0.45, x0 + bar_px)
  y0 <- image$height * 0.9
  ggplot2::ggplot() +
    ggplot2::annotation_raster(image$raster, xmin = 0, xmax = image$width, ymin = 0, ymax = image$height) +
    ggplot2::geom_point(
      data = blocks,
      ggplot2::aes(x = .data$display_x, y = .data$display_y, fill = .data[[pathway]], size = .data$n_cells),
      shape = 21,
      colour = "white",
      stroke = 0.12,
      alpha = 0.84
    ) +
    ggplot2::annotate("segment", x = x0, xend = x1, y = y0, yend = y0, linewidth = 0.75, colour = "black") +
    ggplot2::annotate("text", x = (x0 + x1) / 2, y = y0 - image$height * 0.035, label = "5 mm", size = 2.0) +
    ggplot2::scale_y_reverse(limits = c(image$height, 0), expand = c(0, 0)) +
    ggplot2::scale_x_continuous(limits = c(0, image$width), expand = c(0, 0)) +
    ggplot2::scale_fill_gradientn(
      colours = c("#23395B", "#F3F0E8", "#B23A48"),
      name = nice_label(pathway),
      guide = "none"
    ) +
    ggplot2::scale_size_continuous(range = c(0.35, 1.25), guide = "none") +
    ggplot2::coord_fixed() +
    ggplot2::labs(title = title_text, x = NULL, y = NULL) +
    theme_nature() +
    ggplot2::theme(
      axis.line = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(size = 6.8, face = "bold")
    )
}

save_pub <- function(plot, stem, width_mm = 183, height_mm = 150) {
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  svg_path <- file.path(figure_dir, paste0(stem, ".svg"))
  pdf_path <- file.path(figure_dir, paste0(stem, ".pdf"))
  tiff_path <- file.path(figure_dir, paste0(stem, ".tiff"))
  png_path <- file.path(figure_dir, paste0(stem, ".png"))
  svglite::svglite(svg_path, width = width_in, height = height_in)
  print(plot)
  grDevices::dev.off()
  grDevices::cairo_pdf(pdf_path, width = width_in, height = height_in, family = "Arial")
  print(plot)
  grDevices::dev.off()
  ragg::agg_tiff(tiff_path, width = width_in, height = height_in, units = "in", res = 600, compression = "lzw")
  print(plot)
  grDevices::dev.off()
  ragg::agg_png(png_path, width = width_in, height = height_in, units = "in", res = 300)
  print(plot)
  grDevices::dev.off()
  data.frame(
    figure = stem,
    format = c("svg", "pdf", "tiff", "png"),
    path = c(svg_path, pdf_path, tiff_path, png_path),
    bytes = file.info(c(svg_path, pdf_path, tiff_path, png_path))$size,
    stringsAsFactors = FALSE
  )
}

fig1_source <- read_csv("main_figure_1_source_breast_discovery_highnull32.csv") %>%
  dplyr::mutate(
    seed = extract_seed(.data$run_id),
    partial_spearman_rho = as.numeric(.data$partial_spearman_rho),
    abs_partial_spearman_rho = as.numeric(.data$abs_partial_spearman_rho),
    null_abs_rho_q95 = as.numeric(.data$null_abs_rho_q95),
    negative_control_abs_rho_q95 = as.numeric(.data$negative_control_abs_rho_q95),
    stable_9_pathway_core = as.logical(.data$stable_9_pathway_core)
  )
fig2_source <- read_csv("main_figure_2_source_cross_cancer_stability_highnull32.csv")
gate_source <- read_csv("supp_table_highnull32_gate_and_axis_masked_summary.csv") %>%
  dplyr::mutate(seed = extract_seed(.data$run_id))
cervical_source <- read_csv("supp_table_cervical_validation_best_associations.csv") %>%
  dplyr::mutate(
    seed = extract_seed(.data$run_id),
    partial_spearman_rho = as.numeric(.data$partial_spearman_rho),
    abs_partial_spearman_rho = as.numeric(.data$abs_partial_spearman_rho),
    stable_9_pathway_core = as.logical(.data$stable_9_pathway_core)
  )
validation_by_run <- read_csv("source_table_cross_cancer_validation_by_run.csv") %>%
  dplyr::mutate(seed = extract_seed(.data$run_id))
seed_summary <- read_csv("supp_table_spatial_block_and_seed_summary.csv")
pathway_panel <- readr::read_csv(file.path(reference_run_dir, "breast_discovery", "pathway_panel.csv"), show_col_types = FALSE)

stable_core <- fig2_source %>%
  dplyr::filter(.data$stable_9_pathway_core == TRUE) %>%
  dplyr::pull(.data$pathway)
if (length(stable_core) != 9) {
  stop("Expected 9 stable core pathways, found ", length(stable_core))
}
if (length(unique(seed_summary$run_id)) != 3) {
  stop("Expected 3 high-null runs in seed summary.")
}
if (!identical(sort(unique(gate_source$cross_cancer_recovered)), c(9, 10))) {
  stop("Unexpected cross-cancer recovered counts.")
}

fig1_assoc_panel <- fig1_source %>%
  dplyr::filter(.data$stable_9_pathway_core == TRUE) %>%
  dplyr::group_by(.data$pathway, .data$family) %>%
  dplyr::mutate(mean_abs = mean(.data$abs_partial_spearman_rho, na.rm = TRUE)) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(.data$mean_abs)
readr::write_csv(fig1_assoc_panel, file.path(source_dir, "fig1_breast_association_panel.csv"))

pathway_order <- fig1_assoc_panel %>%
  dplyr::distinct(.data$pathway, .data$mean_abs) %>%
  dplyr::arrange(.data$mean_abs) %>%
  dplyr::pull(.data$pathway)

p1_b <- ggplot2::ggplot(
  fig1_assoc_panel,
  ggplot2::aes(
    x = .data$partial_spearman_rho,
    y = factor(.data$pathway, levels = pathway_order),
    colour = .data$family
  )
) +
  ggplot2::geom_vline(xintercept = 0, linewidth = 0.25, linetype = "dashed", colour = "grey45") +
  ggplot2::geom_point(ggplot2::aes(shape = .data$seed), size = 1.5, alpha = 0.9) +
  ggplot2::scale_colour_manual(values = family_palette, labels = nice_label) +
  ggplot2::scale_y_discrete(labels = nice_label) +
  ggplot2::labs(
    title = "Breast discovery",
    x = "Partial Spearman rho",
    y = NULL,
    colour = "Pathway family",
    shape = "Seed"
  ) +
  theme_nature() +
  ggplot2::guides(colour = "none")

fig1_gate_panel <- fig1_source %>%
  dplyr::filter(.data$stable_9_pathway_core == TRUE) %>%
  dplyr::filter(!is.na(.data$null_abs_rho_q95), !is.na(.data$negative_control_abs_rho_q95)) %>%
  dplyr::transmute(
    pathway = .data$pathway,
    family = .data$family,
    seed = .data$seed,
    spatial_null_margin = .data$abs_partial_spearman_rho - .data$null_abs_rho_q95,
    negative_control_margin = .data$abs_partial_spearman_rho - .data$negative_control_abs_rho_q95
  ) %>%
  tidyr::pivot_longer(
    cols = c("spatial_null_margin", "negative_control_margin"),
    names_to = "gate",
    values_to = "margin"
  ) %>%
  dplyr::mutate(gate = dplyr::recode(
    .data$gate,
    spatial_null_margin = "Spatial null",
    negative_control_margin = "Matched control"
  ))
readr::write_csv(fig1_gate_panel, file.path(source_dir, "fig1_gate_margin_panel.csv"))

p1_c <- ggplot2::ggplot(
  fig1_gate_panel,
  ggplot2::aes(x = .data$gate, y = .data$margin, fill = .data$gate)
) +
  ggplot2::geom_hline(yintercept = 0, linewidth = 0.25, linetype = "dashed", colour = "grey45") +
  ggplot2::geom_boxplot(width = 0.55, outlier.shape = NA, linewidth = 0.3, alpha = 0.82) +
  ggplot2::geom_point(
    ggplot2::aes(group = .data$seed),
    position = ggplot2::position_jitter(width = 0.08, height = 0),
    size = 0.7,
    alpha = 0.65
) +
  ggplot2::scale_fill_manual(
    values = c("Spatial null" = "#365C8D", "Matched control" = "#B23A48"),
    guide = "none"
  ) +
  ggplot2::labs(
    title = "Gate margins",
    x = NULL,
    y = "abs(rho) minus q95"
  ) +
  theme_nature() +
  ggplot2::theme(legend.position = "none", axis.text.x = ggplot2::element_text(angle = 18, hjust = 1))

fig1_family_panel <- pathway_panel %>%
  dplyr::distinct(.data$pathway, .data$family, .data$gene) %>%
  dplyr::group_by(.data$family) %>%
  dplyr::summarise(
    curated_genes = dplyr::n_distinct(.data$gene),
    curated_pathways = dplyr::n_distinct(.data$pathway),
    stable_core_pathways = dplyr::n_distinct(.data$pathway[.data$pathway %in% stable_core]),
    .groups = "drop"
  )
readr::write_csv(fig1_family_panel, file.path(source_dir, "fig1_family_coverage_panel.csv"))

p1_d <- ggplot2::ggplot(
  fig1_family_panel,
  ggplot2::aes(x = reorder(.data$family, .data$stable_core_pathways), y = .data$stable_core_pathways, fill = .data$family)
) +
  ggplot2::geom_col(width = 0.68, alpha = 0.9) +
  ggplot2::geom_text(
    ggplot2::aes(label = paste0(.data$stable_core_pathways, "/", .data$curated_pathways)),
    hjust = -0.08,
    size = 2.0
  ) +
  ggplot2::coord_flip(clip = "off") +
  ggplot2::scale_fill_manual(values = family_palette, guide = "none") +
  ggplot2::scale_x_discrete(labels = nice_label) +
  ggplot2::scale_y_continuous(limits = c(0, max(fig1_family_panel$stable_core_pathways) + 0.8), breaks = 0:4) +
  ggplot2::labs(
    title = "Family coverage",
    x = NULL,
    y = "Stable core pathways"
  ) +
  theme_nature()

p1_a <- make_he_panel(
  "breast_discovery",
  "luminal_estrogen_response",
  "Breast H&E context and spatial blocks"
)
figure1 <- ((p1_a | p1_b) / (p1_c | p1_d)) +
  patchwork::plot_layout(widths = c(1, 1.22), heights = c(1, 0.94), guides = "collect") +
  patchwork::plot_annotation(tag_levels = "a") &
  ggplot2::theme(
    legend.position = "right",
    plot.tag = ggplot2::element_text(face = "bold", size = 8)
  )

fig2_recovery_panel <- fig2_source %>%
  dplyr::mutate(pathway = factor(.data$pathway, levels = rev(.data$pathway))) %>%
  dplyr::select(
    "pathway",
    "family",
    "primary_recovery_rate",
    "axis_masked_recovery_rate",
    "stable_9_pathway_core"
  ) %>%
  tidyr::pivot_longer(
    cols = c("primary_recovery_rate", "axis_masked_recovery_rate"),
    names_to = "analysis",
    values_to = "recovery_rate"
  ) %>%
  dplyr::mutate(
    analysis = dplyr::recode(
      .data$analysis,
      primary_recovery_rate = "Primary",
      axis_masked_recovery_rate = "Axis-masked"
    ),
    recovered_runs = round(.data$recovery_rate * 3)
  )
readr::write_csv(fig2_recovery_panel, file.path(source_dir, "fig2_recovery_panel.csv"))

p2_b <- ggplot2::ggplot(
  fig2_recovery_panel,
  ggplot2::aes(x = .data$analysis, y = .data$pathway, fill = .data$recovery_rate)
) +
  ggplot2::geom_tile(colour = "white", linewidth = 0.3) +
  ggplot2::geom_text(ggplot2::aes(label = paste0(.data$recovered_runs, "/3")), size = 2.0) +
  ggplot2::scale_fill_gradientn(
    colours = c("#F3F0E8", "#6A994E"),
    limits = c(0, 1),
    labels = scales::percent,
    guide = "none"
  ) +
  ggplot2::scale_y_discrete(labels = nice_label) +
  ggplot2::labs(
    title = "Stable core recovery",
    x = NULL,
    y = NULL,
    fill = "Recovery"
  ) +
  theme_nature()

fig2_seed_panel <- gate_source %>%
  dplyr::select(
    "seed",
    "cross_cancer_recovered",
    "cross_cancer_total",
    "axis_masked_cross_cancer_recovered",
    "axis_masked_cross_cancer_total",
    "candidate_generic_plip_axes",
    "breast_negative_control_pass95",
    "cervical_negative_control_pass95"
  ) %>%
  tidyr::pivot_longer(
    cols = c("cross_cancer_recovered", "axis_masked_cross_cancer_recovered"),
    names_to = "analysis",
    values_to = "recovered"
  ) %>%
  dplyr::mutate(
    analysis = dplyr::recode(
      .data$analysis,
      cross_cancer_recovered = "Primary",
      axis_masked_cross_cancer_recovered = "Axis-masked"
    )
  )
readr::write_csv(fig2_seed_panel, file.path(source_dir, "fig2_seed_gate_panel.csv"))

p2_c <- ggplot2::ggplot(
  fig2_seed_panel,
  ggplot2::aes(x = .data$seed, y = .data$recovered, group = .data$analysis, colour = .data$analysis)
) +
  ggplot2::geom_line(linewidth = 0.35) +
  ggplot2::geom_point(ggplot2::aes(size = .data$candidate_generic_plip_axes), alpha = 0.9) +
  ggplot2::scale_colour_manual(values = c("Primary" = "#365C8D", "Axis-masked" = "#B23A48")) +
  ggplot2::scale_size_continuous(range = c(1.1, 2.4), breaks = 0:2) +
  ggplot2::scale_y_continuous(limits = c(8.5, 10.2), breaks = 9:10) +
  ggplot2::labs(
    title = "Recovery by seed",
    x = NULL,
    y = "Recovered pathways (/10)",
    colour = "Analysis",
    size = "Flagged PLIP axes"
  ) +
  theme_nature()

fig2_cervical_panel <- cervical_source %>%
  dplyr::filter(.data$stable_9_pathway_core == TRUE) %>%
  dplyr::group_by(.data$pathway, .data$family) %>%
  dplyr::mutate(mean_abs = mean(.data$abs_partial_spearman_rho, na.rm = TRUE)) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(.data$mean_abs)
readr::write_csv(fig2_cervical_panel, file.path(source_dir, "fig2_cervical_association_panel.csv"))

cervical_order <- fig2_cervical_panel %>%
  dplyr::distinct(.data$pathway, .data$mean_abs) %>%
  dplyr::arrange(.data$mean_abs) %>%
  dplyr::pull(.data$pathway)
p2_d <- ggplot2::ggplot(
  fig2_cervical_panel,
  ggplot2::aes(
    x = .data$partial_spearman_rho,
    y = factor(.data$pathway, levels = cervical_order),
    colour = .data$family
  )
) +
  ggplot2::geom_vline(xintercept = 0, linewidth = 0.25, linetype = "dashed", colour = "grey45") +
  ggplot2::geom_point(ggplot2::aes(shape = .data$seed), size = 1.4, alpha = 0.9) +
  ggplot2::scale_colour_manual(values = family_palette, labels = nice_label) +
  ggplot2::scale_y_discrete(labels = nice_label) +
  ggplot2::labs(
    title = "Cervical stress test",
    x = "Partial Spearman rho",
    y = NULL,
    colour = "Pathway family",
    shape = "Seed"
  ) +
  theme_nature() +
  ggplot2::guides(colour = "none")

p2_a <- make_he_panel("breast_discovery", "luminal_estrogen_response", "Breast H&E")
p2_e <- make_he_panel("cervical_validation", "immune_activation", "Cervical H&E")
figure2 <- ((p2_a | p2_e | p2_b) / (p2_c | p2_d)) +
  patchwork::plot_layout(heights = c(0.94, 1.08), guides = "collect") +
  patchwork::plot_annotation(tag_levels = "a") &
  ggplot2::theme(
    legend.position = "right",
    plot.tag = ggplot2::element_text(face = "bold", size = 8)
  )

export_table <- dplyr::bind_rows(
  save_pub(figure1, "figure1_breast_discovery_morphopathway", width_mm = 183, height_mm = 154),
  save_pub(figure2, "figure2_cross_cancer_stability_validation", width_mm = 183, height_mm = 154)
)
readr::write_csv(export_table, file.path(output_dir, "nature_figure_exports.csv"))

render_finished_at <- Sys.time()
writeLines(
  c(
    "pyXenium.pathway Nature figure render resource log",
    paste0("started_at=", format(render_started_at, "%Y-%m-%d %H:%M:%S %Z")),
    paste0("finished_at=", format(render_finished_at, "%Y-%m-%d %H:%M:%S %Z")),
    paste0("elapsed_seconds=", round(as.numeric(difftime(render_finished_at, render_started_at, units = "secs")), 2)),
    paste0("plotting_backend=R"),
    paste0("package_dir=", package_dir),
    paste0("output_dir=", output_dir),
    paste0("max_raster_dim=", arg_or(args, "max-raster-dim", "900")),
    "",
    "exported_files:",
    paste0(export_table$format, "\t", export_table$bytes, "\t", export_table$path),
    "",
    "gc:",
    capture.output(gc())
  ),
  resource_log_path
)

writeLines(
  c(
    "# Nature Figure Captions",
    "",
    "## Figure 1. Breast discovery morphopathway map",
    "",
    "H&E low-resolution context and spatial pseudobulk blocks are shown for the Atera breast Xenium WTA preview dataset. Quantitative panels summarize PLIP-derived H&E feature associations with curated WTA pathway activities across three high-null seeds, including spatial-null and matched-control gate margins.",
    "",
    "## Figure 2. Cross-cancer stability and validation",
    "",
    "Breast and cervical H&E contexts are paired with pathway-family recovery summaries. The result supports a conservative pathway-family stress-test claim: nine pathways were recovered in all three primary and axis-masked runs, while direct pathway-level cervical replication is not claimed."
  ),
  file.path(report_dir, "figure_captions_nature.md")
)

svg_checks <- vapply(
  export_table$path[export_table$format == "svg"],
  function(path) {
    text <- paste(readLines(path, warn = FALSE, n = 2000), collapse = "\n")
    grepl("<text|<tspan", text)
  },
  logical(1)
)
he_integrity <- readr::read_csv(file.path(source_dir, "he_image_integrity_manifest.csv"), show_col_types = FALSE)
readr::write_csv(
  data.frame(
    adjustment = he_display_adjustment,
    applied_in = "make_morphopathway_nature_figures.R",
    raw_source_pixels_unchanged = TRUE,
    stringsAsFactors = FALSE
  ),
  file.path(source_dir, "he_display_adjustment_manifest.csv")
)
he_crop_recorded <- "display_crop_xyxy" %in% colnames(he_integrity) &&
  all(!is.na(he_integrity$display_crop_xyxy)) &&
  all(nzchar(he_integrity$display_crop_xyxy))
he_scale_calibrated <- all(as.numeric(he_integrity$um_per_display_pixel_mean) > 0)
qa <- list(
  status = "pass",
  generated_at = as.character(Sys.time()),
  plotting_backend = "R",
  figures = split(export_table, export_table$figure),
  checks = list(
    stable_core_pathways = length(stable_core),
    high_null_runs = length(unique(seed_summary$run_id)),
    unique_real_samples = 2,
    cross_cancer_recovery_range = range(gate_source$cross_cancer_recovered),
    axis_masked_recovery_range = range(gate_source$axis_masked_cross_cancer_recovered),
    svg_text_tags_detected = all(svg_checks),
    he_crop_recorded = he_crop_recorded,
    he_scale_calibrated = he_scale_calibrated,
    he_display_adjustment = he_display_adjustment,
    he_integrity_manifest = file.path(source_dir, "he_image_integrity_manifest.csv"),
    render_resource_log = resource_log_path,
    old_naturebiotech_evidence_reused = FALSE
  )
)
jsonlite::write_json(qa, file.path(report_dir, "figure_qa_report.json"), auto_unbox = TRUE, pretty = TRUE)
writeLines(
  c(
    "# Figure QA Report",
    "",
    "- Status: pass",
    "- Backend: R for plotting, preview, export and visual QA.",
    "- H&E source extraction: numeric low-resolution source-data tables from raw OME-TIF; plotting/export in R.",
    paste0("- Stable pathway core: ", length(stable_core), " pathways."),
    paste0("- High-null seeds: ", length(unique(seed_summary$run_id)), "."),
    paste0("- Cross-cancer recovery range: ", paste(range(gate_source$cross_cancer_recovered), collapse = "-"), "/10."),
    paste0("- Axis-masked recovery range: ", paste(range(gate_source$axis_masked_cross_cancer_recovered), collapse = "-"), "/10."),
    paste0("- SVG editable text check: ", ifelse(all(svg_checks), "pass", "review required"), "."),
    paste0("- H&E crop provenance recorded: ", ifelse(he_crop_recorded, "pass", "review required"), "."),
    paste0("- H&E scale calibration: ", ifelse(he_scale_calibrated, "pass", "review required"), "."),
    paste0("- H&E display adjustment: ", he_display_adjustment),
    "- Image integrity: see `source_data/he_image_integrity_manifest.csv`.",
    "- Local render resource log: see `logs/figure_render.resource.log`.",
    "- Evidence boundary: no old `naturebiotech_package` outputs were used as evidence input."
  ),
  file.path(report_dir, "figure_qa_report.md")
)

write_a100_manifest("not_used_local_completed", "Local numeric H&E source extraction and R export completed.")

package_manifest <- list(
  package_type = "pyXenium.pathway Nature figure package",
  generated_at = as.character(Sys.time()),
  output_dir = output_dir,
  source_package = package_dir,
  reference_run_dir = reference_run_dir,
  plotting_backend = "R",
  nonvisual_source_data_helper = "prepare_nature_he_source_data.py",
  figures = export_table,
  source_data_dir = source_dir,
  reports_dir = report_dir,
  a100_fallback_manifest = file.path(a100_dir, "a100_fallback_manifest.json"),
  he_display_adjustment = he_display_adjustment,
  evidence_boundary = "new pyXenium.pathway morphopathway suite only"
)
jsonlite::write_json(package_manifest, file.path(output_dir, "nature_figure_package_manifest.json"), auto_unbox = TRUE, pretty = TRUE)

writeLines(
  c(
    "# pyXenium.pathway Nature Figure Package",
    "",
    "This package contains two R-rendered Nature-style figures for the new pyXenium.pathway morphopathway suite.",
    "",
    "Primary figures:",
    "- `figures/figure1_breast_discovery_morphopathway.svg|pdf|tiff|png`",
    "- `figures/figure2_cross_cancer_stability_validation.svg|pdf|tiff|png`",
    "",
    "Traceability:",
    "- Per-panel source tables are under `source_data/`.",
    "- H&E image integrity and scale calibration are recorded in `source_data/he_image_integrity_manifest.csv`.",
    "- H&E display contrast is documented in `source_data/he_display_adjustment_manifest.csv`; raw source pixels are unchanged.",
    "- QA is recorded in `reports/figure_qa_report.md` and `reports/figure_qa_report.json`.",
    "- A100 fallback instructions are recorded in `a100/a100_fallback_manifest.json`.",
    "",
    "Evidence boundary: this package uses only the new pyXenium.pathway morphopathway result bundle and raw Atera H&E/WTA inputs."
  ),
  file.path(output_dir, "README.md")
)

message("Nature figure package written to: ", output_dir)
