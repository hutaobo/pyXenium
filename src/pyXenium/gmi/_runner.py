from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from ._types import ContourGmiConfig
from ._vendor import assert_vendored_gmi_complete, get_vendored_gmi_path


REQUIRED_R_PACKAGES = ("cPCG", "MASS", "Rcpp", "RcppArmadillo", "RcppEigen")


def _r_string(value: str | Path) -> str:
    return json.dumps(str(value).replace("\\", "/"))


def _r_env(config: ContourGmiConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.r_lib_path:
        lib_path = str(Path(config.r_lib_path))
        Path(lib_path).mkdir(parents=True, exist_ok=True)
        existing = env.get("R_LIBS_USER", "")
        env["R_LIBS_USER"] = lib_path if not existing else os.pathsep.join([lib_path, existing])
    return env


def check_r_packages(
    packages: Sequence[str] = REQUIRED_R_PACKAGES,
    *,
    rscript: str = "Rscript",
    r_lib_path: str | None = None,
) -> list[str]:
    expr = (
        "pkgs <- c("
        + ",".join(_r_string(package) for package in packages)
        + "); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; "
        "cat(paste(missing, collapse='\\n'))"
    )
    config = ContourGmiConfig(rscript=rscript, r_lib_path=r_lib_path)
    result = subprocess.run(
        [rscript, "-e", expr],
        text=True,
        capture_output=True,
        env=_r_env(config),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not inspect R packages with {rscript!r}: {result.stderr.strip()}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def build_gmi_install_command(
    *,
    rscript: str = "Rscript",
    vendor_path: str | Path | None = None,
) -> list[str]:
    vendor = Path(vendor_path) if vendor_path is not None else get_vendored_gmi_path()
    expr = (
        "vendor <- normalizePath("
        + _r_string(vendor)
        + ", mustWork=TRUE); "
        "install.packages(vendor, repos=NULL, type='source')"
    )
    return [rscript, "-e", expr]


def ensure_vendored_gmi_installed(config: ContourGmiConfig) -> None:
    assert_vendored_gmi_complete()
    missing = check_r_packages(rscript=config.rscript, r_lib_path=config.r_lib_path)
    if missing:
        raise RuntimeError(
            "Gmi R dependencies are missing: "
            + ", ".join(missing)
            + ". Install them in the active R environment before running pyXenium.gmi."
        )

    has_gmi = check_r_packages(("Gmi",), rscript=config.rscript, r_lib_path=config.r_lib_path) == []
    if has_gmi and not config.force_reinstall_gmi:
        return

    command = build_gmi_install_command(rscript=config.rscript)
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=_r_env(config),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install vendored Gmi from local source. "
            f"Command: {' '.join(command)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def _runner_script() -> str:
    return r'''
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 11) {
  stop("Expected arguments: train_x train_meta out_dir test_x_or_empty penalty lambda_min_ratio n_lambda eta tune ebic_gamma max_iter")
}
train_x_path <- args[[1]]
train_meta_path <- args[[2]]
out_dir <- args[[3]]
test_x_path <- args[[4]]
penalty <- args[[5]]
lambda_min_ratio <- as.numeric(args[[6]])
n_lambda <- as.integer(args[[7]])
eta <- as.numeric(args[[8]])
tune <- args[[9]]
ebic_gamma <- as.numeric(args[[10]])
max_iter <- as.integer(args[[11]])
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (!requireNamespace("Gmi", quietly = TRUE)) {
  stop("R package 'Gmi' is not installed. pyXenium installs only from the local vendored source.")
}

X <- read.table(train_x_path, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE, comment.char = "")
meta <- read.table(train_meta_path, sep = "\t", header = TRUE, check.names = FALSE, comment.char = "")
y <- meta$y[match(rownames(X), meta$sample_id)]
if (any(is.na(y))) {
  stop("Sample metadata does not provide y for every design-matrix row.")
}
y <- as.numeric(y)
Xmat <- as.matrix(X)
fit <- Gmi::Gmi(
  Xmat,
  y,
  beta = rep(0, ncol(Xmat)),
  lambda.min.ratio = lambda_min_ratio,
  n.lambda = n_lambda,
  penalty = penalty,
  eta = eta,
  tune = tune,
  ebic.gamma = ebic_gamma,
  max.iter = max_iter
)
saveRDS(fit, file.path(out_dir, "gmi_fit.rds"))

features <- colnames(Xmat)
main_ind <- fit$mainInd
if (is.null(main_ind) || length(main_ind) == 0) {
  main_df <- data.frame(feature_index=integer(), feature=character(), coefficient=numeric())
} else {
  main_df <- data.frame(
    feature_index = as.integer(main_ind),
    feature = features[as.integer(main_ind)],
    coefficient = as.numeric(fit$beta.m)
  )
}
write.table(main_df, file.path(out_dir, "main_effects.tsv"), sep="\t", quote=FALSE, row.names=FALSE)

inter_names <- fit$interInd
inter_beta <- fit$beta.i
if (is.null(inter_names) || length(inter_names) == 0) {
  inter_df <- data.frame(
    interaction=character(), feature_index_a=integer(), feature_index_b=integer(),
    feature_a=character(), feature_b=character(), coefficient=numeric()
  )
} else {
  parse_pair <- function(value) {
    pair <- as.integer(strsplit(value, "X")[[1]][2:3])
    if (length(pair) != 2 || any(is.na(pair))) {
      return(c(NA_integer_, NA_integer_))
    }
    pair
  }
  pairs <- t(vapply(inter_names, parse_pair, integer(2)))
  inter_df <- data.frame(
    interaction = as.character(inter_names),
    feature_index_a = pairs[, 1],
    feature_index_b = pairs[, 2],
    feature_a = features[pairs[, 1]],
    feature_b = features[pairs[, 2]],
    coefficient = as.numeric(inter_beta)
  )
}
write.table(inter_df, file.path(out_dir, "interaction_effects.tsv"), sep="\t", quote=FALSE, row.names=FALSE)

predict_and_write <- function(new_x_path, target_path, split_name) {
  if (is.null(new_x_path) || new_x_path == "" || !file.exists(new_x_path)) {
    return(invisible(NULL))
  }
  new_x <- read.table(new_x_path, sep = "\t", header = TRUE, row.names = 1, check.names = FALSE, comment.char = "")
  pred <- tryCatch(
    as.numeric(Gmi::predict_Gmi(fit, as.matrix(new_x), type = "response")),
    error = function(e) rep(NA_real_, nrow(new_x))
  )
  pred_df <- data.frame(sample_id = rownames(new_x), prediction = pred, split = split_name)
  write.table(pred_df, target_path, sep="\t", quote=FALSE, row.names=FALSE)
}
predict_and_write(train_x_path, file.path(out_dir, "predictions.tsv"), "train")
predict_and_write(test_x_path, file.path(out_dir, "predictions_test.tsv"), "test")

diagnostics <- data.frame(
  cri_loc = ifelse(is.null(fit$cri.loc), NA_integer_, as.integer(fit$cri.loc)),
  selected_main = nrow(main_df),
  selected_interactions = nrow(inter_df),
  selected_lambda = ifelse(is.null(fit$lambda) || is.null(fit$cri.loc), NA_real_, fit$lambda[fit$cri.loc])
)
write.table(diagnostics, file.path(out_dir, "gmi_diagnostics.tsv"), sep="\t", quote=FALSE, row.names=FALSE)
'''


def run_gmi_fit(
    *,
    design_matrix_path: str | Path,
    sample_metadata_path: str | Path,
    output_dir: str | Path,
    config: ContourGmiConfig,
    prediction_matrix_path: str | Path | None = None,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if config.install_gmi:
        ensure_vendored_gmi_installed(config)

    params_path = out / "gmi_params.json"
    params = {
        "penalty": config.penalty,
        "lambda_min_ratio": config.lambda_min_ratio,
        "n_lambda": config.n_lambda,
        "eta": config.eta,
        "tune": config.tune,
        "ebic_gamma": config.ebic_gamma,
        "max_iter": config.max_iter,
    }
    params_path.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")

    with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False, encoding="utf-8") as handle:
        handle.write(_runner_script())
        script_path = Path(handle.name)

    try:
        command = [
            config.rscript,
            str(script_path),
            str(design_matrix_path),
            str(sample_metadata_path),
            str(out),
            "" if prediction_matrix_path is None else str(prediction_matrix_path),
            str(config.penalty),
            str(config.lambda_min_ratio),
            str(config.n_lambda),
            str(config.eta),
            str(config.tune),
            str(config.ebic_gamma),
            str(config.max_iter),
        ]
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=_r_env(config),
            check=False,
        )
        (out / "gmi_stdout.log").write_text(result.stdout, encoding="utf-8")
        (out / "gmi_stderr.log").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"Gmi R runner failed with exit code {result.returncode}. "
                f"See {out / 'gmi_stderr.log'} for details."
            )
    finally:
        try:
            script_path.unlink()
        except FileNotFoundError:
            pass

    files = {
        "gmi_fit_rds": str(out / "gmi_fit.rds"),
        "main_effects": str(out / "main_effects.tsv"),
        "interaction_effects": str(out / "interaction_effects.tsv"),
        "predictions": str(out / "predictions.tsv"),
        "diagnostics": str(out / "gmi_diagnostics.tsv"),
        "stdout": str(out / "gmi_stdout.log"),
        "stderr": str(out / "gmi_stderr.log"),
        "params": str(params_path),
    }
    if prediction_matrix_path is not None:
        files["predictions_test"] = str(out / "predictions_test.tsv")
    return files


def read_gmi_outputs(output_dir: str | Path) -> dict[str, pd.DataFrame]:
    out = Path(output_dir)

    def read_tsv(name: str) -> pd.DataFrame:
        path = out / name
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, sep="\t")

    return {
        "main_effects": read_tsv("main_effects.tsv"),
        "interaction_effects": read_tsv("interaction_effects.tsv"),
        "predictions": read_tsv("predictions.tsv"),
        "predictions_test": read_tsv("predictions_test.tsv"),
        "diagnostics": read_tsv("gmi_diagnostics.tsv"),
    }
