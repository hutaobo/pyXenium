from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "naturebiotech_package"


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "pyXenium").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the mTM WTA Nature Biotechnology package autopilot locally.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--min-abs-partial-rho", type=float, default=0.5)
    parser.add_argument("--max-programs", type=int, default=3)
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path, log: list[str]) -> None:
    log.append(f"[RUN] {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.stdout.strip():
        log.append(result.stdout.strip())
    if result.stderr.strip():
        log.append("[STDERR]\n" + result.stderr.strip())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _write_skeleton(output_dir: Path, log: list[str]) -> None:
    selected = _read_csv(output_dir / "figure2_selected_programs.csv")
    maz = _read_csv(output_dir / "MAZ_LeadLag_Report.csv")
    hero = _read_csv(output_dir / "figure2_hero_contours.csv")
    top_rows = []
    for _, row in selected.iterrows():
        top_rows.append(
            "- {structure} {target}: H&E axis `{image}`, partial rho={rho:.3f}, n={n}".format(
                structure=row.get("assigned_structure"),
                target=str(row.get("target_feature", "")).replace("program__wta_", ""),
                image=row.get("image_feature"),
                rho=float(row.get("partial_spearman_rho", float("nan"))),
                n=int(row.get("n_contours", 0)),
            )
        )
    maz_summary = "No MAZ proxy rows generated."
    if not maz.empty:
        lead_counts = maz["lead_lag_class"].value_counts().to_dict()
        maz_summary = ", ".join(f"{key}: {value}" for key, value in lead_counts.items())
    skeleton = f"""# Manuscript Skeleton v1

## Working Title

Contour-constrained H&E foundation model embeddings decode hidden molecular continua in single-cell spatial transcriptomics.

## One-sentence Pitch

We introduce morphomolecular translation mapping, a contour-constrained framework that anchors pathology foundation-model embeddings to biologically meaningful HistoSeg tissue structures and decodes residual WTA molecular programs within apparently homogeneous histologic compartments.

## Abstract Draft

Spatial whole-transcriptome imaging now measures thousands of genes at single-cell resolution, but the relationship between routine H&E morphology and local molecular state remains difficult to quantify. We developed a contour-constrained morphomolecular translation workflow that combines HistoSeg tissue structures, direct WSI LazySlide embeddings, and Atera WTA contour summaries. In a breast cancer WTA sample, discrete HistoSeg labels captured much of the coarse molecular variance, so H&E embeddings did not improve all cross-validated molecular prediction benchmarks beyond structure labels. However, after controlling for tissue structure, spatial position, and boundary proximity, foundation-model embedding axes remained strongly associated with residual WTA programs. The strongest Figure 2 candidates were:

{chr(10).join(top_rows)}

These results support a residual-decoding interpretation: HistoSeg defines compartments, while H&E embeddings resolve continuous molecular states inside those compartments.

## Main Results

### Result 1: Direct WSI contour anchoring creates the mTM coordinate system

The workflow maps 3,114 LazySlide PLIP tiles onto 1,578 HistoSeg contours and aggregates tile embeddings, text-prompt scores, cell summaries, and WTA gene programs at contour scale.

### Result 2: H&E embeddings decode an S3 luminal/ER continuum beyond labels

The primary Figure 2 package focuses on S3 contours. The selected luminal estrogen-response axis passes the quality gate and shows a strong within-structure residual association.

### Result 3: Metabolic and stress programs create a second hidden axis

UPR and oxidative phosphorylation also pass the quality gate in S3, suggesting that contour-scale H&E embeddings capture metabolic or stress-linked tissue states rather than only epithelial identity.

### Result 4: Boundary MAZ evidence is currently a proxy, not yet a manuscript claim

The MAZ lead-lag table currently uses contour-level distance-to-boundary summaries. Lead-lag class counts: {maz_summary}. Ring-level validation should be the next automated branch before making a strong Molecularly Active Zone claim.

## Figure Legends

### Figure 1

Overview of morphomolecular translation mapping. HistoSeg defines tissue contours, LazySlide extracts direct WSI foundation-model embeddings, and Atera WTA profiles are aggregated within the same contours.

### Figure 2

Hidden within-structure molecular continua in breast cancer WTA. For selected S3 programs, panels show a single HistoSeg compartment label, the H&E embedding gradient, the matched WTA program gradient, and the residual image-molecular association.

### Figure 3

Candidate molecularly active zones at tissue interfaces. Contour-level boundary proxy analysis nominates programs and structures for ring-level validation of molecular lead or morphological lag.

## Required Caveats

- The current benchmark does not support a blanket claim that H&E embeddings outperform HistoSeg labels for molecular prediction.
- The stronger claim is residual: H&E embeddings carry within-structure molecular information after labels and spatial covariates are fixed.
- MAZ lead-lag rows are contour-level proxies and need ring-level validation before main-text use.

## Generated Files

- `Final_Figure2_Pack.pdf`
- `figure2_selected_programs.csv`
- `figure2_hero_contours.csv`
- `MAZ_LeadLag_Report.csv`
- `Autopilot_Decision_Log.txt`
"""
    (output_dir / "Manuscript_Skeleton_v1.md").write_text(skeleton, encoding="utf-8")
    log.append("[WRITE] Manuscript_Skeleton_v1.md")
    if not hero.empty:
        log.append(f"[INFO] Figure 2 hero contours: {len(hero)}")


def _write_status(output_dir: Path, log: list[str]) -> None:
    status = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "figure2_pdf": str(output_dir / "Final_Figure2_Pack.pdf"),
        "maz_report": str(output_dir / "MAZ_LeadLag_Report.csv"),
        "manuscript_skeleton": str(output_dir / "Manuscript_Skeleton_v1.md"),
        "decision_log": str(output_dir / "Autopilot_Decision_Log.txt"),
    }
    (output_dir / "AUTOPILOT_STATUS.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.append("[WRITE] AUTOPILOT_STATUS.json")


def main() -> None:
    args = _parse_args()
    repo = _repo_root()
    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = [
        "mTM WTA Autopilot Decision Log",
        f"Started UTC: {datetime.now(timezone.utc).isoformat()}",
        "Directive: prioritize residual partial-correlation evidence over delta R2.",
    ]
    script_dir = repo / "benchmarking" / "lazyslide_a100" / "scripts"
    try:
        _run(
            [
                sys.executable,
                str(script_dir / "plot_morphomolecular_wta_figure2.py"),
                "--data-dir",
                args.data_dir,
                "--output-dir",
                args.output_dir,
                "--min-abs-partial-rho",
                str(args.min_abs_partial_rho),
                "--max-programs",
                str(args.max_programs),
            ],
            cwd=repo,
            log=log,
        )
        _run(
            [
                sys.executable,
                str(script_dir / "boundary_lead_lag_analysis.py"),
                "--data-dir",
                args.data_dir,
                "--output-dir",
                args.output_dir,
            ],
            cwd=repo,
            log=log,
        )
        _write_skeleton(output_dir, log)
        _write_status(output_dir, log)
    except Exception as exc:
        log.append(f"[FAIL] {type(exc).__name__}: {exc}")
        (output_dir / "LOG_FOR_BOSS.md").write_text("\n".join(log) + "\n", encoding="utf-8")
        raise
    else:
        log.append(f"Completed UTC: {datetime.now(timezone.utc).isoformat()}")
        (output_dir / "Autopilot_Decision_Log.txt").write_text(
            "\n\n".join(log) + "\n",
            encoding="utf-8",
        )
        (output_dir / "LOG_FOR_BOSS.md").write_text(
            "# LOG_FOR_BOSS\n\nNo blocking failures in the latest autopilot pass.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
