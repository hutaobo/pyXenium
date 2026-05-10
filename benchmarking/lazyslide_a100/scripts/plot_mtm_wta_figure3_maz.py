from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Figure 3-ready MAZ ring-level evidence panels.",
    )
    parser.add_argument(
        "--package-dir",
        default=(
            "docs/_static/tutorials/multimodal_histoseg_lazyslide_breast_wta/"
            "naturebiotech_package"
        ),
    )
    parser.add_argument("--max-profile-panels", type=int, default=8)
    return parser.parse_args()


def _zscore(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    std = numeric.std(ddof=0)
    if not np.isfinite(std) or std == 0:
        return numeric * np.nan
    return (numeric - numeric.mean()) / std


def _label(text: str) -> str:
    return str(text).replace("_", " ")


def _load(package_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profile = pd.read_csv(package_dir / "MAZ_RingLevel_Profile.csv")
    report = pd.read_csv(package_dir / "MAZ_RingLevel_LeadLag_Report.csv")
    sensitivity = pd.read_csv(package_dir / "MAZ_RingWidth_Sensitivity_Summary.csv")
    return profile, report, sensitivity


def _stable_candidates(report: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    stable = sensitivity.loc[sensitivity["stable_class"].eq("coupled_boundary_zone")].copy()
    merged = stable.merge(
        report,
        on=["program", "assigned_structure"],
        how="left",
        suffixes=("_sensitivity", ""),
    )
    merged["sort_score"] = (
        pd.to_numeric(merged["min_ring_profile_spearman_rho"], errors="coerce").fillna(-np.inf)
        + np.log10(pd.to_numeric(merged["min_image_observations"], errors="coerce").fillna(1.0))
    )
    return merged.sort_values(
        ["assigned_structure", "sort_score", "program"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _plot_sensitivity_heatmap(ax: plt.Axes, sensitivity: pd.DataFrame) -> None:
    class_to_value = {
        "coupled_boundary_zone": 3,
        "molecular_lead": 2,
        "morphology_lead": 1,
        "weak_ring_gradient": 0,
        "unstable": -1,
    }
    programs = list(dict.fromkeys(sensitivity["program"].astype(str).tolist()))
    structures = sorted(sensitivity["assigned_structure"].astype(str).unique().tolist())
    matrix = np.full((len(programs), len(structures)), np.nan)
    labels = [["" for _ in structures] for _ in programs]
    for _, row in sensitivity.iterrows():
        i = programs.index(str(row["program"]))
        j = structures.index(str(row["assigned_structure"]))
        stable_class = str(row["stable_class"])
        matrix[i, j] = class_to_value.get(stable_class, -1)
        labels[i][j] = stable_class.replace("_", "\n")
    cmap = matplotlib.colors.ListedColormap(["#7f7f7f", "#d9d9d9", "#fdae61", "#abd9e9", "#2c7bb6"])
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(structures)), structures)
    ax.set_yticks(range(len(programs)), [_label(program) for program in programs])
    ax.set_title("Ring-width sensitivity across MAZ candidates")
    for i in range(len(programs)):
        for j in range(len(structures)):
            if labels[i][j]:
                color = "white" if matrix[i, j] in {2, 3} else "black"
                ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=6, color=color)
    ax.tick_params(axis="x", rotation=0)


def _profile_summary(profile: pd.DataFrame, structure: str, feature: str, feature_type: str) -> pd.DataFrame:
    subset = profile.loc[
        profile["assigned_structure"].astype(str).eq(str(structure))
        & profile["feature"].astype(str).eq(str(feature))
        & profile["feature_type"].astype(str).eq(feature_type)
    ].copy()
    if subset.empty:
        return pd.DataFrame(columns=["ring_center_um", "mean_value", "n_observations", "z_mean"])
    out = (
        subset.groupby("ring_center_um", as_index=False)
        .agg(mean_value=("mean_value", "mean"), n_observations=("n_observations", "sum"))
        .sort_values("ring_center_um")
    )
    out["z_mean"] = _zscore(out["mean_value"])
    return out


def _plot_profile(ax: plt.Axes, profile: pd.DataFrame, candidate: pd.Series) -> None:
    structure = str(candidate["assigned_structure"])
    program = str(candidate["program"])
    target = str(candidate["molecular_feature"])
    image = str(candidate["image_feature"])
    sign = float(np.sign(pd.to_numeric(candidate["global_partial_spearman_rho"], errors="coerce")))
    if sign == 0 or not np.isfinite(sign):
        sign = 1.0
    molecular = _profile_summary(profile, structure, target, "molecular")
    image_profile = _profile_summary(profile, structure, image, "image")
    image_profile["z_mean"] = sign * image_profile["z_mean"]
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.plot(
        molecular["ring_center_um"],
        molecular["z_mean"],
        marker="o",
        color="#2c7bb6",
        label="WTA program",
    )
    ax.plot(
        image_profile["ring_center_um"],
        image_profile["z_mean"],
        marker="s",
        color="#d7191c",
        label="H&E axis",
    )
    ax.set_title(f"{structure}: {_label(program)}", fontsize=9)
    ax.set_xlabel("Signed distance to contour boundary (um)")
    ax.set_ylabel("Ring mean z-score")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.02,
        0.04,
        (
            f"rho range {candidate['min_ring_profile_spearman_rho']:.2f}-"
            f"{candidate['max_ring_profile_spearman_rho']:.2f}\n"
            f"median lag {candidate['median_peak_shift_um']:.0f} um"
        ),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
    )


def _write_candidate_table(stable: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "program",
        "assigned_structure",
        "stable_class",
        "median_peak_shift_um",
        "min_ring_profile_spearman_rho",
        "max_ring_profile_spearman_rho",
        "min_image_observations",
        "min_molecular_observations",
    ]
    stable.loc[:, columns].to_csv(output_path, index=False)


def main() -> None:
    args = _parse_args()
    package_dir = Path(args.package_dir).resolve()
    profile, report, sensitivity = _load(package_dir)
    stable = _stable_candidates(report, sensitivity)
    _write_candidate_table(stable, package_dir / "Figure3_MAZ_Stable_Candidates.csv")

    pdf_path = package_dir / "Final_Figure3_MAZ_Pack.pdf"
    heatmap_path = package_dir / "figure3_maz_sensitivity_heatmap.png"
    profile_path = package_dir / "figure3_maz_stable_coupled_profiles.png"

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        _plot_sensitivity_heatmap(axes[0], sensitivity)
        counts = sensitivity["stable_class"].value_counts().sort_values(ascending=True)
        axes[1].barh([_label(item) for item in counts.index], counts.values, color="#4c78a8")
        axes[1].set_title("Candidate stability across ring-bin settings")
        axes[1].set_xlabel("Program-structure pairs")
        for i, value in enumerate(counts.values):
            axes[1].text(value + 0.1, i, str(int(value)), va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(heatmap_path, dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        show = stable.head(int(args.max_profile_panels)).copy()
        n = len(show)
        cols = 2
        rows = int(np.ceil(max(n, 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(11, max(4, 3.6 * rows)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (_, candidate) in zip(axes.ravel(), show.iterrows()):
            ax.axis("on")
            _plot_profile(ax, profile, candidate)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle("Stable coupled MAZ ring profiles", y=0.995, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(profile_path, dpi=220, bbox_inches="tight")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        summary = [
            "Figure 3 MAZ package",
            "",
            f"Ring profile rows: {len(profile):,}",
            f"Ring-level lead-lag rows: {len(report):,}",
            f"Sensitivity rows: {len(sensitivity):,}",
            f"Stable coupled candidates: {int(stable.shape[0])}",
            "",
            "Strongest stable coupled S3 candidates:",
        ]
        s3 = stable.loc[stable["assigned_structure"].astype(str).eq("S3")]
        for _, row in s3.head(4).iterrows():
            summary.append(
                (
                    f"- {_label(row['program'])}: min rho "
                    f"{row['min_ring_profile_spearman_rho']:.2f}, "
                    f"median lag {row['median_peak_shift_um']:.0f} um"
                )
            )
        ax.text(0.04, 0.96, "\n".join(summary), va="top", ha="left", fontsize=13)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {pdf_path}")
    print(f"Wrote {heatmap_path}")
    print(f"Wrote {profile_path}")
    print(f"Stable candidates: {stable.shape[0]}")


if __name__ == "__main__":
    main()
