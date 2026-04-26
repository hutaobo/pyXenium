from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class ContourGmiConfig:
    """Configuration for canonical contour-native GMI workflows."""

    contour_key: str = "s1_s5_contours"
    contour_label_col: str = "assigned_structure"
    positive_label: str = "S1"
    negative_label: str = "S5"
    contour_geojson: str | None = None
    contour_pixel_size_um: float = 0.2125
    contour_id_key: str = "name"
    feature_count: int = 500
    spatial_feature_count: int = 100
    min_cells_per_contour: int = 20
    min_library_size: float = 1.0
    min_feature_prevalence: float = 0.05
    layer: str | None = None
    random_seed: int = 1
    coordinate_shuffle: bool = False
    exclude_coordinate_spatial_features: bool = False
    include_pathomics: bool = False
    inner_rim_um: float = 20.0
    outer_rim_um: float = 30.0
    rscript: str = "Rscript"
    r_lib_path: str | None = None
    install_gmi: bool = True
    force_reinstall_gmi: bool = False
    penalty: str = "SCAD"
    lambda_min_ratio: float = 0.02
    n_lambda: int = 100
    eta: float = 0.6
    tune: str = "EBIC"
    ebic_gamma: float = 1.0
    max_iter: int = 50
    spatial_cv_folds: int = 0
    bootstrap_repeats: int = 0
    bootstrap_fraction: float = 0.8
    run_label_permutation_control: bool = False
    run_coordinate_shuffle_control: bool = False
    run_spatial_feature_shuffle_control: bool = False
    run_within_label_heterogeneity: bool = True
    heterogeneity_min_contours: int = 6
    write_spatial_visualizations: bool = True
    visualization_genes: tuple[str, ...] = ("NIBAN1", "SORL1", "CCND1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy_with(self, **updates: Any) -> "ContourGmiConfig":
        return replace(self, **updates)


@dataclass
class ContourGmiDataset:
    """A contour-level design matrix and binary endpoint for GMI."""

    X: pd.DataFrame
    y: pd.Series
    sample_metadata: pd.DataFrame
    feature_metadata: pd.DataFrame
    config: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def subset(self, sample_ids: Sequence[str]) -> "ContourGmiDataset":
        ids = pd.Index([str(item) for item in sample_ids])
        return ContourGmiDataset(
            X=self.X.loc[ids].copy(),
            y=self.y.loc[ids].copy(),
            sample_metadata=self.sample_metadata.set_index("sample_id", drop=False).loc[ids].reset_index(drop=True),
            feature_metadata=self.feature_metadata.copy(),
            config=dict(self.config),
            provenance=dict(self.provenance),
        )


@dataclass
class ContourGmiResult:
    """Parsed GMI fit outputs and report metadata."""

    output_dir: Path
    main_effects: pd.DataFrame
    interaction_effects: pd.DataFrame
    groups: pd.DataFrame
    predictions: pd.DataFrame
    cv_metrics: pd.DataFrame
    stability: pd.DataFrame
    heterogeneity: pd.DataFrame
    summary: dict[str, Any]
    files: dict[str, str]


# Backwards-compatible names from the original spatial GMI prototype. These
# aliases no longer imply tile construction; they point at the contour workflow.
SpatialGmiConfig = ContourGmiConfig
SpatialGmiDataset = ContourGmiDataset
SpatialGmiResult = ContourGmiResult
