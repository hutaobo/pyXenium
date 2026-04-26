from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


DEFAULT_MECHANOSTRESS_RADII_UM = (5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0)


@dataclass(frozen=True)
class AxisStrengthConfig:
    """Configuration for axial orientation and ANE density analysis."""

    er_threshold: float = 2.0
    radii_um: tuple[float, ...] = DEFAULT_MECHANOSTRESS_RADII_UM
    groupby: tuple[str, ...] = ("cluster",)
    local_k: int = 15
    angle_col: str = "axis_angle_radians"
    x_col: str = "centroid_x"
    y_col: str = "centroid_y"
    cell_query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy_with(self, **updates: Any) -> "AxisStrengthConfig":
        return replace(self, **updates)


@dataclass(frozen=True)
class TumorStromaGrowthConfig:
    """Configuration for tumor-stroma growth pattern classification."""

    annotation_col: str = "Annotation"
    tumor_label: str = "Tumor"
    stroma_label: str = "Stromal"
    x_col: str = "x_centroid"
    y_col: str = "y_centroid"
    cell_id_col: str = "cell_id"
    method: str = "delaunay_hop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy_with(self, **updates: Any) -> "TumorStromaGrowthConfig":
        return replace(self, **updates)


@dataclass(frozen=True)
class PolarityConfig:
    """Configuration for cell/nucleus centroid polarity analysis."""

    offset_norm_threshold: float = 0.30
    cell_id_col: str = "cell_id"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def copy_with(self, **updates: Any) -> "PolarityConfig":
        return replace(self, **updates)


@dataclass(frozen=True)
class MechanostressConfig:
    """Configuration for the integrated mechanostress workflow."""

    axis: AxisStrengthConfig = field(default_factory=AxisStrengthConfig)
    tumor_stroma: TumorStromaGrowthConfig = field(default_factory=TumorStromaGrowthConfig)
    polarity: PolarityConfig = field(default_factory=PolarityConfig)
    coupling_genes: tuple[str, ...] = ()
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.to_dict(),
            "tumor_stroma": self.tumor_stroma.to_dict(),
            "polarity": self.polarity.to_dict(),
            "coupling_genes": tuple(self.coupling_genes),
            "sample_id": self.sample_id,
        }

    def copy_with(self, **updates: Any) -> "MechanostressConfig":
        return replace(self, **updates)


@dataclass
class MechanostressResult:
    """Computed mechanostress tables, summary metadata, and artifact paths."""

    cell_axes: pd.DataFrame
    axis_strength_by_radius: pd.DataFrame
    orientation_summary: pd.DataFrame
    tumor_growth_cells: pd.DataFrame
    tumor_growth_summary: pd.DataFrame
    distance_expression_coupling: pd.DataFrame
    cell_polarity: pd.DataFrame
    polarity_summary: pd.DataFrame
    summary: dict[str, Any]
    config: Mapping[str, Any]
    files: dict[str, str] = field(default_factory=dict)
    output_dir: Path | None = None

    def table_map(self) -> dict[str, pd.DataFrame]:
        return {
            "cell_axes": self.cell_axes,
            "axis_strength_by_radius": self.axis_strength_by_radius,
            "orientation_summary": self.orientation_summary,
            "tumor_growth_cells": self.tumor_growth_cells,
            "tumor_growth_summary": self.tumor_growth_summary,
            "distance_expression_coupling": self.distance_expression_coupling,
            "cell_polarity": self.cell_polarity,
            "polarity_summary": self.polarity_summary,
        }


@dataclass
class MechanostressCohortResult:
    """Per-sample mechanostress results and cohort-level artifact paths."""

    results: dict[str, MechanostressResult] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    files: dict[str, str] = field(default_factory=dict)
    output_dir: Path | None = None


def normalize_radii(values: Sequence[float]) -> tuple[float, ...]:
    radii = tuple(float(value) for value in values)
    if not radii:
        raise ValueError("At least one radius is required.")
    if any(value <= 0 for value in radii):
        raise ValueError("All radii must be greater than 0.")
    return radii
