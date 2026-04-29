from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MINIMUM_SPATIALPERTURB_VERSION = "0.3"
SPATIALPERTURB_REQUIREMENT = f"SpatialPerturb>={MINIMUM_SPATIALPERTURB_VERSION}"
DEFAULT_SPATIALPERTURB_REFERENCE_DATASETS = ("gse241115_breast_cropseq",)

_PROGRAM_SIMILARITY_CAVEAT = (
    "SpatialPerturb Bridge scores represent Perturb-seq-derived program similarity "
    "projected onto Xenium tissue. They do not mean a tissue cell contains the "
    "corresponding knockout, guide, or drug perturbation."
)


@dataclass(frozen=True)
class SpatialPerturbBridgeConfig:
    """Configuration for a pyXenium-to-SpatialPerturb handoff specification."""

    xenium_path: str | Path
    output_dir: str | Path
    prepared_h5ad: str | Path | None = None
    cache_dir: str | Path = ".spatialperturb-cache"
    cell_group_path: str | Path | None = None
    roi_geojson_path: str | Path | None = None
    sample_name: str | None = None
    reference_datasets: Sequence[str] | str = DEFAULT_SPATIALPERTURB_REFERENCE_DATASETS
    install_requirement: str = SPATIALPERTURB_REQUIREMENT


def spatialperturb_status() -> dict[str, Any]:
    """Return environment status for the optional external SpatialPerturb package."""

    version = None
    try:
        version = importlib.metadata.version("SpatialPerturb")
    except importlib.metadata.PackageNotFoundError:
        try:
            version = importlib.metadata.version("spatialperturb")
        except importlib.metadata.PackageNotFoundError:
            version = None

    import_available = importlib.util.find_spec("spatialperturb") is not None
    cli_path = shutil.which("spatialperturb")
    return {
        "bridge": "SpatialPerturb Bridge",
        "distribution": "SpatialPerturb",
        "import_name": "spatialperturb",
        "minimum_version": MINIMUM_SPATIALPERTURB_VERSION,
        "requirement": SPATIALPERTURB_REQUIREMENT,
        "installed": bool(import_available or version),
        "import_available": bool(import_available),
        "installed_version": version,
        "cli_path": cli_path,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_compatible": sys.version_info >= (3, 9),
        "install_command": ["python", "-m", "pip", "install", SPATIALPERTURB_REQUIREMENT],
    }


def build_spatialperturb_handoff(config: SpatialPerturbBridgeConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Build a JSON-serializable handoff spec for the external SpatialPerturb CLI."""

    cfg = _coerce_config(config)
    reference_datasets = _normalize_reference_datasets(cfg.reference_datasets)
    if not reference_datasets:
        raise ValueError("SpatialPerturbBridgeConfig.reference_datasets must contain at least one dataset name.")

    xenium_path = _path_text(cfg.xenium_path)
    output_dir = _path_text(cfg.output_dir)
    prepared_h5ad = _path_text(cfg.prepared_h5ad or Path(cfg.output_dir) / "spatialperturb_xenium.h5ad")
    cache_dir = _path_text(cfg.cache_dir)

    prepare_command = ["spatialperturb", "prepare-xenium", xenium_path, prepared_h5ad]
    if cfg.cell_group_path is not None:
        prepare_command.extend(["--cell-group-path", _path_text(cfg.cell_group_path)])
    if cfg.roi_geojson_path is not None:
        prepare_command.extend(["--roi-geojson-path", _path_text(cfg.roi_geojson_path)])
    if cfg.sample_name:
        prepare_command.extend(["--sample-name", str(cfg.sample_name)])

    run_command = [
        "spatialperturb",
        "run-reference-benchmark",
        prepared_h5ad,
        output_dir,
        "--cache-dir",
        cache_dir,
        "--reference-datasets",
        ",".join(reference_datasets),
    ]

    install_command = ["python", "-m", "pip", "install", str(cfg.install_requirement)]
    return {
        "schema_version": 1,
        "bridge": "SpatialPerturb Bridge",
        "description": "Optional pyXenium bridge for Perturb-seq reference projection onto Xenium tissue.",
        "external_project": {
            "name": "SpatialPerturb",
            "package": "SpatialPerturb",
            "import_name": "spatialperturb",
            "minimum_version": MINIMUM_SPATIALPERTURB_VERSION,
            "python_requires": ">=3.9",
            "source": "https://github.com/hutaobo/SpatialPerturb",
            "pypi": "https://pypi.org/project/SpatialPerturb/",
        },
        "status": spatialperturb_status(),
        "inputs": {
            "xenium_path": xenium_path,
            "cell_group_path": _optional_path_text(cfg.cell_group_path),
            "roi_geojson_path": _optional_path_text(cfg.roi_geojson_path),
            "sample_name": cfg.sample_name,
        },
        "outputs": {
            "prepared_h5ad": prepared_h5ad,
            "report_dir": output_dir,
        },
        "cache_dir": cache_dir,
        "reference_datasets": list(reference_datasets),
        "commands": {
            "install": install_command,
            "prepare_xenium": prepare_command,
            "run_reference_benchmark": run_command,
        },
        "command_text": {
            "install": _format_command(install_command),
            "prepare_xenium": _format_command(prepare_command),
            "run_reference_benchmark": _format_command(run_command),
        },
        "interpretation_caveat": _PROGRAM_SIMILARITY_CAVEAT,
        "pyxenium_boundary": (
            "pyXenium provides the Xenium data foundation and handoff specification; "
            "SpatialPerturb owns the perturbation reference projection workflow."
        ),
    }


def write_spatialperturb_handoff(
    config: SpatialPerturbBridgeConfig | Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write a SpatialPerturb Bridge handoff spec as JSON and return the spec."""

    spec = build_spatialperturb_handoff(config)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def _coerce_config(config: SpatialPerturbBridgeConfig | Mapping[str, Any]) -> SpatialPerturbBridgeConfig:
    if isinstance(config, SpatialPerturbBridgeConfig):
        return config
    if isinstance(config, Mapping):
        return SpatialPerturbBridgeConfig(**dict(config))
    raise TypeError("config must be a SpatialPerturbBridgeConfig or mapping.")


def _path_text(value: str | Path) -> str:
    return str(Path(value))


def _optional_path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return _path_text(value)


def _normalize_reference_datasets(names: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(names, str):
        return tuple(name.strip() for name in names.split(",") if name.strip())
    return tuple(str(name) for name in names if str(name))


def _format_command(args: Sequence[str]) -> str:
    return " ".join(_quote_arg(str(arg)) for arg in args)


def _quote_arg(arg: str) -> str:
    if not arg or any(char.isspace() for char in arg) or any(char in arg for char in '"\'<>|&;()'):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg
