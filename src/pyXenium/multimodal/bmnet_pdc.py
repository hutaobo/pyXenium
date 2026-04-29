from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyXenium.contour import add_contours_from_geojson, build_contour_feature_table
from pyXenium.io import read_xenium
from pyXenium.io.sdata_model import XeniumSData

from .contour_boundary_ecology import score_contour_boundary_programs
from .morphology_increment import compare_he_vs_xenium_morphology_sources
from .pathology import BMNetBackend

__all__ = [
    "BMNetMorphologyPilotConfig",
    "DeterministicBreastBMNetLikeBackend",
    "HuggingFacePathologyBackboneBackend",
    "TimmBMNetLikeBackend",
    "build_bmnet_pilot_backend",
    "run_bmnet_morphology_increment_pilot",
]


BMNET_LABELS = ("normal", "benign", "in_situ", "invasive")
DEFAULT_HF_PATHOLOGY_MODEL = "1aurent/vit_small_patch8_224.lunit_dino"
DEFAULT_PDC_XENIUM_ROOT = (
    "/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/"
    "data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs"
)
DEFAULT_PDC_CONTOUR_GEOJSON = "xenium_explorer_annotations.s1_s5.generated.geojson"


@dataclass
class BMNetMorphologyPilotConfig:
    """Configuration for a breast H&E morphology increment pilot."""

    output_dir: str | Path
    dataset_root: str | Path | None = None
    contour_geojson: str | Path | None = None
    contour_key: str = "s1_s5_contours"
    contour_id_key: str = "polygon_id"
    contour_coordinate_space: str = "xenium_pixel"
    contour_pixel_size_um: float | None = None
    he_image_key: str = "he"
    cells_parquet: str | None = "cells.parquet"
    clusters_relpath: str | None = "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"
    cluster_column_name: str = "cluster"
    backend: str = "deterministic-smoke"
    checkpoint: str | Path | None = None
    hf_model: str = DEFAULT_HF_PATHOLOGY_MODEL
    timm_architecture: str = "mobilenetv3_small_100"
    timm_pretrained: bool = False
    max_contours: int | None = None
    inner_rim_um: float = 20.0
    outer_rim_um: float = 30.0
    include_pathomics: bool = True
    include_transcripts: bool = False
    program_library: str = "breast_boundary_bmnet_v1"
    random_state: int = 0
    min_contours: int = 8


class DeterministicBreastBMNetLikeBackend:
    """Dependency-free BM-Net-like smoke backend.

    The scores are deterministic H&E color/texture proxies. They are useful for
    validating contour cropping, artifact writing, and PDC orchestration, but are
    not trained BM-Net predictions.
    """

    feature_prefix = "bmnet"

    def __init__(self) -> None:
        self._adapter = BMNetBackend(predict_fn=self._predict, labels=BMNET_LABELS)

    def extract_features(self, image_patch: Any, **metadata: Any) -> dict[str, float]:
        return self._adapter.extract_features(image_patch, **metadata)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "deterministic-smoke",
            "model_source": "deterministic_h_and_e_proxy",
            "checkpoint": None,
            "trained_on": None,
            "semantic_status": "smoke_test_only_not_biological_evidence",
            "labels": list(BMNET_LABELS),
        }

    @staticmethod
    def _predict(image_patch: Any, **_: Any) -> dict[str, float]:
        rgb = _to_rgb_float(image_patch)
        if rgb.size == 0:
            return {label: 0.25 for label in BMNET_LABELS}
        mean = np.nanmean(rgb, axis=(0, 1))
        std = float(np.nanstd(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]))
        brightness = float(np.nanmean(rgb))
        rgb_sum = rgb.sum(axis=2) + 1e-6
        blue_ratio = float(np.nanmean(rgb[:, :, 2] / rgb_sum))
        pink_ratio = float(np.nanmean((rgb[:, :, 0] + 0.5 * rgb[:, :, 1]) / rgb_sum))
        dark_fraction = float(np.mean(np.mean(rgb, axis=2) < 0.58))

        invasive = _sigmoid(7.5 * (pink_ratio - 0.37) + 2.5 * dark_fraction + 1.5 * std)
        in_situ = _sigmoid(6.0 * (blue_ratio - 0.34) + 1.2 * dark_fraction + 0.7 * std)
        normal = _sigmoid(5.0 * (brightness - 0.78) - 2.0 * dark_fraction)
        benign = _sigmoid(3.0 * (mean[1] - mean[0] * 0.85) + 2.0 * (0.75 - std))
        return _normalize_probabilities(
            {
                "normal": normal,
                "benign": benign,
                "in_situ": in_situ,
                "invasive": invasive,
            }
        )


class TimmBMNetLikeBackend:
    """Optional MobileNetV3-small classifier with BM-Net-like output semantics."""

    feature_prefix = "bmnet"

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        architecture: str = "mobilenetv3_small_100",
        pretrained: bool = False,
        device: str = "auto",
    ) -> None:
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.architecture = str(architecture)
        self.pretrained = bool(pretrained)
        self.device_name = device
        self._adapter = BMNetBackend(predict_fn=self._predict, labels=BMNET_LABELS)

        try:
            import torch
            import timm
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "The timm BM-Net-like backend requires optional dependencies "
                "`torch` and `timm`."
            ) from exc

        self._torch = torch
        self._device = _resolve_torch_device(torch, device)
        self._model = timm.create_model(
            self.architecture,
            pretrained=self.pretrained,
            num_classes=len(BMNET_LABELS),
        )
        if self.checkpoint:
            payload = torch.load(self.checkpoint, map_location="cpu")
            state_dict = _extract_state_dict(payload)
            missing, unexpected = self._model.load_state_dict(state_dict, strict=False)
            self._load_state = {
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }
        else:
            self._load_state = {"missing_keys": [], "unexpected_keys": []}
        self._model.to(self._device)
        self._model.eval()

    def extract_features(self, image_patch: Any, **metadata: Any) -> dict[str, float]:
        return self._adapter.extract_features(image_patch, **metadata)

    def metadata(self) -> dict[str, Any]:
        trained_on = "checkpoint" if self.checkpoint else "untrained_or_imagenet_pretrained_head"
        return {
            "backend": "bmnet-local" if self.checkpoint else "bmnet-like-trainable",
            "model_source": "timm_mobilenetv3_small_bmnet_like",
            "architecture": self.architecture,
            "checkpoint": self.checkpoint,
            "pretrained_backbone": self.pretrained,
            "trained_on": trained_on,
            "semantic_status": (
                "trained_checkpoint" if self.checkpoint else "pipeline_smoke_only_without_breast_checkpoint"
            ),
            "labels": list(BMNET_LABELS),
            "load_state": self._load_state,
        }

    def _predict(self, image_patch: Any, **_: Any) -> dict[str, float]:
        torch = self._torch
        tensor = _torch_image_tensor(image_patch, torch=torch).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)
        return {label: float(probs[index]) for index, label in enumerate(BMNET_LABELS)}


class HuggingFacePathologyBackboneBackend:
    """Optional Hugging Face histopathology feature extractor.

    This backend emits `pathology__...` features rather than BM-Net class
    probabilities because these models are surrogates, not the BM-Net classifier.
    """

    feature_prefix = "pathology"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_HF_PATHOLOGY_MODEL,
        device: str = "auto",
        output_dim: int = 16,
    ) -> None:
        self.model_name = str(model_name)
        self.device_name = str(device)
        self.output_dim = int(output_dim)
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ImportError(
                "The Hugging Face pathology backend requires optional dependency `torch`."
            ) from exc
        self._torch = torch
        self._device = _resolve_torch_device(torch, device)
        self._processor = None
        self._backend_kind = "transformers"
        try:
            from transformers import AutoImageProcessor, AutoModel

            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
        except Exception as transformers_exc:  # pragma: no cover - optional dependency path
            try:
                import timm

                self._backend_kind = "timm_hf_hub"
                self._model = timm.create_model(f"hf_hub:{self.model_name}", pretrained=True, num_classes=0)
                self._transformers_error = repr(transformers_exc)
            except Exception as timm_exc:
                raise ImportError(
                    "The Hugging Face pathology backend could not load this model with "
                    "`transformers` or `timm`. Install the matching optional backend or "
                    "choose a transformers/timm-compatible pathology checkpoint."
                ) from timm_exc
        self._model.to(self._device)
        self._model.eval()

    def extract_features(self, image_patch: Any, **metadata: Any) -> dict[str, float]:
        if self._backend_kind == "timm_hf_hub":
            tensor = _torch_image_tensor(image_patch, torch=self._torch).to(self._device)
            with self._torch.no_grad():
                output = self._model(tensor)
            vector = _torch_output_vector(output)
            return _named_embedding_features(vector, output_dim=self.output_dim)

        pil_image = _patch_to_pil(image_patch)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        vector = _hf_output_vector(outputs)
        return _named_embedding_features(vector, output_dim=self.output_dim)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "hf-pathology-backbone",
            "model_source": "huggingface_pathology_surrogate",
            "hf_model": self.model_name,
            "backend_kind": self._backend_kind,
            "checkpoint": self.model_name,
            "trained_on": "model_card",
            "semantic_status": "surrogate_feature_extractor_not_bmnet_classifier",
            "feature_prefix": self.feature_prefix,
            "output_dim": self.output_dim,
        }


def build_bmnet_pilot_backend(
    backend: str,
    *,
    checkpoint: str | Path | None = None,
    hf_model: str = DEFAULT_HF_PATHOLOGY_MODEL,
    timm_architecture: str = "mobilenetv3_small_100",
    timm_pretrained: bool = False,
) -> Any:
    """Construct a pathology backend for the BM-Net morphology pilot."""

    key = str(backend).strip().lower().replace("_", "-")
    if key in {"deterministic-smoke", "dry-smoke", "smoke"}:
        return DeterministicBreastBMNetLikeBackend()
    if key in {"hf-pathology-backbone", "huggingface", "hf"}:
        return HuggingFacePathologyBackboneBackend(model_name=hf_model)
    if key in {"bmnet-like-trainable", "timm-bmnet-like", "timm"}:
        return TimmBMNetLikeBackend(
            checkpoint=checkpoint,
            architecture=timm_architecture,
            pretrained=timm_pretrained,
        )
    if key == "bmnet-local":
        if checkpoint is None:
            raise ValueError("`bmnet-local` requires `checkpoint`.")
        return TimmBMNetLikeBackend(
            checkpoint=checkpoint,
            architecture=timm_architecture,
            pretrained=False,
        )
    raise ValueError(
        "`backend` must be one of deterministic-smoke, bmnet-local, "
        "bmnet-like-trainable, or hf-pathology-backbone."
    )


def run_bmnet_morphology_increment_pilot(
    sdata_or_path: XeniumSData | str | Path | None = None,
    *,
    config: BMNetMorphologyPilotConfig | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run BM-Net/H&E morphology feature extraction and increment validation."""

    if config is None:
        if "output_dir" not in overrides:
            raise TypeError("`output_dir` is required when `config` is not provided.")
        config = BMNetMorphologyPilotConfig(**overrides)
    elif overrides:
        config = BMNetMorphologyPilotConfig(**{**asdict(config), **overrides})

    out = Path(config.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    sdata = _resolve_sdata(sdata_or_path, config=config)
    _ensure_contours(sdata, config=config)
    retained_contours = _limit_contours(sdata, contour_key=config.contour_key, max_contours=config.max_contours)

    backend = build_bmnet_pilot_backend(
        config.backend,
        checkpoint=config.checkpoint,
        hf_model=config.hf_model,
        timm_architecture=config.timm_architecture,
        timm_pretrained=config.timm_pretrained,
    )
    model_metadata = _backend_metadata(backend)
    feature_table = build_contour_feature_table(
        sdata,
        contour_key=config.contour_key,
        he_image_key=config.he_image_key,
        inner_rim_um=float(config.inner_rim_um),
        outer_rim_um=float(config.outer_rim_um),
        include_pathomics=bool(config.include_pathomics),
        pathology_backends=[backend],
    )
    program_result = score_contour_boundary_programs(
        sdata,
        contour_key=config.contour_key,
        feature_table=feature_table,
        program_library=config.program_library,
    )
    increment = compare_he_vs_xenium_morphology_sources(
        sdata,
        contour_key=config.contour_key,
        feature_table=feature_table,
        program_scores=program_result["program_scores"],
        output_dir=out,
        random_state=int(config.random_state),
        min_contours=int(config.min_contours),
    )

    contour_features = pd.DataFrame(feature_table["contour_features"]).copy()
    contour_features_path = out / "contour_features_with_bmnet.csv"
    contour_features.to_csv(contour_features_path, index=False)
    program_scores_path = out / "program_scores.csv"
    program_result["program_scores"].to_csv(program_scores_path, index=False)
    patch_predictions_path = out / "bmnet_patch_predictions.csv"
    _select_pathology_feature_columns(contour_features).to_csv(patch_predictions_path, index=False)

    summary = {
        "dataset_root": str(config.dataset_root) if config.dataset_root is not None else None,
        "output_dir": str(out),
        "contour_key": config.contour_key,
        "n_contours": int(len(contour_features)),
        "retained_contours": retained_contours,
        "backend": config.backend,
        "model_metadata": model_metadata,
        "program_library": config.program_library,
        "max_contours": config.max_contours,
        "artifact_files": {
            "contour_features_with_bmnet": str(contour_features_path),
            "bmnet_patch_predictions": str(patch_predictions_path),
            "program_scores": str(program_scores_path),
            "xenium_native_morphology": str(out / "xenium_native_morphology.csv"),
            "he_morphology_features": str(out / "he_morphology_features.csv"),
            "feature_redundancy": str(out / "feature_redundancy.csv"),
            "incremental_prediction": str(out / "incremental_prediction.csv"),
            "partial_associations": str(out / "partial_associations.csv"),
            "matched_review_table": str(out / "matched_review_table.csv"),
            "morphology_increment_summary": str(out / "morphology_increment_summary.json"),
        },
        "config": _json_ready_config(config),
    }
    summary_path = out / "bmnet_pdc_run_summary.json"
    summary["artifact_files"]["run_summary"] = str(summary_path)

    increment_summary = dict(increment["summary"])
    increment_summary["model_metadata"] = model_metadata
    increment_summary["run_summary_json"] = str(summary_path)
    (out / "morphology_increment_summary.json").write_text(
        json.dumps(increment_summary, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    return {
        "feature_table": feature_table,
        "program_result": program_result,
        "increment": increment,
        "summary": summary,
        "artifact_dir": str(out),
    }


def _resolve_sdata(
    sdata_or_path: XeniumSData | str | Path | None,
    *,
    config: BMNetMorphologyPilotConfig,
) -> XeniumSData:
    if isinstance(sdata_or_path, XeniumSData):
        return sdata_or_path
    dataset_root = sdata_or_path if sdata_or_path is not None else config.dataset_root
    if dataset_root is None:
        dataset_root = DEFAULT_PDC_XENIUM_ROOT
    return read_xenium(
        str(dataset_root),
        as_="sdata",
        prefer="h5",
        include_transcripts=bool(config.include_transcripts),
        include_boundaries=True,
        include_images=True,
        cells_parquet=config.cells_parquet,
        clusters_relpath=config.clusters_relpath,
        cluster_column_name=config.cluster_column_name,
    )


def _ensure_contours(sdata: XeniumSData, *, config: BMNetMorphologyPilotConfig) -> None:
    if config.contour_key in sdata.shapes:
        if bool(config.include_pathomics) and config.contour_key not in sdata.contour_images:
            raise ValueError(
                f"`sdata.shapes[{config.contour_key!r}]` exists but H&E contour patches are missing. "
                "Load/add contours with `extract_he_patches=True` or pass a GeoJSON before the shape exists."
            )
        return

    geojson = _resolve_contour_geojson(config)
    contour_id_key = _resolve_contour_id_key(geojson, requested=config.contour_id_key)
    add_contours_from_geojson(
        sdata,
        geojson,
        key=config.contour_key,
        id_key=contour_id_key,
        coordinate_space=config.contour_coordinate_space,
        pixel_size_um=config.contour_pixel_size_um,
        extract_he_patches=bool(config.include_pathomics),
        he_image_key=config.he_image_key,
    )


def _resolve_contour_geojson(config: BMNetMorphologyPilotConfig) -> Path:
    if config.contour_geojson is not None:
        candidate = Path(config.contour_geojson).expanduser()
        if not candidate.is_absolute() and config.dataset_root is not None:
            nested = Path(config.dataset_root) / candidate
            if nested.exists():
                candidate = nested
    elif config.dataset_root is not None:
        candidate = Path(config.dataset_root) / DEFAULT_PDC_CONTOUR_GEOJSON
    else:
        candidate = Path(DEFAULT_PDC_XENIUM_ROOT) / DEFAULT_PDC_CONTOUR_GEOJSON
    if not candidate.exists():
        raise FileNotFoundError(
            "Contour GeoJSON was not found. Pass `contour_geojson` or create "
            f"{DEFAULT_PDC_CONTOUR_GEOJSON} under the dataset root."
        )
    return candidate


def _resolve_contour_id_key(geojson: Path, *, requested: str) -> str:
    requested = str(requested or "auto")
    payload = json.loads(Path(geojson).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not features:
        return requested
    properties = [feature.get("properties", {}) if isinstance(feature, dict) else {} for feature in features]
    nested = [
        item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {}
        for item in properties
        if isinstance(item, dict)
    ]

    def values_for(key: str) -> list[Any]:
        values = []
        for index, item in enumerate(properties):
            value = item.get(key) if isinstance(item, dict) else None
            if value is None and index < len(nested):
                value = nested[index].get(key)
            values.append(value)
        return values

    def complete_unique(key: str) -> bool:
        values = values_for(key)
        text = [str(value) for value in values if value is not None]
        return len(text) == len(features) and len(set(text)) == len(text)

    if requested.lower() != "auto" and complete_unique(requested):
        return requested
    for candidate in ("polygon_id", "id", "contour_id", "name", "object_id", "annotation_id"):
        if complete_unique(candidate):
            return candidate
    if requested.lower() != "auto":
        return requested
    raise KeyError(
        "Unable to infer a unique contour id key from GeoJSON properties. "
        "Pass `--contour-id-key` explicitly."
    )


def _limit_contours(
    sdata: XeniumSData,
    *,
    contour_key: str,
    max_contours: int | None,
) -> list[str]:
    frame = sdata.shapes[contour_key]
    ordered = pd.Index(frame["contour_id"].astype(str)).drop_duplicates().astype(str).tolist()
    if max_contours is None or int(max_contours) <= 0 or len(ordered) <= int(max_contours):
        return ordered
    selected = ordered[: int(max_contours)]
    sdata.shapes[contour_key] = frame.loc[frame["contour_id"].astype(str).isin(selected)].copy()
    if contour_key in sdata.contour_images:
        sdata.contour_images[contour_key] = {
            contour_id: image
            for contour_id, image in sdata.contour_images[contour_key].items()
            if str(contour_id) in set(selected)
        }
    return selected


def _select_pathology_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    id_columns = [column for column in ("sample_id", "contour_key", "contour_id") if column in frame.columns]
    prefixes = (
        "bmnet__",
        "pathology__",
        "edge_contrast__bmnet__",
        "edge_contrast__pathology__",
    )
    feature_columns = [column for column in frame.columns if str(column).startswith(prefixes)]
    return frame.loc[:, id_columns + feature_columns].copy()


def _backend_metadata(backend: Any) -> dict[str, Any]:
    if hasattr(backend, "metadata"):
        metadata = backend.metadata()
        if isinstance(metadata, dict):
            return metadata
    return {
        "backend": getattr(backend, "feature_prefix", backend.__class__.__name__),
        "model_source": backend.__class__.__name__,
        "semantic_status": "unrecorded_backend_metadata",
    }


def _to_rgb_float(image_patch: Any) -> np.ndarray:
    arr = np.asarray(image_patch)
    if arr.size == 0:
        return np.zeros((0, 0, 3), dtype=float)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1 and arr.size in {1, 3, 4}:
        arr = arr.reshape(1, 1, arr.size)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3:
        raise ValueError(f"Expected an image-like array, got shape {arr.shape}.")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 2:
        arr = np.concatenate([arr, np.zeros((*arr.shape[:2], 1), dtype=arr.dtype)], axis=2)
    arr = arr[:, :, :3].astype(float)
    scale = 255.0 if np.nanmax(arr) > 1.5 else 1.0
    return np.clip(arr / scale, 0.0, 1.0)


def _patch_to_pil(image_patch: Any) -> Any:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise ImportError("Pillow is required for torch/Hugging Face image backends.") from exc
    rgb = (_to_rgb_float(image_patch) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _torch_image_tensor(image_patch: Any, *, torch: Any) -> Any:
    pil = _patch_to_pil(image_patch).resize((224, 224))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    tensor = torch.from_numpy(arr).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _resolve_torch_device(torch: Any, device: str) -> Any:
    if str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict):
                payload = value
                break
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a state_dict or contain state_dict/model_state_dict/model.")
    cleaned = {}
    for key, value in payload.items():
        text = str(key)
        if text.startswith("module."):
            text = text[len("module.") :]
        cleaned[text] = value
    return cleaned


def _hf_output_vector(outputs: Any) -> np.ndarray:
    tensor = getattr(outputs, "pooler_output", None)
    if tensor is None:
        tensor = getattr(outputs, "last_hidden_state", None)
        if tensor is not None:
            tensor = tensor.mean(dim=1)
    if tensor is None:
        tensor = outputs[0]
        if getattr(tensor, "ndim", 0) == 3:
            tensor = tensor.mean(dim=1)
    return tensor.detach().cpu().numpy().reshape(-1).astype(float)


def _torch_output_vector(output: Any) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.detach().cpu().numpy().reshape(-1).astype(float)


def _named_embedding_features(vector: np.ndarray, *, output_dim: int) -> dict[str, float]:
    values = np.asarray(vector, dtype=float).reshape(-1)[: int(output_dim)]
    norm = float(np.linalg.norm(values))
    normalized = values / norm if norm > 0 else values
    features = {f"embedding_dim_{index:03d}": float(value) for index, value in enumerate(normalized)}
    features["embedding_norm"] = norm
    return features


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def _normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    clipped = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {label: 1.0 / len(BMNET_LABELS) for label in BMNET_LABELS}
    return {label: float(clipped.get(label, 0.0) / total) for label in BMNET_LABELS}


def _json_ready_config(config: BMNetMorphologyPilotConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
