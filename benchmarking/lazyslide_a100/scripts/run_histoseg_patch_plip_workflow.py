from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


DEFAULT_TEXT_TERMS = (
    "ductal epithelium",
    "invasive carcinoma",
    "in situ carcinoma",
    "fibrotic stroma",
    "immune infiltrate",
    "necrosis",
    "adipose tissue",
    "vascular stroma",
    "lumen or secretion",
)


def _repo_root() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "pyXenium").exists():
            return candidate
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PLIP on HistoSeg contour patches and aggregate by structure.",
    )
    parser.add_argument("--manifest", required=True, help="HistoSeg contour patch manifest JSON.")
    parser.add_argument("--output-dir", required=True, help="Output artifact directory.")
    parser.add_argument("--model-name", default=os.environ.get("LAZYSLIDE_MODEL_PATH", "vinid/plip"))
    parser.add_argument("--model-label", default="plip")
    parser.add_argument("--device", default=os.environ.get("LAZYSLIDE_DEVICE", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-patches", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--text-terms",
        nargs="*",
        default=list(DEFAULT_TEXT_TERMS),
        help="Zero-shot text labels for PLIP text-image similarity.",
    )
    parser.add_argument(
        "--table-format",
        choices=("csv", "parquet"),
        default="parquet",
    )
    return parser.parse_args()


def main() -> None:
    repo = _repo_root()
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    from transformers import CLIPModel, CLIPProcessor

    args = _parse_args()
    started = time.time()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_manifest(Path(args.manifest), max_patches=args.max_patches)
    if not records:
        raise ValueError("No patch records were found in the manifest.")

    device = _resolve_device(args.device, torch)
    processor = _load_processor(CLIPProcessor, args.model_name, token=args.hf_token)
    model = _load_model(CLIPModel, args.model_name, token=args.hf_token).to(device)
    model.eval()

    text_terms = tuple(args.text_terms or DEFAULT_TEXT_TERMS)
    text_features = _encode_text(
        model=model,
        processor=processor,
        terms=text_terms,
        device=device,
    )
    tile_features = _encode_images(
        records=records,
        model=model,
        processor=processor,
        text_features=text_features,
        text_terms=text_terms,
        device=device,
        batch_size=int(args.batch_size),
        amp=bool(args.amp),
    )

    structure_features = _aggregate_structure_image_features(
        tile_features,
        text_terms=text_terms,
    )
    differential = _differential_image_features(tile_features)
    structure_text = _structure_text_summary(tile_features, text_terms)
    embedding_plot = _write_embedding_projection(tile_features, output_dir)
    heatmap = _write_structure_heatmap(structure_text, output_dir)
    spatial_map = _write_spatial_map(tile_features, output_dir)
    montage = _write_representative_montage(tile_features, output_dir)

    paths = {
        "tile_features": _write_table(
            tile_features,
            output_dir / f"tile_features.{args.table_format}",
            table_format=args.table_format,
        ),
        "tile_feature_summary": _write_table(
            _tile_feature_summary(tile_features, text_terms),
            output_dir / f"tile_feature_summary.{args.table_format}",
            table_format=args.table_format,
        ),
        "structure_image_features": _write_table(
            structure_features,
            output_dir / f"structure_image_features.{args.table_format}",
            table_format=args.table_format,
        ),
        "structure_text_summary": _write_table(
            structure_text,
            output_dir / f"structure_text_summary.{args.table_format}",
            table_format=args.table_format,
        ),
        "structure_differential_features": _write_table(
            differential,
            output_dir / f"structure_differential_features.{args.table_format}",
            table_format=args.table_format,
        ),
        "embedding_projection": embedding_plot["table"],
        "tile_embedding_umap": embedding_plot["figure"],
        "structure_heatmap": heatmap,
        "spatial_tile_map": spatial_map,
        "representative_tile_montage": montage,
    }
    summary = _build_summary(
        args=args,
        tile_features=tile_features,
        structure_text=structure_text,
        paths=paths,
        started=started,
        device=device,
        torch_module=torch,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote PLIP patch artifacts to: {output_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Patches: {summary['outputs']['n_tiles']}; structures: {summary['outputs']['n_structures']}")


def _read_manifest(path: Path, *, max_patches: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("patches") or payload.get("records") or payload.get("items") or []
    if not isinstance(payload, list):
        raise ValueError("Patch manifest must be a JSON list or contain a `patches` list.")
    records = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        image_path = item.get("image_path") or (item.get("patch") or {}).get("path")
        if not image_path or not Path(str(image_path)).exists():
            continue
        record = dict(item)
        record["tile_id"] = index
        record["image_path"] = str(image_path)
        record["contour_id"] = str(record.get("contour_id", index))
        record["assigned_structure"] = str(
            record.get("structure_label")
            or record.get("structure_name")
            or record.get("assigned_structure")
            or f"structure_{record.get('structure_id', 'unknown')}"
        )
        if max_patches is not None and len(records) >= int(max_patches):
            break
        records.append(record)
    return records


def _load_model(cls: Any, model_name: str, *, token: str | None) -> Any:
    try:
        return cls.from_pretrained(model_name, token=token)
    except TypeError:
        return cls.from_pretrained(model_name, use_auth_token=token)


def _load_processor(cls: Any, model_name: str, *, token: str | None) -> Any:
    try:
        return cls.from_pretrained(model_name, token=token)
    except TypeError:
        return cls.from_pretrained(model_name, use_auth_token=token)


def _resolve_device(device: str, torch_module: Any) -> str:
    if device.startswith("cuda") and not torch_module.cuda.is_available():
        return "cpu"
    return device


def _encode_text(*, model: Any, processor: Any, terms: Sequence[str], device: str) -> Any:
    import torch

    inputs = processor(
        text=list(terms),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = _feature_tensor(model.get_text_features(**inputs), preferred="text")
        return torch.nn.functional.normalize(features, p=2, dim=-1)


def _encode_images(
    *,
    records: Sequence[dict[str, Any]],
    model: Any,
    processor: Any,
    text_features: Any,
    text_terms: Sequence[str],
    device: str,
    batch_size: int,
    amp: bool,
) -> pd.DataFrame:
    import torch

    rows: list[dict[str, Any]] = []
    batch_size = max(int(batch_size), 1)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images = [_load_patch(record["image_path"]) for record in batch]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if amp and str(device).startswith("cuda")
            else torch.autocast("cpu", enabled=False)
        )
        with torch.inference_mode(), context:
            image_features = _feature_tensor(model.get_image_features(**inputs), preferred="image")
            image_norm = torch.nn.functional.normalize(image_features.float(), p=2, dim=-1)
            sims = image_norm @ text_features.T
        image_np = image_features.float().cpu().numpy()
        sims_np = sims.float().cpu().numpy()
        for offset, record in enumerate(batch):
            row = _base_patch_row(record)
            for idx, value in enumerate(image_np[offset]):
                row[f"embedding_{idx:03d}"] = float(value)
            for idx, term in enumerate(text_terms):
                row[f"text_sim_{_slug(term)}"] = float(sims_np[offset, idx])
            best_idx = int(np.nanargmax(sims_np[offset]))
            row["top_image_label"] = str(text_terms[best_idx])
            row["top_image_score"] = float(sims_np[offset, best_idx])
            row.update(_rgb_summary(images[offset]))
            rows.append(row)
        print(f"encoded {min(start + batch_size, len(records))}/{len(records)} patches", flush=True)
    return pd.DataFrame(rows)


def _feature_tensor(output: Any, *, preferred: str) -> Any:
    if hasattr(output, "shape"):
        return output
    candidate_names = (
        ("text_embeds", "pooler_output", "last_hidden_state")
        if preferred == "text"
        else ("image_embeds", "pooler_output", "last_hidden_state")
    )
    for name in candidate_names:
        value = getattr(output, name, None)
        if value is None:
            continue
        if name == "last_hidden_state" and hasattr(value, "ndim") and value.ndim >= 3:
            return value[:, 0, :]
        return value
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    raise TypeError(f"Could not extract tensor features from {type(output)!r}.")


def _load_patch(path: str) -> Image.Image:
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def _base_patch_row(record: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    patch = record.get("patch") or {}
    bbox = record.get("bbox") or {}
    x0 = _float_or_nan(bbox.get("x0"))
    y0 = _float_or_nan(bbox.get("y0"))
    x1 = _float_or_nan(bbox.get("x1"))
    y1 = _float_or_nan(bbox.get("y1"))
    return {
        "tile_id": int(record["tile_id"]),
        "contour_id": str(record.get("contour_id")),
        "structure_id": record.get("structure_id"),
        "assigned_structure": str(record.get("assigned_structure")),
        "structure_label": str(record.get("structure_label") or record.get("assigned_structure")),
        "image_path": str(record.get("image_path")),
        "tile_x": float((x0 + x1) / 2.0) if not math.isnan(x0 + x1) else np.nan,
        "tile_y": float((y0 + y1) / 2.0) if not math.isnan(y0 + y1) else np.nan,
        "bbox_x0": x0,
        "bbox_y0": y0,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "patch_width": patch.get("saved_width") or patch.get("original_width"),
        "patch_height": patch.get("saved_height") or patch.get("original_height"),
        "nonzero_fraction": patch.get("nonzero_fraction"),
        "pyramid_level": record.get("pyramid_level") or patch.get("pyramid_level"),
    }


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _rgb_summary(image: Image.Image) -> dict[str, float]:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.size == 0:
        return {}
    flat = arr.reshape(-1, 3)
    gray = flat.mean(axis=1)
    return {
        "rgb_mean_r": float(flat[:, 0].mean()),
        "rgb_mean_g": float(flat[:, 1].mean()),
        "rgb_mean_b": float(flat[:, 2].mean()),
        "rgb_std_r": float(flat[:, 0].std()),
        "rgb_std_g": float(flat[:, 1].std()),
        "rgb_std_b": float(flat[:, 2].std()),
        "rgb_brightness_mean": float(gray.mean()),
        "rgb_brightness_std": float(gray.std()),
        "rgb_tissue_fraction": float(np.mean(gray < 0.92)),
    }


def _tile_feature_summary(tile_features: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    keep = [
        "tile_id",
        "contour_id",
        "structure_id",
        "assigned_structure",
        "image_path",
        "tile_x",
        "tile_y",
        "patch_width",
        "patch_height",
        "nonzero_fraction",
        "top_image_label",
        "top_image_score",
    ]
    keep.extend(f"text_sim_{_slug(term)}" for term in terms)
    keep.extend(
        [
            "rgb_mean_r",
            "rgb_mean_g",
            "rgb_mean_b",
            "rgb_brightness_mean",
            "rgb_tissue_fraction",
        ]
    )
    return tile_features.loc[:, [column for column in keep if column in tile_features.columns]].copy()


def _structure_text_summary(tile_features: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    rows = []
    text_columns = [f"text_sim_{_slug(term)}" for term in terms]
    for structure, group in tile_features.groupby("assigned_structure", dropna=True):
        row: dict[str, Any] = {
            "assigned_structure": structure,
            "n_tiles": int(len(group)),
            "top_image_label_mode": _mode(group["top_image_label"]) if "top_image_label" in group else None,
        }
        for term, column in zip(terms, text_columns):
            if column in group:
                row[f"mean_{column}"] = float(pd.to_numeric(group[column], errors="coerce").mean())
                row[f"frac_top_{_slug(term)}"] = float((group["top_image_label"] == term).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        mean_columns = [column for column in frame.columns if column.startswith("mean_text_sim_")]
        if mean_columns:
            best = frame.loc[:, mean_columns].idxmax(axis=1)
            frame["top_mean_text_label"] = best.str.replace("mean_text_sim_", "", regex=False).str.replace("_", " ")
            frame["top_mean_text_score"] = frame.loc[:, mean_columns].max(axis=1)
    return frame


def _aggregate_structure_image_features(
    tile_features: pd.DataFrame,
    *,
    text_terms: Sequence[str],
) -> pd.DataFrame:
    if tile_features.empty or "assigned_structure" not in tile_features.columns:
        return pd.DataFrame(columns=["assigned_structure", "n_tiles"])
    numeric_columns = _image_numeric_columns(tile_features)
    rows = []
    text_columns = [f"text_sim_{_slug(term)}" for term in text_terms]
    for structure, group in tile_features.groupby("assigned_structure", dropna=True):
        row: dict[str, Any] = {
            "assigned_structure": str(structure),
            "n_tiles": int(len(group)),
            "n_contours": int(group["contour_id"].nunique()) if "contour_id" in group else int(len(group)),
        }
        if "top_image_label" in group:
            row["top_image_label_mode"] = _mode(group["top_image_label"])
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=0))
        for column in text_columns:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_std"] = float(values.std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("assigned_structure").reset_index(drop=True)


def _differential_image_features(tile_features: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    columns = [
        "assigned_structure",
        "feature_name",
        "feature_kind",
        "effect_size",
        "mean_in_structure",
        "mean_rest",
        "p_value",
        "fdr",
        "n_structure_tiles",
        "n_rest_tiles",
    ]
    if tile_features.empty or "assigned_structure" not in tile_features.columns:
        return pd.DataFrame(columns=columns)
    numeric_columns = _image_numeric_columns(tile_features)
    rows = []
    assigned = tile_features[tile_features["assigned_structure"].notna()].copy()
    for structure in sorted(assigned["assigned_structure"].astype(str).unique()):
        in_group = assigned[assigned["assigned_structure"].astype(str) == structure]
        rest = assigned[assigned["assigned_structure"].astype(str) != structure]
        if len(in_group) < 2 or len(rest) < 2:
            continue
        for column in numeric_columns:
            left = pd.to_numeric(in_group[column], errors="coerce").dropna()
            right = pd.to_numeric(rest[column], errors="coerce").dropna()
            if len(left) < 2 or len(right) < 2:
                continue
            mean_left = float(left.mean())
            mean_right = float(right.mean())
            pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
            effect = (mean_left - mean_right) / pooled if pooled > 0 else 0.0
            try:
                p_value = float(stats.ttest_ind(left, right, equal_var=False, nan_policy="omit").pvalue)
            except Exception:
                p_value = 1.0
            rows.append(
                {
                    "assigned_structure": structure,
                    "feature_name": column,
                    "feature_kind": _feature_kind(column),
                    "effect_size": float(effect),
                    "mean_in_structure": mean_left,
                    "mean_rest": mean_right,
                    "p_value": p_value,
                    "n_structure_tiles": int(len(left)),
                    "n_rest_tiles": int(len(right)),
                }
            )
    frame = pd.DataFrame(rows, columns=[column for column in columns if column != "fdr"])
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["fdr"] = _benjamini_hochberg(frame["p_value"].to_numpy(dtype=float))
    return frame.loc[:, columns].sort_values(["fdr", "p_value", "assigned_structure"]).reset_index(drop=True)


def _image_numeric_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("embedding_", "text_sim_", "rgb_")
    allowed = []
    for column in frame.columns:
        if not column.startswith(prefixes):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            allowed.append(column)
    for column in ("top_image_score", "nonzero_fraction", "patch_width", "patch_height"):
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            allowed.append(column)
    return sorted(set(allowed))


def _feature_kind(column: str) -> str:
    if column.startswith("text_sim_"):
        return "text_similarity"
    if column.startswith("embedding_"):
        return "embedding"
    if column.startswith("rgb_"):
        return "color_morphology"
    return "numeric"


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _mode(values: pd.Series) -> str | None:
    counts = Counter(str(value) for value in values.dropna())
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _write_embedding_projection(tile_features: pd.DataFrame, output_dir: Path) -> dict[str, str | None]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    embedding_columns = [column for column in tile_features.columns if column.startswith("embedding_")]
    if len(embedding_columns) < 2 or tile_features.empty:
        return {"table": None, "figure": None}
    x = tile_features.loc[:, embedding_columns].to_numpy(dtype=float)
    method = "pca"
    try:
        import umap

        coords = umap.UMAP(n_components=2, random_state=7, metric="cosine").fit_transform(x)
        method = "umap"
    except Exception:
        coords = PCA(n_components=2, random_state=7).fit_transform(x)
    projection = tile_features.loc[:, ["tile_id", "assigned_structure", "top_image_label"]].copy()
    projection[f"{method}_1"] = coords[:, 0]
    projection[f"{method}_2"] = coords[:, 1]
    table_path = output_dir / f"tile_embedding_{method}.csv"
    projection.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    labels = sorted(projection["assigned_structure"].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    for idx, label in enumerate(labels):
        mask = projection["assigned_structure"].astype(str) == label
        ax.scatter(
            projection.loc[mask, f"{method}_1"],
            projection.loc[mask, f"{method}_2"],
            s=10,
            alpha=0.78,
            color=cmap(idx % 20),
            label=label,
        )
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title("PLIP contour-patch embeddings by HistoSeg structure")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    figure_path = output_dir / f"tile_embedding_{method}.png"
    fig.savefig(figure_path)
    plt.close(fig)
    return {"table": str(table_path), "figure": str(figure_path)}


def _write_structure_heatmap(structure_text: pd.DataFrame, output_dir: Path) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [column for column in structure_text.columns if column.startswith("mean_text_sim_")]
    if structure_text.empty or not columns:
        return None
    matrix = structure_text.set_index("assigned_structure").loc[:, columns].copy()
    matrix.columns = [column.replace("mean_text_sim_", "").replace("_", " ") for column in matrix.columns]
    fig_width = max(8, 0.8 * len(matrix.columns))
    fig_height = max(4, 0.35 * len(matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=160)
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title("Mean PLIP text-image similarity by HistoSeg structure")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    path = output_dir / "structure_text_similarity_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_spatial_map(tile_features: pd.DataFrame, output_dir: Path) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not {"tile_x", "tile_y", "assigned_structure"}.issubset(tile_features.columns):
        return None
    plot = tile_features.dropna(subset=["tile_x", "tile_y"]).copy()
    if plot.empty:
        return None
    labels = sorted(plot["assigned_structure"].astype(str).unique())
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    for idx, label in enumerate(labels):
        mask = plot["assigned_structure"].astype(str) == label
        ax.scatter(
            plot.loc[mask, "tile_x"],
            plot.loc[mask, "tile_y"],
            s=8,
            alpha=0.75,
            color=cmap(idx % 20),
            label=label,
        )
    ax.invert_yaxis()
    ax.set_xlabel("Xenium x")
    ax.set_ylabel("Xenium y")
    ax.set_title("HistoSeg contour patches used for PLIP feature extraction")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    path = output_dir / "spatial_tile_map.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _write_representative_montage(tile_features: pd.DataFrame, output_dir: Path) -> str | None:
    if tile_features.empty or "image_path" not in tile_features:
        return None
    selected = []
    for structure, group in tile_features.groupby("assigned_structure"):
        ordered = group.sort_values("top_image_score", ascending=False).head(3)
        for _, row in ordered.iterrows():
            selected.append((str(structure), row))
    if not selected:
        return None

    thumb_w, thumb_h = 150, 150
    label_h = 34
    cols = 3
    rows = math.ceil(len(selected) / cols)
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (structure, row) in enumerate(selected):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        try:
            img = _load_patch(str(row["image_path"]))
        except Exception:
            continue
        img.thumbnail((thumb_w, thumb_h))
        canvas.paste(img, (x + (thumb_w - img.width) // 2, y))
        text = f"{structure[:28]}\n{str(row.get('top_image_label', ''))[:28]}"
        draw.text((x + 3, y + thumb_h + 2), text, fill=(20, 20, 20))
    path = output_dir / "representative_tile_montage.png"
    canvas.save(path)
    return str(path)


def _write_table(frame: pd.DataFrame, path: Path, *, table_format: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if table_format == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    csv_path = path.with_suffix(".csv")
    if csv_path != path:
        compact = frame
        if len(compact) > 500:
            compact = compact.head(500)
        compact.to_csv(csv_path, index=False)
    return str(path)


def _build_summary(
    *,
    args: argparse.Namespace,
    tile_features: pd.DataFrame,
    structure_text: pd.DataFrame,
    paths: dict[str, str | None],
    started: float,
    device: str,
    torch_module: Any,
) -> dict[str, Any]:
    top_by_structure = []
    if not structure_text.empty:
        columns = [
            "assigned_structure",
            "n_tiles",
            "top_mean_text_label",
            "top_mean_text_score",
            "top_image_label_mode",
        ]
        top_by_structure = structure_text.loc[
            :,
            [column for column in columns if column in structure_text.columns],
        ].to_dict("records")
    return {
        "workflow": "histoseg_patch_plip_structure_features",
        "status": "completed",
        "started_at_unix": float(started),
        "runtime_seconds": float(time.time() - started),
        "model": {
            "model_label": args.model_label,
            "model_name": args.model_name,
            "backend": "transformers.CLIPModel",
            "method_note": (
                "Patch fallback used HistoSeg contour patches because the full WSI LazySlide "
                "tiling path was killed by memory pressure on a non-pyramidal 17GB OME-TIFF."
            ),
        },
        "inputs": {
            "manifest": str(Path(args.manifest).resolve()),
            "max_patches": args.max_patches,
            "text_terms": list(args.text_terms),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
            "torch": getattr(torch_module, "__version__", None),
            "cuda_available": bool(torch_module.cuda.is_available()),
            "gpu_name": torch_module.cuda.get_device_name(0) if torch_module.cuda.is_available() else None,
            "git_commit": _git_commit(),
        },
        "outputs": {
            "n_tiles": int(len(tile_features)),
            "n_structures": int(tile_features["assigned_structure"].nunique())
            if "assigned_structure" in tile_features
            else 0,
            "n_embedding_features": int(
                sum(column.startswith("embedding_") for column in tile_features.columns)
            ),
            "paths": paths,
            "top_by_structure": top_by_structure,
        },
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text).lower()).strip("_")


if __name__ == "__main__":
    main()
