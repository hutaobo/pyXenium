from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from shapely import wkt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact RTD summaries for a direct LazySlide run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wsi-path", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace("&", "and")
        .replace("+", "plus")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def _export_tables(run_dir: Path) -> None:
    for stem in [
        "tile_features",
        "tile_assignments",
        "structure_image_features",
        "structure_differential_features",
        "structure_rna_summary",
        "structure_program_scores",
        "program_image_associations",
        "rna_image_associations",
        "image_contours",
    ]:
        path = run_dir / f"{stem}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if stem in {"tile_features", "image_contours"}:
            frame.head(500).to_csv(run_dir / f"{stem}.csv", index=False)
        else:
            frame.to_csv(run_dir / f"{stem}.csv", index=False)


def _write_tile_summary(run_dir: Path, tile: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    summary_cols = [
        "tile_id",
        "tissue_id",
        "tile_x",
        "tile_y",
        "assigned",
        "contour_id",
        "assigned_structure",
        "structure_id",
        "classification_name",
        "top_image_label",
        "top_image_label_score",
        *text_cols,
    ]
    summary = tile[[column for column in summary_cols if column in tile.columns]].copy()
    summary.to_parquet(run_dir / "tile_feature_summary.parquet", index=False)
    summary.to_csv(run_dir / "tile_feature_summary.csv", index=False)
    return summary


def _write_structure_text_summary(
    run_dir: Path,
    tile: pd.DataFrame,
    structure: pd.DataFrame,
    terms: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in structure.iterrows():
        scores = {
            term: float(row.get(f"text_similarity__{_slug(term)}__mean", np.nan))
            for term in terms
        }
        valid = {key: value for key, value in scores.items() if not math.isnan(value)}
        top = max(valid, key=valid.get) if valid else None
        mode = None
        if "top_image_label" in tile.columns:
            subset = tile[tile["assigned_structure"].astype(str) == str(row["assigned_structure"])]
            if not subset.empty:
                modes = subset["top_image_label"].mode(dropna=True)
                mode = str(modes.iloc[0]) if not modes.empty else None
        out: dict[str, object] = {
            "assigned_structure": row["assigned_structure"],
            "structure_id": row.get("structure_id"),
            "n_tiles": int(row["n_tiles"]),
            "n_contours": int(row["n_contours"]),
            "top_mean_text_label": top,
            "top_mean_text_score": valid.get(top) if top else np.nan,
            "top_tile_label_mode": mode,
        }
        for term, value in scores.items():
            out[f"mean_text_sim_{_slug(term)}"] = value
        rows.append(out)
    summary = pd.DataFrame(rows)
    summary.to_parquet(run_dir / "structure_text_summary.parquet", index=False)
    summary.to_csv(run_dir / "structure_text_summary.csv", index=False)
    return summary


def _plot_embedding(run_dir: Path, tile: pd.DataFrame) -> None:
    embed_cols = [column for column in tile.columns if column.startswith("embedding__")]
    matrix = tile[embed_cols].to_numpy(dtype=np.float32)
    try:
        import umap

        coords = umap.UMAP(
            n_components=2,
            random_state=17,
            n_neighbors=30,
            min_dist=0.15,
        ).fit_transform(matrix)
        method = "UMAP"
    except Exception:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=17).fit_transform(matrix)
        method = "PCA"
    projection = pd.DataFrame(
        {
            "tile_id": tile["tile_id"].astype(str).to_numpy(),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "assigned_structure": tile["assigned_structure"].astype(str).to_numpy(),
            "top_image_label": tile.get("top_image_label", pd.Series([None] * len(tile)))
            .astype(str)
            .to_numpy(),
        }
    )
    projection.to_csv(run_dir / "tile_embedding_umap.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=180)
    for name, group in projection.groupby("assigned_structure", sort=True):
        ax.scatter(group["x"], group["y"], s=8, alpha=0.7, label=name)
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title("Direct LazySlide PLIP tile embeddings")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "tile_embedding_umap.png", bbox_inches="tight")
    plt.close(fig)


def _plot_structure_heatmap(run_dir: Path, summary: pd.DataFrame, terms: list[str]) -> None:
    heat = summary.set_index("assigned_structure")[
        [f"mean_text_sim_{_slug(term)}" for term in terms]
    ]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=180)
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(terms)))
    ax.set_xticklabels(terms, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)
    ax.set_title("Structure-level PLIP text similarity")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(run_dir / "structure_text_similarity_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def _plot_spatial_map(run_dir: Path, tile: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=180)
    for name, group in tile.groupby("assigned_structure", sort=True):
        ax.scatter(group["tile_x"], group["tile_y"], s=7, alpha=0.7, label=str(name))
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("H&E x pixel")
    ax.set_ylabel("H&E y pixel")
    ax.set_title("Direct LazySlide tiles assigned to HistoSeg structures")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "spatial_tile_map.png", bbox_inches="tight")
    plt.close(fig)


def _write_montage(run_dir: Path, tile: pd.DataFrame, wsi_path: Path) -> None:
    try:
        import tiffslide
    except ImportError:
        return
    slide = tiffslide.TiffSlide(str(wsi_path))
    reps: list[tuple[str, Image.Image]] = []
    for structure, group in tile.groupby("assigned_structure", sort=True):
        if "top_image_label_score" in group.columns:
            row = group.sort_values("top_image_label_score", ascending=False).iloc[0]
        else:
            row = group.iloc[0]
        geom = wkt.loads(row["geometry_wkt"])
        minx, miny, maxx, maxy = geom.bounds
        pad = 32
        x0 = max(int(minx) - pad, 0)
        y0 = max(int(miny) - pad, 0)
        width = max(int(maxx - minx) + 2 * pad, 64)
        height = max(int(maxy - miny) + 2 * pad, 64)
        image = slide.read_region((x0, y0), 0, (width, height)).convert("RGB").resize((180, 180))
        label = f"{structure}\n{row.get('top_image_label', '')}"
        reps.append((label, image))
    if not reps:
        return
    cols = min(5, len(reps))
    rows = int(math.ceil(len(reps) / cols))
    tile_w, tile_h = 220, 230
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(reps):
        x = (index % cols) * tile_w
        y = (index // cols) * tile_h
        canvas.paste(image, (x + 20, y + 10))
        draw.text((x + 10, y + 195), label, fill=(0, 0, 0))
    canvas.save(run_dir / "representative_tile_montage.png")


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser()
    wsi_path = Path(args.wsi_path).expanduser()
    if args.output_dir is not None:
        raise NotImplementedError("Use --run-dir as output directory for now.")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    terms = list(manifest["config"]["text_terms"])
    text_cols = [f"text_similarity__{_slug(term)}" for term in terms]

    _export_tables(run_dir)
    tile = pd.read_parquet(run_dir / "tile_features.parquet")
    structure = pd.read_parquet(run_dir / "structure_image_features.parquet")
    _write_tile_summary(run_dir, tile, text_cols)
    text_summary = _write_structure_text_summary(run_dir, tile, structure, terms)
    _plot_embedding(run_dir, tile)
    _plot_structure_heatmap(run_dir, text_summary, terms)
    _plot_spatial_map(run_dir, tile)
    _write_montage(run_dir, tile, wsi_path)

    wsi_manifest = wsi_path.with_suffix(wsi_path.suffix + ".manifest.json")
    if wsi_manifest.exists():
        (run_dir / "prepared_wsi_manifest.json").write_text(
            wsi_manifest.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(text_summary.to_string(index=False))


if __name__ == "__main__":
    main()
