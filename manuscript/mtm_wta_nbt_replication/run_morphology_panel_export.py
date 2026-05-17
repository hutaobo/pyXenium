#!/usr/bin/env python3
"""Export blinded morphology-review panels from existing hero patches."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


REVIEW_COLUMNS = [
    "blind_id",
    "tumor_cellularity_score_0_3",
    "glandular_luminal_morphology_score_0_3",
    "stromal_content_score_0_3",
    "necrosis_or_stress_like_score_0_3",
    "immune_infiltrate_score_0_3",
    "free_text_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hero-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--programs",
        nargs="+",
        default=[
            "program__wta_luminal_estrogen_response",
            "program__wta_unfolded_protein_response",
            "program__wta_oxidative_phosphorylation",
        ],
    )
    parser.add_argument("--panel-width", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    image_dir = out_dir / "blinded_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.hero_manifest.expanduser().resolve())
    selected = manifest.loc[manifest["target_feature"].isin(args.programs)].copy()
    if selected.empty:
        raise SystemExit("No requested program patches found in hero manifest.")
    selected = assign_polarity(selected)
    rng = random.Random(args.seed)
    order = list(selected.index)
    rng.shuffle(order)
    selected = selected.loc[order, :].reset_index(drop=True)

    key_rows: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for idx, row in selected.iterrows():
        blind_id = f"NBT-MORPH-{idx + 1:03d}"
        src = Path(str(row["patch_path"]))
        suffix = src.suffix.lower() if src.suffix else ".png"
        dest_name = f"{blind_id}{suffix}"
        dest = image_dir / dest_name
        if src.exists():
            shutil.copy2(src, dest)
            status = "copied"
        else:
            status = "missing_patch_source"
        key_rows.append(
            {
                "blind_id": blind_id,
                "blinded_image": f"blinded_images/{dest_name}",
                "copy_status": status,
                "program": row["target_feature"],
                "polarity": row["polarity"],
                "contour_id": row["contour_id"],
                "assigned_structure": row.get("assigned_structure", ""),
                "hidden_program_score": row.get("hidden_program_score", ""),
                "target_z_within_structure": row.get("target_z_within_structure", ""),
                "image_z_within_structure": row.get("image_z_within_structure", ""),
                "source_patch_path": row["patch_path"],
            }
        )
        review_rows.append({column: "" for column in REVIEW_COLUMNS} | {"blind_id": blind_id})

    key = pd.DataFrame(key_rows)
    review = pd.DataFrame(review_rows)
    key.to_csv(out_dir / "morphology_panel_blinding_key.csv", index=False)
    review.to_csv(out_dir / "morphology_panel_review_form.csv", index=False)
    write_report(out_dir / "morphology_panel_export_report.md", key)
    try_write_contact_sheet(image_dir, key, out_dir / "morphology_panel_contact_sheet.png", args.panel_width)
    print(f"Wrote blinded morphology panel to {out_dir}")
    print(key.groupby(["program", "polarity", "copy_status"]).size().to_string())
    return 0


def assign_polarity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["polarity"] = ""
    for _, group in out.groupby("target_feature", sort=False):
        values = pd.to_numeric(group["image_z_within_structure"], errors="coerce")
        median = float(values.median())
        out.loc[group.index, "polarity"] = [
            "high_oriented_axis" if float(value) >= median else "low_oriented_axis"
            for value in values
        ]
    return out


def write_report(path: Path, key: pd.DataFrame) -> None:
    lines = [
        "# Blinded Morphology Panel Export",
        "",
        "This package contains blinded H&E patch images and a scoring form for descriptive morphology review. It is intended as biological plausibility support, not as independent diagnostic or protein validation.",
        "",
        "## Contents",
        "",
        f"- Blinded images: {int(key['copy_status'].eq('copied').sum())}",
        f"- Missing image sources: {int(key['copy_status'].ne('copied').sum())}",
        "- Review form: `morphology_panel_review_form.csv`",
        "- Blinding key: `morphology_panel_blinding_key.csv`",
        "",
        "## Program Counts",
        "",
    ]
    counts = key.groupby(["program", "polarity"]).size().reset_index(name="n")
    for _, row in counts.iterrows():
        lines.append(f"- {row['program']} / {row['polarity']}: {int(row['n'])} patches.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_write_contact_sheet(image_dir: Path, key: pd.DataFrame, out_path: Path, panel_width: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return
    copied = key.loc[key["copy_status"].eq("copied"), :].copy()
    if copied.empty:
        return
    thumbs = []
    for _, row in copied.iterrows():
        path = image_dir / Path(str(row["blinded_image"])).name
        image = Image.open(path).convert("RGB")
        image.thumbnail((360, 360))
        canvas = Image.new("RGB", (380, 420), "white")
        canvas.paste(ImageOps.expand(image, border=1, fill=(180, 180, 180)), (10, 10))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 382), str(row["blind_id"]), fill=(0, 0, 0))
        thumbs.append(canvas)
    width = max(1, panel_width)
    rows = (len(thumbs) + width - 1) // width
    sheet = Image.new("RGB", (width * 380, rows * 420), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % width) * 380, (idx // width) * 420))
    sheet.save(out_path)


if __name__ == "__main__":
    raise SystemExit(main())
