# Xenium RNA+Protein+H&E alignment note

This fixture bundle treats the polygon as the minimal pathology unit that can link Xenium RNA, Xenium protein, and H&E image context.

## Why this exists

- Internal analysis stays in `physical_um` so RNA points, same-cell protein summaries, and polygon geometry are expressed in one canonical space.
- Export to Xenium Explorer happens only at the boundary, through an explicit `um -> pixel` transform using the dataset pixel size.
- The same transform chain is what lets a polygon be re-used as an H&E patch boundary for downstream pathology-AI handoff.

## Included cases

- `identity_points`: Transcript centroids remain unchanged inside the canonical micron space.
- `scale_um_to_pixel`: Convert transcript, protein ROI, and polygon coordinates from micron space to Xenium Explorer pixels.
- `translation_origin_shift`: Account for origin shifts introduced by H&E subregion extraction or ROI remapping.
- `axis_order_xy_yx`: Make x/y versus row/column conventions explicit for image-linked modalities.
- `compose_scale_then_translate`: Compose a micron-to-pixel scaling step with a patch-local translation for polygon review bundles.

## Migration boundary to spatialdata

- Keep the object taxonomy portable: `points`, `images`, `labels`, `shapes`, and `transforms`.
- Keep the Xenium-specific loader logic and artifact discovery in `pyXenium`.
- Migrate the manifest and transform layer later through the `bundle_to_spatialdata_payload(...)` adapter rather than rewriting the current analysis APIs first.

## Bundle metadata

- `pixel_size_um`: 0.2125
- `pixel_size_source`: default_constant
- `segmentation_source`: ranger_default

## Example polygon unit

- `polygon_id`: cell_boundaries_aaaacegh-1
- `cell_ids`: aaaacegh-1
- `top_rna`: {}
- `top_protein`: {}

## External contour sources

This workflow also uses externally generated Xenium Explorer contours as part of the real-data review loop.
- `protein_cluster_contour`
- `gene_cluster_contour`
