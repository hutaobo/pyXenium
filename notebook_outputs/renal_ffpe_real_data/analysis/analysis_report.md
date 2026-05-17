# Renal FFPE Result Analysis

## Overview
### protein_cluster_contour
- Polygons: 1121
- Structures: 5
- Median cells per polygon: 3.0
- Non-zero-cell polygons: 1042
- Saved H&E patches: 306
- Oversized patches skipped: 12
- Empty-bbox patches: 803

### gene_cluster_contour
- Polygons: 778
- Structures: 3
- Median cells per polygon: 1.0
- Non-zero-cell polygons: 727
- Saved H&E patches: 263
- Oversized patches skipped: 7
- Empty-bbox patches: 508

## Top Structures
### gene_cluster_contour
- Structure 3: 294 polygons, 423572 total cells
- Structure 1: 270 polygons, 136973 total cells
- Structure 2: 214 polygons, 426859 total cells

### protein_cluster_contour
- Structure 1: 395 polygons, 345379 total cells
- Structure 3: 379 polygons, 462171 total cells
- Structure 2: 214 polygons, 94787 total cells
- Structure 5: 110 polygons, 123913 total cells
- Structure 4: 23 polygons, 35395 total cells

## Top Markers
- protein_cluster_contour rna: VIM (20555652), HLA-DRA (7927879), FCGR3A (1816673), HLA-DRB1 (1669862), CXCL6 (1345494)
- protein_cluster_contour protein: Vimentin (257033), CD45 (149399), PTEN (124433), CD3E (90129), CD68 (80573)
- gene_cluster_contour rna: VIM (18022808), HLA-DRA (7013534), HLA-DRB1 (1812325), FCGR3A (1649843), PTPRC (828939)
- gene_cluster_contour protein: Vimentin (185710), CD45 (84207), PTEN (79768), CD3E (65332), CD68 (31178)

## Notes
- H&E patch extraction completed for a subset of polygons; very large polygons were skipped intentionally to avoid giant patch files.
- `morphology_focus` patch extraction is still missing because the current TIFF loader returns `ValueError` on that image.
- The high number of `empty_bbox` records suggests many contour bboxes do not overlap the current H&E pixel frame after clipping, which is worth checking as a spatial alignment follow-up.
