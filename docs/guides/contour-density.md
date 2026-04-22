# Contour-aware density profiling

`pyXenium.contour` adds a contour-native workflow on top of `XeniumSData`.

It covers two steps:

1. Import polygon or multipolygon annotations from GeoJSON.
2. Quantify cells or transcripts as signed-distance profiles around the contour boundary.

## Import contour annotations

Use `add_contours_from_geojson(...)` to normalize GeoJSON features into
`XeniumSData.shapes[...]`.

```python
from pyXenium.contour import add_contours_from_geojson

add_contours_from_geojson(
    sdata,
    "/path/to/polygon_units.geojson",
    key="protein_cluster_contours",
)
```

The importer:

- supports `Polygon` and `MultiPolygon`
- expands commonly used metadata fields such as `assigned_structure`
- converts Xenium pixel coordinates into microns when needed
- stores the original per-contour properties under `sdata.metadata["contours"][key]`

## Ring-based density

Use `ring_density(...)` when you want a discrete inward/outward summary.

```python
from pyXenium.contour import ring_density

ring_df = ring_density(
    sdata,
    contour_key="protein_cluster_contours",
    target="transcripts",
    contour_query='assigned_structure == "Structure 4"',
    feature_values="VIM",
    inward=100.0,
    outward=100.0,
    ring_width=50.0,
)
```

Signed-distance semantics are:

- boundary = `0`
- inward = negative
- outward = positive

The output includes one row per contour and ring with:

- `count`
- `area`
- `density`
- contour metadata such as `assigned_structure`

For `target="cells"`, pyXenium uses cell centroid coordinates.
For `target="transcripts"`, pyXenium uses transcript point coordinates and optional
`feature_key` / `feature_values` filtering.

## Smooth signed-distance density

Use `smooth_density_by_distance(...)` when you want a continuous profile instead of bins.

```python
from pyXenium.contour import smooth_density_by_distance

smooth_df = smooth_density_by_distance(
    sdata,
    contour_key="protein_cluster_contours",
    target="transcripts",
    contour_query='assigned_structure == "Structure 4"',
    feature_values="VIM",
    inward=100.0,
    outward=100.0,
    bandwidth=25.0,
)
```

The smooth estimator:

- uses Gaussian kernel smoothing on signed distance
- keeps an area-normalized density interpretation
- applies reflection correction near the analysis boundaries

The output includes:

- `signed_distance`
- `count_density`
- `geometry_measure`
- `density`

This makes it easier to compare contour-adjacent enrichment or depletion patterns without
committing to a single ring width.
