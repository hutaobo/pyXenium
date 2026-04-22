# Contour-aware density profiling

`pyXenium.contour` adds contour-native analysis on top of `XeniumSData`.

## Import contour annotations

```python
from pyXenium.contour import add_contours_from_geojson

add_contours_from_geojson(
    sdata,
    "/path/to/polygon_units.geojson",
    key="protein_cluster_contours",
)
```

## Ring-based density

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

## Smooth signed-distance density

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
