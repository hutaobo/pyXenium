# Contour-aware density profiling

`pyXenium.contour` adds contour-native analysis on top of `XeniumSData`.

## Import contour annotations

```python
from pyXenium.contour import add_contours_from_geojson, expand_contours

add_contours_from_geojson(
    sdata,
    "/path/to/polygon_units.geojson",
    key="protein_cluster_contours",
)

expand_contours(
    sdata,
    contour_key="protein_cluster_contours",
    distance=25.0,
)
```

## Expand contours

Use ordinary overlap-preserving expansion when you want a simple buffered layer:

```python
from pyXenium.contour import expand_contours

expand_contours(
    sdata,
    contour_key="protein_cluster_contours",
    distance=25.0,
    mode="overlap",
    output_key="protein_cluster_contours_expanded",
)
```

Use Voronoi expansion when neighboring contour supports must stay mutually exclusive:

```python
from pyXenium.contour import expand_contours

expand_contours(
    sdata,
    contour_key="protein_cluster_contours",
    distance=25.0,
    mode="voronoi",
    output_key="protein_cluster_contours_voronoi",
    voronoi_sample_step=2.0,
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
