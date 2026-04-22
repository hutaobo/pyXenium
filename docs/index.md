# pyXenium

pyXenium now has five canonical public surfaces:

- `pyXenium.io`: Xenium artifact loading, partial export recovery, SData I/O, and SpatialData-compatible export.
- `pyXenium.multimodal`: joint RNA + Protein preparation, analysis, immune-resistance scoring, and packaged workflows.
- `pyXenium.ligand_receptor`: topology-native ligand-receptor analysis.
- `pyXenium.pathway`: pathway topology analysis and pathway activity scoring.
- `pyXenium.contour`: contour GeoJSON import and contour-aware density profiling for cells or transcripts.

`pyXenium.analysis` remains importable only as a compatibility facade and is no longer the
primary implementation home for public APIs.
