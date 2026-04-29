# AI-Driven Spatial Pathologist via spatho

## Overview

[AI-Driven Spatial Pathologist](https://ai-driven-spatial-pathologist.readthedocs.io/en/latest/?badge=latest)
is an external workflow layer for AI-assisted pathology review around Xenium-scale spatial
transcriptomics. The public Python package and CLI are named `spatho`.

pyXenium treats this as an eighth feature area in the documentation, not as a new
`pyXenium.spatho` namespace. The goal is to show where pyXenium hands a structured
Xenium case to `spatho`, while keeping the AI workflow implementation in the
AI-Driven Spatial Pathologist project.

## Relationship to XeniumSData

The `spatho` tutorial workflow is enabled by the same data model pyXenium uses for
Xenium I/O and downstream analysis. `XeniumSData` keeps the core case components
together:

- a cell table in `XeniumSData.table`
- transcript points in `XeniumSData.points` or streaming `point_sources`
- cell and nucleus boundaries in `XeniumSData.shapes`
- H&E image metadata in `XeniumSData.images`
- SpatialData-compatible conversion through `XeniumSData.to_spatialdata()`

This gives AI-Driven Spatial Pathologist a consistent Xenium foundation without
duplicating pyXenium readers or storing a large spatho wrapper inside pyXenium.

## Minimal setup

Install `spatho` separately from pyXenium:

```bash
pip install -U spatho
```

Create a starter workflow JSON for a Xenium case:

```bash
spatho init-workflow \
  --organ breast \
  --case-name breast_case_01 \
  --dataset-root /path/to/Xenium_outs \
  --base-pipeline-config /path/to/project/configs/breast_case_01.json \
  --output /path/to/workflows/breast_case_01.json
```

Check the workflow before running it:

```bash
spatho doctor --config /path/to/workflows/breast_case_01.json
```

Run the workflow:

```bash
spatho run --config /path/to/workflows/breast_case_01.json
```

## Backend choices

`spatho` can run pathology review through several paths:

- `openai`: uses `OPENAI_API_KEY` for managed OpenAI API calls.
- `pathology_ai_api`: calls a local or PDC-hosted `pathology-ai` HTTP service.
- heuristic-only mode: runs deterministic smoke checks without AI review calls.

For the full operational tutorial, including local deployment and the Atera WTA breast
PDC run, use the upstream documentation:

- [AI-Driven Spatial Pathologist docs](https://ai-driven-spatial-pathologist.readthedocs.io/en/latest/?badge=latest)
- [Atera WTA Breast Cancer on PDC](https://ai-driven-spatial-pathologist.readthedocs.io/en/latest/ATERA_WTA_BREAST_PDC_TUTORIAL.html)
- [Local Deployment](https://ai-driven-spatial-pathologist.readthedocs.io/en/latest/local_deployment.html)

## pyXenium boundary

pyXenium provides the Xenium data foundation and the reusable `XeniumSData`
structure. `spatho` owns the AI-driven spatial pathology workflow, organ packs,
pathology review prompts, and optional local `pathology-ai` backend.

This boundary keeps pyXenium lightweight: no new runtime dependency, no vendored
AI workflow code, and no extra pyXenium CLI wrapper are required.
