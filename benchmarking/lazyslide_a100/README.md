# HistoSeg + LazySlide A100 workflow

This runner executes the optional `pyXenium.multimodal`
HistoSeg-anchored LazySlide workflow on the Atera breast WTA Xenium export.

The package boundary is intentional:

- HistoSeg provides structure/contour GeoJSON.
- LazySlide provides WSI tiling and PLIP/CONCH image features.
- pyXenium aligns those artifacts to Xenium RNA and writes structure-level
  image/RNA association tables.

## Environment

```bash
export PYXENIUM_REPO=/path/to/pyXenium
export A100_ENV_DIR=/path/to/envs/pyxenium-lazyslide
bash benchmarking/lazyslide_a100/scripts/bootstrap_a100_env.sh
```

## Smoke run

```bash
export PYXENIUM_ATERA_DATASET=/path/to/WTA_Preview_FFPE_Breast_Cancer_outs
export HISTOSEG_GEOJSON=/path/to/xenium_explorer_annotations.s1_s5.generated.geojson
export A100_OUTPUT_DIR=/path/to/runs/histoseg_lazyslide_smoke
export LAZYSLIDE_MODEL=plip

bash benchmarking/lazyslide_a100/scripts/run_a100_histoseg_lazyslide.sh \
  --max-tiles 2000
```

For CONCH, set `LAZYSLIDE_MODEL=conch` and provide the required model token or
cache in the A100 environment. If CONCH cannot be loaded, the pyXenium workflow
records the skipped reason instead of fabricating feature tables.
