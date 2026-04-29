# BM-Net/H&E morphology increment pilot on PDC

This scaffold runs a breast H&E morphology increment pilot for the aligned
Xenium RNA + H&E contour workflow. It is intentionally separate from the
published tutorial path so existing `run_contour_boundary_ecology_pilot`
behavior stays unchanged.

Default dataset:

```text
/cfs/klemming/scratch/h/hutaobo/topolink_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs
```

Default BM-Net pilot root:

```text
/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04
```

## Backends

- `deterministic-smoke`: dependency-free BM-Net-like H&E proxy for validating
  contour cropping, schema, artifacts, Slurm, and downstream increment tests.
  This is not biological evidence.
- `bmnet-local`: loads a local checkpoint through a MobileNetV3-small BM-Net-like
  head. Use this only when a compatible BM-Net checkpoint is available.
- `bmnet-like-trainable`: MobileNetV3-small + classifier head for local
  training/fine-tuning experiments.
- `hf-pathology-backbone`: uses a Hugging Face pathology backbone such as
  `1aurent/vit_small_patch8_224.lunit_dino` or `wisdomik/QuiltNet-B-32` as a
  surrogate feature extractor and writes `pathology__...` features rather than
  BM-Net probabilities.

## PDC workflow

From the staged repo on Dardel:

```bash
export PDC_ROOT=/cfs/klemming/scratch/h/hutaobo/pyxenium_bmnet_morphology_2026-04
export PDC_XENIUM_ROOT=/cfs/klemming/scratch/h/hutaobo/topolink_cci_benchmark_2026-04/data/source_cache/breast/WTA_Preview_FFPE_Breast_Cancer_outs

bash benchmarking/bmnet_pdc/scripts/bootstrap_pdc_env.sh
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh --backend deterministic-smoke --include-full
```

For a real BM-Net checkpoint:

```bash
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh \
  --backend bmnet-local \
  --checkpoint /cfs/klemming/scratch/h/hutaobo/models/bmnet/bmnet.pt \
  --include-full
```

For the Hugging Face surrogate backbone discovered during setup:

```bash
bash benchmarking/bmnet_pdc/scripts/submit_pdc_bmnet_pilot.sh \
  --backend hf-pathology-backbone \
  --hf-model 1aurent/vit_small_patch8_224.lunit_dino \
  --smoke-max-contours 20
```

The smoke job limits the run to 50 contours by default. The full job is
submitted with an `afterok` dependency when `--include-full` is used.

## Outputs

Each run directory writes:

- `contour_features_with_bmnet.csv`
- `bmnet_patch_predictions.csv`
- `program_scores.csv`
- `xenium_native_morphology.csv`
- `he_morphology_features.csv`
- `feature_redundancy.csv`
- `incremental_prediction.csv`
- `partial_associations.csv`
- `matched_review_table.csv`
- `morphology_increment_summary.json`
- `bmnet_pdc_run_summary.json`

`morphology_increment_summary.json` includes `model_metadata` so downstream
reports can distinguish trained BM-Net, Hugging Face surrogate, and smoke-only
outputs.
