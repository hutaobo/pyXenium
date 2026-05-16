# Atera Morphopathway Brief Communication Runbook

This directory contains the new `pyXenium.pathway` H&E+WTA morphopathway suite outputs for the Atera Xenium WTA preview datasets.

## Scope

- Breast discovery dataset: `Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs`
- Cervical validation dataset: `Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Cervical_Cancer_outs`
- H&E backend used for the final package: `transformers_clip:vinid/plip`
- Final spatial pseudobulk setting: `--spatial-block-bins 12`
- Final high-null setting: `--permutations 32 --negative-controls 32`

The old `naturebiotech_package` outputs are not evidence inputs for this work.

## Main Commands

Run one Atera PLIP smoke bundle:

```powershell
$env:PYTHONPATH='D:\GitHub\pyXenium\src'
python benchmarking\morphopathway_atera\scripts\run_atera_morphopathway_smoke.py `
  --output-dir benchmarking\morphopathway_atera\results\<run_name> `
  --feature-set plip `
  --aggregation spatial-block `
  --spatial-block-bins 12 `
  --min-cells-per-block 6 `
  --max-cells 3000 `
  --permutations 32 `
  --negative-controls 32 `
  --clip-model-name vinid/plip `
  --clip-model-label plip `
  --clip-output-dim 64 `
  --clip-device cpu `
  --random-state 17
```

Create an evidence pack for a run:

```powershell
python benchmarking\morphopathway_atera\scripts\make_morphopathway_evidence_pack.py benchmarking\morphopathway_atera\results\<run_name>
```

Create the final Brief Communication package:

```powershell
python benchmarking\morphopathway_atera\scripts\make_brief_communication_package.py <run17> <run29> <run43> `
  --stability-dir <stability_dir> `
  --sensitivity-dir <sensitivity_dir> `
  --output-dir benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
```

Generate the reviewer evidence audit, validate the final package, and archive it:

```powershell
python benchmarking\morphopathway_atera\scripts\make_reviewer_evidence_audit.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
python benchmarking\morphopathway_atera\scripts\validate_brief_communication_package.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049 --require-reviewer-audit
python benchmarking\morphopathway_atera\scripts\archive_brief_communication_package.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
```

## Final Package

Final package directory:

```text
benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
```

Archive:

```text
benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049.zip
```

Latest archive SHA256:

```text
db731299997fbdbadab9d4eccba981dfe49330e457bdc907a93b6b3d0d1acf54
```

## Conservative Claim Boundary

Supported:

- Stable 9-pathway pathway-family stress-test core across three high-null seeds.
- Cross-cancer recovery range: 9/10 to 10/10.
- Axis-masked recovery range: 9/10 to 10/10.

Not supported:

- Direct pathway-level cervical replication of the top breast signal.
- Clinical biomarker performance.
- Patient-level generalization.
- Causal morphology-pathway mechanism claims.

## Verification

Focused tests:

```powershell
$env:PYTHONPATH='D:\GitHub\pyXenium\src'
pytest tests\test_morphopathway.py tests\test_topology_analysis.py -q
```

Package QC:

```powershell
python benchmarking\morphopathway_atera\scripts\make_reviewer_evidence_audit.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049
python benchmarking\morphopathway_atera\scripts\validate_brief_communication_package.py benchmarking\morphopathway_atera\results\brief_communication_package_highnull32_20260512_2049 --require-reviewer-audit
```

Current QC status is pass. Remaining expected warnings:

- Breast matched negative-control pass95 minimum is 8/10.
- Cervical matched negative-control pass95 minimum is 9/10.
