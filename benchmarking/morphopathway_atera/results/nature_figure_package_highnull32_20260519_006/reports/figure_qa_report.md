# Figure QA Report

- Status: pass
- Backend: R for plotting, preview, export and visual QA.
- H&E source extraction: numeric low-resolution source-data tables from raw OME-TIF; plotting/export in R.
- Stable pathway core: 9 pathways.
- High-null seeds: 3.
- Cross-cancer recovery range: 9-10/10.
- Axis-masked recovery range: 9-10/10.
- SVG editable text check: pass.
- H&E crop provenance recorded: pass.
- H&E scale calibration: pass.
- H&E display adjustment: R-only per-channel 2-98 percentile contrast stretch for H&E display; raw numeric source pixels unchanged.
- Image integrity: see `source_data/he_image_integrity_manifest.csv`.
- Local render resource log: see `logs/figure_render.resource.log`.
- Evidence boundary: no old `naturebiotech_package` outputs were used as evidence input.
