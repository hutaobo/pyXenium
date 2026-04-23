# Contour Boundary Ecology Discovery Package

Sample ID: `WTA_Preview_FFPE_Breast_Cancer_outs`
Contour layer: `atera_boundary_ecology_contours`

## Sample Summary

- Contours analysed: `28`
- Cells in table: `170057`
- Ecotypes discovered: `4`
- Bootstrap ecotype stability (ARI): `0.662`
- Embedding backend enabled: `False`

## Boundary Programs

- `immune_exclusion`: top program for `10` contours
- `stromal_encapsulation`: top program for `6` contours
- `emt_invasive_front`: top program for `3` contours
- `tls_adjacent_activation`: top program for `3` contours
- `necrotic_hypoxic_rim`: top program for `3` contours
- `myeloid_vascular_belt`: top program for `3` contours

## Ranked Hypotheses

- `stromal_encapsulation`: Stromal matrix features wrap the contour and may form a physical barrier; top evidence: CXCL12, COL4A1, VIM.
- `emt_invasive_front`: The boundary shows an EMT- and stroma-shifted invasive front relative to matched controls; top evidence: spp1_cd44__outer_minus_inner, cxcl12_cxcr4__outer_minus_inner, tgfb1_tgfbr2__outer_minus_inner.
- `tls_adjacent_activation`: Lymphoid activation is enriched near the outer rim, consistent with TLS-adjacent immune organization; top evidence: VEGFA, KDR, TAGLN.
- `myeloid_vascular_belt`: Myeloid and vascular programs co-accumulate along the contour edge, consistent with a perivascular suppressive belt; top evidence: TAGLN, SLC2A1, VEGFA.
- `necrotic_hypoxic_rim`: Hypoxia-like rim features and reduced cellularity suggest a necrotic or stressed boundary ecology; top evidence: CD3E, CXCR5, ACTA2.
- `immune_exclusion`: Outer-rim immune signals dominate the boundary while the inner rim stays comparatively immune-cold; top evidence: spp1_cd44__outer_minus_inner, immune_activation, myeloid_activation.

## Matched Exemplars

- `immune_exclusion` exemplar `S5 Endothelial Cells #4.1` vs control `S5 Endothelial Cells #2.1` (delta `0.431`)
- `immune_exclusion` exemplar `S1 11q13 Invasive Tumor Cells #2.1` vs control `S2 Basal-like Structured DCIS Cells #1.1` (delta `1.008`)
- `immune_exclusion` exemplar `S6 Apocrine Cells #2.1` vs control `S6 Apocrine Cells #4.1` (delta `1.094`)
- `myeloid_vascular_belt` exemplar `S7 Luminal-like Amorphous DCIS Cells #1.1` vs control `S4 Plasma Cells #1.1` (delta `0.478`)
- `myeloid_vascular_belt` exemplar `S3 Macrophages #3.1` vs control `S1 11q13 Invasive Tumor Cells #2.1` (delta `0.644`)
- `myeloid_vascular_belt` exemplar `S2 Basal-like Structured DCIS Cells #3.1` vs control `S1 11q13 Invasive Tumor Cells #3.1` (delta `0.501`)

## Top Differential Signals

- `emt_invasive_front` / `marker_pair` / `spp1_cd44__outer_minus_inner`: effect `1.099`, matched delta `1.099`, FDR `0.928`
- `emt_invasive_front` / `marker_pair` / `cxcl12_cxcr4__outer_minus_inner`: effect `0.934`, matched delta `0.934`, FDR `1`
- `emt_invasive_front` / `marker_pair` / `tgfb1_tgfbr2__outer_minus_inner`: effect `0.522`, matched delta `0.522`, FDR `1`
- `emt_invasive_front` / `gene` / `VIM`: effect `0.241`, matched delta `0.241`, FDR `1`
- `emt_invasive_front` / `marker_pair` / `vegfa_kdr__outer_minus_inner`: effect `0.162`, matched delta `0.162`, FDR `1`
- `emt_invasive_front` / `gene` / `CD3E`: effect `0.134`, matched delta `0.134`, FDR `1`
- `emt_invasive_front` / `gene` / `CXCR4`: effect `0.134`, matched delta `0.134`, FDR `1`
- `emt_invasive_front` / `gene` / `LDHA`: effect `0.119`, matched delta `0.119`, FDR `1`
- `emt_invasive_front` / `gene` / `COL4A1`: effect `0.117`, matched delta `0.117`, FDR `1`
- `emt_invasive_front` / `gene` / `CD3D`: effect `0.100`, matched delta `0.100`, FDR `1`
- `emt_invasive_front` / `gene` / `TRAC`: effect `0.100`, matched delta `0.100`, FDR `1`
- `emt_invasive_front` / `gene` / `PRF1`: effect `0.100`, matched delta `0.100`, FDR `1`
