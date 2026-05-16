# NBT scientific strengthening plan, 2026-05-16

Scope: strengthen the mTM/Atera WTA Brief Communication from a defensible initial submission into a stronger Nature Biotechnology-level manuscript or revision package.

## Strategic diagnosis

The current manuscript is format-ready and internally consistent, but its scientific strength is still limited by four evidence gaps:

1. No orthogonal biological validation of the top transcriptomic claims.
2. Limited cohort/sample generality: one breast Atera WTA discovery sample plus one cervical cross-cancer stress test.
3. Spatial dependence is defended by stratified permutation and block bootstrap, but not yet by stronger spatial sensitivity analyses.
4. Processed data and exact review artifacts are in GitHub/source-data form, but not yet deposited as a DOI-backed processed-data archive.

The strongest path is therefore not to broaden the claim. It is to make the existing residual-decoding claim harder to dismiss.

## Priority 1: processed-data archive and reproducibility hardening

Goal: make the manuscript auditable without requiring reviewer access to A100/PDC or raw vendor files.

Actions:

- Create a DOI-backed Zenodo/Figshare archive containing:
  - final one-figure upload package;
  - all source-data CSVs;
  - processed contour-level tables needed to recompute reported partial correlations;
  - figure-generation inputs;
  - run manifests and provenance summaries with private paths removed;
  - exact `pyXenium` and `HistoSeg` commits/tags.
- Add a small `recompute_main_numbers.py` script that reads only deposited processed tables and reproduces:
  - breast S3 luminal estrogen response rho `-0.639`;
  - UPR rho `0.515`;
  - OXPHOS rho `0.531`;
  - permutation empirical P `9.999e-05`;
  - Supplementary Tables 1-4.
- Add a machine-readable reproducibility checklist:
  - inputs;
  - output files;
  - expected hashes;
  - expected row counts;
  - expected numeric tolerances.

Expected lift: high credibility gain, low biological risk, fast to complete.

Decision gate: complete before any resubmission or revision. This is the safest improvement.

## Priority 2: stronger spatial sensitivity analysis

Goal: reduce the easiest statistical criticism: residual morphology-WTA associations may still reflect unmodeled local spatial autocorrelation or registration artifacts.

Status, 2026-05-16: first-pass PLIP spatial sensitivity checks completed on A100 for the locked breast top-three associations and cervical cross-cancer associations. Compact outputs are tracked in `spatial_sensitivity_A100_summary_20260516.csv` and `spatial_sensitivity_A100_summary_20260516.md`. The current result supports sign stability under leave-one-spatial-block-out, local mismatch controls and centroid-covariate jitter, but does not yet cover UNI or full registration perturbation.

Actions:

- Add leave-one-spatial-block-out analysis:
  - partition contours by spatial blocks or tissue neighborhoods;
  - recompute top associations after holding out each block;
  - report effect stability and sign stability.
- Add distance-matched negative controls:
  - for each contour, pair with spatially proximate but non-identical contours under the same contour label;
  - test whether H&E-WTA pairing beats local mismatched pairing.
- Add morphology-only negative controls:
  - random embedding dimensions;
  - shuffled tile-to-contour assignments within structure;
  - low-information morphology features if available.
- Add registration sensitivity:
  - jitter contour boundaries or tile assignments within a small pixel radius;
  - recompute the top three breast S3 associations.
- Add multiple-testing accounting:
  - report tested program-feature search space;
  - show FDR or max-statistic/permutation family-wise control for the top claims.

Expected lift: high statistical credibility. This directly answers likely reviewer attack lines without requiring new samples.

Decision gate: if the top three breast S3 results remain stable in sign and approximate magnitude, promote this from Supplementary Information into main text wording.

## Priority 3: orthogonal biological validation without new wet lab

Goal: strengthen biological interpretation using information already available or quickly obtainable.

Actions:

- Gene-level sanity checks within Atera WTA:
  - luminal/ER axis: `ESR1`, `PGR`, `GATA3`, `KRT8`, `KRT18`, `KRT19`;
  - stress/UPR axis: `HSPA5`, `XBP1`, `ATF4`, `DDIT3`, `HERPUD1`;
  - oxidative phosphorylation: mitochondrial/respiratory-chain genes and curated OXPHOS gene sets.
- Show that mTM-ranked contours have coherent gene-level shifts, not only aggregate program-score shifts.
- Add contour-neighborhood enrichment:
  - compare top and bottom mTM quantiles for cell-type composition, tumor/stroma/immune markers and boundary distance.
- Add pathologist-facing morphology review:
  - export blinded high/low patch panels for top S3 luminal ER and UPR/OXPHOS axes;
  - collect simple scores: tumor cellularity, glandular/luminal morphology, stromal content, necrosis/stress-like morphology, immune infiltrate;
  - report concordance descriptively, not as a diagnostic claim.

Expected lift: medium-to-high. This is the fastest way to make the biology feel real without claiming protein validation.

Decision gate: if gene-level and blinded morphology annotations align, add one supplementary validation table and one concise main-text sentence.

## Priority 4: true orthogonal validation if sample access exists

Goal: convert the paper from a computational discovery note into a stronger biological validation story.

Actions if adjacent tissue or comparable samples can be accessed:

- Breast validation:
  - ER, PR, HER2 and Ki-67 IHC on adjacent or comparable breast section;
  - quantify IHC signal over registered or approximately matched contours;
  - test whether the luminal estrogen-response mTM axis tracks ER/PR protein positivity.
- Stress/metabolic validation:
  - HSPA5/BiP or XBP1/CHOP for UPR;
  - OXPHOS marker panel if feasible.
- Boundary/MAZ validation:
  - collagen/ECM markers, alpha-SMA/FAP for CAF, CD3/CD20/CD68 for immune ecology depending on the selected boundary program.

Expected lift: very high, but dependent on sample access and turnaround time.

Decision gate: do this only if tissue/source access is realistic. Do not delay initial submission indefinitely for validation that cannot be obtained.

## Priority 5: independent cohort or public-data replication

Goal: show that mTM is not tuned to one Atera WTA breast sample.

Candidate data routes:

- Search current public 10x Atera WTA examples beyond breast/cervical as they appear.
- Use Xenium Prime or Xenium panel datasets as lower-plex replication of the workflow, clearly labeled as panel-based rather than WTA.
- Use Visium/HEST-style spatial transcriptomics plus H&E as a coarser-resolution stress test, if contour geometry can be defined reproducibly.

Minimum acceptable replication:

- one additional breast or epithelial cancer sample;
- same contour-constrained residual model;
- same negative controls;
- at least program-family-level concordance, not necessarily the exact breast luminal phenotype.

Expected lift: high for editor confidence. Risk is scope creep and incompatibility between technologies.

Decision gate: do not mix technologies in the main claim unless the result is clean. If using panel/Visium data, frame it as an external stress test.

## Priority 6: manuscript upgrade path

If Priorities 1-3 are completed:

- Keep as Brief Communication.
- Add a compact supplementary validation/sensitivity section.
- Replace one sentence in the main text with stronger evidence wording:
  - from "nominates" to "nominates and validates at the gene-program/sensitivity level";
  - do not claim clinical deployment or protein validation.

If Priorities 1-5 are completed with strong results:

- Consider expanding beyond Brief Communication into a fuller Article/Resource-style submission.
- Add a multi-panel Extended Data package:
  - reproducibility workflow;
  - spatial sensitivity;
  - independent dataset;
  - orthogonal biology;
  - deposited processed-data archive.

## Immediate next work order

1. Completed: build processed-data archive contents locally.
2. Completed: write `recompute_main_numbers.py`.
3. Completed for PLIP first pass: run leave-one-spatial-block-out, local mismatch and centroid-jitter analyses on A100 outputs.
4. Next: run the same spatial-sensitivity checks for UNI where the required contour tables and candidate rows are available.
5. Next: generate blinded patch-review panels for pathologist/morphology review.
6. Next: search and inventory any additional usable public 10x Atera/Xenium datasets.
7. Update manuscript/SI only after the new results pass numeric and claim-audit checks.

## Official-policy anchors

- NBT preparing submission guidance: https://www.nature.com/nbt/submission-guidelines/preparing-your-submission
- NBT content types: https://www.nature.com/nbt/content
- Nature Portfolio reporting standards and data/code/protocol availability: https://www.nature.com/nmeth/editorial-policies/reporting-standards
- 10x Atera public example datasets are vendor/public inputs and should remain cited as source pages, not redistributed raw data.
