# NBT readiness audit, 2026-05-16

Scope: one-figure initial-submission package for `Foundation-model morphology maps residual programs in spatial-omics contours`.

Official guidance checked:

- Nature Biotechnology content types: https://www.nature.com/nbt/content
- Nature Biotechnology submission guidelines: https://www.nature.com/nbt/submission-guidelines
- Initial formatting guidance: https://www.nature.com/nbt/submission-guidelines/initial-formatting

## Current package status

- Brief Communication format now matches the hard NBT constraints checked here: title 8 words/76 characters, abstract 3 sentences/63 words, 20 references, and 1,473 words for abstract plus main body plus references plus figure legend.
- Reference list uses compact NBT style: article titles are omitted from the publication-facing manuscript, while complete DOI/URL evidence remains in `Reference_Audit_20260516.md`.
- Cover letter now includes the required no-related-manuscripts and no-prior-editorial-discussion disclosures.
- Online Methods now includes a large-language-model assistance disclosure, with final scientific responsibility assigned to the author.
- Code availability now distinguishes the software release (`v0.4.6`) from the exact submission-package archive (`nbt-initial-submission-20260516`).
- Upload package has no A100/PDC runtime paths or internal provenance notes.
- Source-data mapping is one-figure consistent and no stale `Figure_2`, `Figure_3`, `Figure_4`, `Supplementary_BlockBootstrap` or `Supplementary_Spatial_Permutation` source-data labels are exposed in the upload package.
- Supplementary Tables 1-4 match the source-data CSVs under the declared rounding conventions, including the breast PLIP invasion/boundary/EMT `not_detected` row.
- DOCX files are structurally valid OOXML. Full visual render QA could not be performed on this machine because `soffice`/LibreOffice is not installed.

## Remaining high-impact journal risks

1. Orthogonal validation is still the largest biological risk. The manuscript deliberately does not include IHC, protein-level validation or an independent molecular assay, so reviewer resistance should be expected if the claim is read biologically rather than methodologically.
2. Cohort generality remains limited. Breast and cervical Atera WTA examples are useful stress tests, but they are not a multi-patient or multi-center validation cohort.
3. The contour construction is intentionally conservative but not independent of WTA. The current framing is correct: mTM tests residual within-contour WTA program variation, not independent H&E discovery of tissue contours.
4. Spatial dependence is mitigated but not exhausted. The compartment-aware permutation and block bootstrap defend against coarse structure and position effects, but they do not fully model fine-scale spatial random fields or registration uncertainty.
5. The study is not a broad benchmark against H&E-to-expression models. The cover letter and main text now state this clearly; expanding into a benchmark would require new experiments and would change the paper's scope.
6. Raw input data are not redistributed. For acceptance-level data availability, a DOI-backed archive of processed contour-level tables and exact source-data CSVs would be stronger than GitHub-only availability.
7. Figure aesthetics still need human visual inspection in Word/PDF after DOCX rendering on a machine with LibreOffice or Microsoft Word. Automated OOXML checks passed, but page-level overflow cannot be ruled out locally.

## Recommended next round

For initial submission, the package is now defensible. For a stronger NBT revision path, prioritize: a deposited processed-data archive with DOI, independent cohort or sample-level replication if available, one orthogonal validation panel for the top luminal ER/stress claim, and a stricter spatial sensitivity analysis.
