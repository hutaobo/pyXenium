17 May 2026

Dear Nature Biotechnology Editors,

Re: Submission of "Foundation-model morphology maps residual programs in spatial-omics contours" as a Brief Communication

Please consider this manuscript for publication as a Brief Communication in Nature Biotechnology.

This submission is prepared for double-anonymous peer review. The reviewer-facing manuscript, Online Methods and Supplementary Information have been anonymized; author identity, affiliation, acknowledgements, funding and code-repository identifiers are provided below for editorial use.

The timing of the study is important. Atera WTA is a newly introduced whole-transcriptome in situ platform for FFPE tissue that moves spatial profiling from targeted panels toward more than 18,000 genes in tissue context. The public breast and cervical cancer preview datasets used here profile 18,028 genes and, to our knowledge at submission, are the publicly available Atera WTA preview examples for epithelial cancer. This creates a timely analysis problem for the journal's readership: how should co-registered H&E foundation-model morphology be interrogated when spatial omics already defines transcriptome-scale molecular contours?

The manuscript introduces morphomolecular translation mapping (mTM), a contour-constrained framework for asking whether H&E foundation-model embeddings map residual WTA program variation after spatial-omics-derived contour labels and spatial covariates have already been fixed. This is not an H&E-to-expression leaderboard. The primary question is whether morphology carries measurable molecular information within already defined WTA contours.

In breast Atera WTA, mTM nominates residual luminal estrogen-response, unfolded-protein-response and oxidative-phosphorylation continua inside S3 contours. In cervical Atera WTA, the same residual-decoding workflow serves as a second epithelial cancer stress test and nominates stromal-remodeling, immune-ecology and invasion-associated program families rather than a direct replication of the breast luminal phenotype. The reported associations are tested against compartment-aware spatial permutations that preserve contour labels, centroid-position bins and boundary-distance bins; 13 candidate associations pass the 95% spatial-null gate.

The study is deliberately conservative about what the data support. The contours were generated from Atera WTA cell-coordinate and cluster information rather than from H&E segmentation, so the manuscript does not claim independent histological contour discovery, morphology-only deployment, causality, directional boundary mechanisms or protein-level validation. We believe the Brief Communication format is appropriate because the manuscript makes one focused point: foundation-model morphology can add residual information within spatial-omics-defined contours rather than merely reproduce those contours.

Editor-only information:

Author and correspondence: Taobo Hu; Science for Life Laboratory, Department of Biochemistry and Biophysics, Stockholm University, Stockholm, Sweden; taobo.hu@scilifelab.se.

Funding and acknowledgements: computations were enabled by resources provided by the National Academic Infrastructure for Supercomputing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2022-06725.

Code availability for editors: to support editorial assessment while preserving double-anonymous review, code identifiers are provided here rather than in reviewer-facing files. The mTM analysis and packaging code is available in the pyXenium repository at https://github.com/hutaobo/pyXenium, with fixed release v0.4.6 at https://github.com/hutaobo/pyXenium/releases/tag/v0.4.6 and submission-packaging hardening commit 3c4617010bb1c76e18921958c4507f91512aa9dc at https://github.com/hutaobo/pyXenium/commit/3c4617010bb1c76e18921958c4507f91512aa9dc. The exact journal-upload package and individual submission files are archived at https://github.com/hutaobo/pyXenium/releases/tag/nbt-submission-upload-20260518-draft. HistoSeg contour-generation software is maintained at https://github.com/hutaobo/HistoSeg. These identifiers are for editorial use and are omitted from reviewer-facing files to preserve double-anonymous review.

There are no related manuscripts by the author under consideration or in press elsewhere. I have not had prior discussions with a Nature Biotechnology editor about the work described in this manuscript. I confirm that this manuscript has not been published elsewhere and is not under consideration by another journal. The author has approved the manuscript and agrees with its submission to Nature Biotechnology. The author declares no competing interests. Taobo Hu conceived the study, implemented the analysis workflow, performed the computational analyses, generated the figures and source data, and wrote the manuscript. The study used public/vendor example spatial WTA and H&E data and did not involve new human sample collection, intervention or generation of new identifiable participant data.

Sincerely,

Taobo Hu
