# Nature Methods submission readiness checklist

- Target format: Nature Methods Brief Communication.
- Abstract limit: <=70 words; see `qa/text_limit_qa.tsv`.
- Main text package limit: <=1,200 words including abstract, references and figure legends; see `qa/text_limit_qa.tsv`.
- Main text structure: continuous main text without Results/Discussion-style subheadings; Online Methods retains subheadings.
- Display items: 2 main figures plus one fallback figure for editorial contingency.
- Manuscript DOCX: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\topolink_cci_short_communication\nature_short_communication_20260516\manuscript\topolink_cci_short_communication_submission_initial.docx`.
- Manuscript PDF: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\topolink_cci_short_communication\nature_short_communication_20260516\manuscript\topolink_cci_short_communication_submission_initial.pdf`.
- Font QA: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\topolink_cci_short_communication\nature_short_communication_20260516\qa\font_qa_report.tsv`.
- Figure format QA: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\topolink_cci_short_communication\nature_short_communication_20260516\qa\figure_format_qa.tsv`.
- Figure fonts: Matplotlib uses `pdf.fonttype=42` and `svg.fonttype=none`; verify with `pdffonts`.
- Figure raster exports: PNG/TIFF generated at 600 dpi; Nature minimum is 300 dpi.
- Figure source data: every panel has a TSV entry in `source_data/` and figure manifests in `metadata/`.
- Official checks consulted: Nature Methods content types, Nature initial submission guide and Nature figure specifications.
