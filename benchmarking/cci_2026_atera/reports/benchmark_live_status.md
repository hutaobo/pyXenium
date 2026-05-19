# TopoLink-CCI Benchmark Live Status

Last updated: 2026-05-19 18:39 Europe/Berlin.

## Current Summary

The main breast full-common comparison is usable. Breast full/authoritative methods include TopoLink-CCI, CellPhoneDB, LARIS, LIANA+, SpatialDM, stLearn, Squidpy, CellChat LR-only, and COMMOT.

The original final closeout source-of-truth was refreshed at `2026-05-13T04:40:40.460301+00:00`. The reporting denominator has now been upgraded to an expanded 18-method benchmark so every mentioned method is counted: 9 full successes, 8 bounded successes, 1 reproducible failure card, 0 deferred candidate methods, 0 pending/running methods, and `all_methods_accounted=true`.

## Expanded A100 Retry Status

- `FastCCC` A100 retry succeeded after PDC failure: `20k_smoke_a100`, `1,319,600` standardized rows.
- FastCCC output: `D:\GitHub\pyXenium\benchmarking\cci_2026_atera\results\expanded_methods_a100_20260514\a100_collected\expanded_methods_a100_20260514\fastccc\smoke20k\standardized.tsv.gz`.
- `Copulacci` A100 real-method retry succeeded with official `copulacci.model2.run_scc`: `20k_bounded`, `13,981` standardized rows from 20k cells x top 200 CCI pairs x top 80 celltype groups. A 50k expansion was stopped because Copulacci materializes dense spatial adjacency and triggered memory risk.
- `NicheNet` A100 retry remains a reproducible failure card: R dependency/API audit failed, and the method is treated as downstream receiver-response support rather than a direct spatial CCI ranker.
- `SCILD` was reopened after the official source was identified and now has an A100 bounded result: `3000 cells x 20 common CCI pairs`, `2,880` standardized rows, peak RSS about `18.9GB`, runtime about `116s`.

## Cervical Cross-Dataset Run

- Dataset: `atera_cervical_wta`.
- PDC root: `/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/datasets/atera_cervical_wta`.
- Prepare, smoke, pilot, and full common-db are completed.
- Full standardized output: `/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/datasets/atera_cervical_wta/runs/full_common/pyxenium/pyxenium_standardized.tsv`.
- Full rows: `2,404,971`.
- Full top hit: `DSC2-DSG3`, `Differentiating Tumor Cells -> Differentiating Tumor Cells`, `score_raw=0.8022`.
- Previous smoke top hit: `ADAM10-CD44`, `Hypoxic Tumor Cells -> Hypoxic Tumor Cells`.
- Previous pilot top hit: `APOE-ABCA1`, `Macrophages -> Macrophages`.

## A100 Rescue Status

- Active monitor automation: `lr-cci-benchmark-10m-monitor`.
- Active A100 rescue supervisor: none.
- Current A100 task: none; no matching `a100_r_method_rescue.py`, `run_niches.R`, or `run_spatalk.R` processes are running.
- `NICHES` pilot50k bounded rescue is complete: 33/33 chunks aggregated, `1,181,042` rows, peak RSS about `108GB`.
- The earlier `NICHES` full170k 100-LR and 50-LR fallback attempts were stopped for A100 memory safety.
- `SpaTalk` is finalized as bounded terminal evidence from smoke20k: 66/66 chunks aggregated, `37,532` total rows, peak RSS about `17.4GB`. The pilot50k `chunk_000_0000_0050` attempt triggered the A100 low-memory guard four times, so no further automatic pilot/full restart is scheduled.
- `CellAgentChat` pilot50k bounded rescue is complete: 16/16 chunks aggregated, `1,143,543` rows.
- `CellNEST` pilot50k bounded rescue is complete: `157,553` rows.
- `Giotto` full170k hit the R Matrix/TsparseMatrix `2^31-1` limit; keep the 50k bounded result.
- `SCILD` is upgraded to bounded appendix evidence after the official source-backed A100 retry succeeded.
- `Copulacci` is upgraded to bounded appendix evidence after the official source-backed A100 `run_scc` retry succeeded at 20k scale; no 50k/full retry is scheduled because of dense adjacency memory risk.

## PDC Status

- PDC queue for `hutaobo` is currently empty.
- COMMOT breast full chunked run is complete: 16/16 chunks, merged to `724,604` rows.
- PDC cervical full is complete.

## Pending Appendix Work

- No pending methods remain in the refreshed `final_closeout_20260511` source-of-truth.
- No deferred methods remain in the expanded 18-method benchmark.

## Automation Policy

- The heartbeat checks every 10 minutes and may restart failed supervisors, lower chunk size, write failure cards, and refresh final status without waiting for user confirmation.
- Machine-readable monitor state is written to `/data/taobo.hu/pyxenium_lr_benchmark_2026-04/results/final_closeout_20260511/monitor_status.json`.
- Completed result files are not overwritten; aggregation requires all expected chunks.
