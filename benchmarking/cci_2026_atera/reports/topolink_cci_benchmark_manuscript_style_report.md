# TopoLink-CCI Benchmarking Manuscript-style Report

## Overall Progress

本轮 TopoLink-CCI benchmarking 已从单一方法验证推进到多层次、可追踪的空间 CCI 评估体系。整体工作以 PDC clean results 为主线，A100 作为补充计算和 bounded rescue 环境，最终形成了 Breast WTA 主 benchmark、A100 supplementary full/appendix runs、Cervical WTA cross-dataset generalization、Synthetic Truth validation 和 false-positive controls 五个互相支撑的证据层。

截至当前收口状态，PDC publication 24h supervisor 的队列为 **0**，A100 侧没有 active rescue process。24h 汇总根目录为 **`D:\GitHub\pyXenium\benchmarking\cci_2026_atera\results\publication_benchmark_24h_20260511`**；PDC 正式运行根目录为 **`/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04`**；A100 历史补充与 rescue 结果主要保存在 **`/data/taobo.hu/pyxenium_lr_benchmark_2026-04`**。这些结果没有被混合覆盖，而是以 full 和 bounded 两级终端结果进入最终解释框架；reproducible failure card 作为记录机制保留，但当前 expanded denominator 中已无剩余 failure card。

benchmark 的主体已经收口为一个分层结果体系。为避免遗漏任何已讨论方法，本报告采用 upgraded expanded denominator：Breast WTA 共计 **18** 个方法被计入，其中 **9** 个 full success、**9** 个 bounded success、**0** 个 reproducible failure card、**0** 个 deferred candidate methods，且 **0** 个 pending/running。因此，`all_methods_accounted=true` 表示所有提到的方法都有明确终端状态。Breast WTA common-db 主比较层包含 TopoLink-CCI、CellPhoneDB、LARIS、LIANA+、SpatialDM 和 stLearn；A100 supplement 提供 Squidpy、CellChat LR-only 和 COMMOT chunked full/appendix 作为方法学参照；bounded appendix 则记录 Giotto、SpaTalk、NICHES、CellNEST、CellAgentChat、FastCCC、SCILD、Copulacci 和 NicheNet 在较大数据量下的可扩展性边界。Cervical WTA 作为 cross-dataset generalization 数据集，已经完成 TopoLink-CCI whole-dataset full common-db run。FastCCC、SCILD、Copulacci 和 NicheNet 现在均已正式终端化，并经 A100 retry 升级为 bounded subset result。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/preview_completed_20260511/figures/fig1_completion_status.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/preview_completed_20260511/source_data/fig1_completion_status.tsv`]**

### Summary of Evaluated Methods

| Dataset | Method | Status | Evidence tier | Phase | Rows | Hardware/source | Interpretation |
|---|---|---:|---|---|---:|---|---|
| Breast WTA | TopoLink-CCI | Full | Main comparison | full common-db | **1,319,600 rows** | PDC | 主方法；PDC clean full-common 结果，路径为 **`pdc_collected/pdc_20260426_1327/runs/full_common/pyxenium`** |
| Breast WTA | CellPhoneDB | Full | Main comparison | full common-db | reported in artifact | PDC | 可复现的 non-spatial expression baseline，用于表达驱动 CCI 参照 |
| Breast WTA | LARIS | Full | Main comparison | full common-db | reported in artifact | PDC | diffusion/spatial smoothing 参照，补充局部拓扑模型 |
| Breast WTA | LIANA+ | Full | Main comparison | full common-db | reported in artifact | PDC | multi-method/spatial bivariate 参照；需注意 Unassigned 相关 top hits |
| Breast WTA | SpatialDM | Full | Main comparison | full common-db | reported in artifact | PDC | spatial co-expression / spatial null 类方法参照 |
| Breast WTA | stLearn | Full | Main comparison | full common-db | reported in artifact | PDC/A100 | spatial neighborhood 和 permutation-style CCI 参照 |
| Breast WTA | Squidpy | Full | A100 supplement | full common-db | **1,319,600 rows** | A100 | Squidpy ligrec full result，作为 permutation-style LR/CCI supplement |
| Breast WTA | CellChat | Full | A100 supplement | full common-db LR-only | **31,592 rows** | A100 | LR-only full；pathway step 因已知 runtime bug 禁用 |
| Breast WTA | COMMOT | Full | A100/PDC appendix | full common-db chunked | **724,604 rows** | A100/PDC | 16/16 chunks merged；用于 transport/diffusion 类方法参照 |
| Breast WTA | Giotto | Bounded | Bounded appendix | pilot50k common-db | **1,151,351 rows** | A100 | pilot50k 完成；full 受 R Matrix/TsparseMatrix **2^31-1** sparse index 限制 |
| Breast WTA | SpaTalk | Bounded | Bounded appendix | smoke20k common-db | **37,532 rows** | A100 | smoke20k 完成 **66/66 chunks**；pilot50k 未产出有效结果，反映 scalability/resource tradeoff |
| Breast WTA | NICHES | Bounded | Bounded appendix | pilot50k common-db | **1,181,042 rows** | A100 | pilot50k 完成；full170k 被 memory safety gate 停止 |
| Breast WTA | CellNEST | Bounded | Bounded appendix | pilot50k common-db | **157,553 rows** | A100 | bounded GPU/source workflow result；full 资源开销不可控 |
| Breast WTA | CellAgentChat | Bounded | Bounded appendix | pilot50k common-db | **1,143,543 rows** | A100 | pilot50k 16/16 chunks 成功；full170k resource_exceeded |
| Breast WTA | SCILD | Bounded | Bounded appendix | smoke3k common-db | **2,880 rows** | A100 | official source-backed ligand-diffusion bounded retry succeeded：3000 cells × 20 CCI pairs，peak RSS about **18.9GB**，runtime about **116s** |
| Breast WTA | FastCCC | Bounded | Bounded appendix | smoke20k common-db | **1,319,600 rows** | A100 | A100 retry succeeded after PDC standardization failure；analytic non-spatial expression baseline |
| Breast WTA | Copulacci | Bounded | Bounded appendix | smoke20k common-db | **13,981 rows** | A100 | official source-backed `copulacci.model2.run_scc` bounded retry succeeded：20k cells × top 200 CCI pairs × top 80 celltype groups；50k 扩容因 dense adjacency memory risk 停止 |
| Breast WTA | NicheNet | Bounded | Bounded appendix | smoke20k common-db | **215,542 rows** | A100 | clean R retry with official `saeyslab/nichenetr` and Zenodo v2 ligand-target prior；作为 downstream receiver-response support 方法，不作为 direct spatial CCI ranker |
| Cervical WTA | TopoLink-CCI | Full | Cross-dataset | full common-db | **2,404,971 rows** | PDC | cervical whole-dataset generalization；top hit 为 **DSC2-DSG3 / Differentiating Tumor Cells -> Differentiating Tumor Cells** |

## Validation & Controls

真实组织切片中的 CCI 发现没有绝对 ground truth，因此本研究先使用 PDC Synthetic Truth 建立方法行为的可解释基线，再将该基线用于解释 Breast WTA full benchmark。Synthetic Truth 的设计目的不是替代真实数据，而是验证 TopoLink-CCI 的核心假设：单一拓扑锚定不能充分排序 CCI 候选，完整模型需要同时整合 Topology Anchor、expression specificity 和 local contact。

PDC synthetic truth 结果显示，完整 TopoLink-CCI 模型获得 **AUROC 0.9919** 和 **AUPRC 0.8333**。作为对照，expression-only 获得 **AUROC 0.9839**、**AUPRC 0.8333**；contact-only 获得 **AUROC 0.9919**、**AUPRC 0.8333**；而 topology-anchor-only 虽然仍有 **AUROC 0.9839**，但 **AUPRC 0.5833** 明显下降。这个结果说明，在候选排序任务中，Topology Anchor 单独使用可以捕获一部分宏观组织结构信息，却不足以稳定地区分真正植入的 positive CCI axes；完整模型通过 expression specificity 避免纯空间共定位假阳性，并通过 local contact 提升局部相互作用的 precision-recall ranking。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/publication_benchmark_24h_20260511/pdc_collected/publication_benchmark_24h_20260511/synthetic_truth/figures/synthetic_truth_auroc_auprc.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/publication_benchmark_24h_20260511/pdc_collected/publication_benchmark_24h_20260511/synthetic_truth/tables/synthetic_truth_metrics.tsv`]**

这组结果也为后续真实 WTA benchmark 的评价方式提供了约束。TopoLink-CCI 的 CCI_score 应被理解为 discovery score，而不是直接的真实性证明；因此后续结果不直接比较不同方法的 raw scores，而是使用 rank、canonical recovery、cross-method consistency、false-positive controls、bootstrap stability 和 cross-dataset generalization 共同评估发现的可信度。PDC publication 24h run 同时收集了 false-positive control 与 method-card 结果，主要路径为 **`D:\GitHub\pyXenium\benchmarking\cci_2026_atera\results\publication_benchmark_24h_20260511`** 和 **`/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/runs/publication_benchmark_24h_20260511`**。

## Core Benchmarking Results

### Breast WTA common-db main comparison

Breast WTA 是本研究的主实验数据集，也是 TopoLink-CCI 方法学主张的核心验证场景。该数据集使用 common CCI resource 进行跨方法公平比较，避免不同方法原生数据库造成的候选空间差异。TopoLink-CCI Breast full result 产生 **1,319,600 rows**，路径为 **`pdc_collected/pdc_20260426_1327/runs/full_common/pyxenium`**。这一结果构成后续 biological interpretation、classic CCI axis selection、false-positive validation 和 cross-method consistency 的主输入。

主比较层由 TopoLink-CCI、CellPhoneDB、LARIS、LIANA+、SpatialDM 和 stLearn 构成。这个组合覆盖了互补的 CCI 推断思想：TopoLink-CCI 代表 topology-guided spatial CCI prioritization；CellPhoneDB 提供表达与置换思想的 non-spatial baseline；LARIS 提供 diffusion/spatial smoothing 参照；LIANA+ 代表 multi-method/spatial bivariate framework；SpatialDM 和 stLearn 则强调 spatial co-expression、spatial neighborhood 和 permutation-style controls。这样的组合使 benchmark 不局限于“哪个 raw score 更高”，而是比较不同方法是否恢复相同 biological themes、是否稳定定位 canonical axes，以及是否在真实组织结构中给出可解释的 sender-receiver direction。

在 Breast WTA 中，TopoLink-CCI 的高分发现集中于 vascular/endothelial activation、vascular-stromal ECM/scavenger interaction、angiogenesis、Notch、CAF-immune chemokine 和 T-cell co-stimulation 等经典轴线。其中 VWF-SELP / Endothelial Cells -> Endothelial Cells 是最高分 CCI axis，后续 deep dive 将其解释为 topology-supported endothelial activation / Weibel-Palade body / vascular adhesion niche，而不是简单的蛋白级 ligand binding 证明。该解释属于计算证据支持的 biological hypothesis，需要与蛋白水平或功能实验区分。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/preview_completed_20260511/figures/fig3_breast_topolink_top_axes.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/preview_completed_20260511/source_data/fig3_breast_topolink_top_axes.tsv`]**

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig6_canonical_pair_rank_heatmap.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig6_canonical_pair_rank_heatmap.tsv`]**

### A100 supplementary full and appendix runs

A100 结果主要承担两类角色：第一，补充 PDC 主比较中没有完全覆盖的方法；第二，记录大规模 WTA 下复杂 CCI 方法的 scalability/resource tradeoff。A100 supplement 中，Squidpy full common-db 生成 **1,319,600 rows**，与 Breast TopoLink-CCI 的候选规模一致，可作为 permutation-style ligrec 参照。CellChat 完成 LR-only full result，产生 **31,592 rows**；pathway aggregation step 因已知 runtime bug 被禁用，因此该结果用于 LR-level comparison，而不用于 pathway-level claim。COMMOT 以 chunked full/appendix 形式完成 **724,604 rows**，16/16 chunks 被合并，用于 transport/diffusion 类空间通信方法参照。

这些补充结果对于解释 cross-method consistency 非常关键。真实组织中，不同方法对同一个 exact pair 的恢复并不总是一致，因为方法假设不同：有的方法强调群体表达，有的方法强调空间邻近，有的方法强调扩散或 transport。因而本研究将 exact-pair overlap、same-pathway recovery、sender-receiver theme 和 top-k Jaccard 一起使用，而不是要求所有方法完全恢复同一排序。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig1_runtime_hours.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig1_runtime_hours.tsv`]**

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig3_top100_pair_jaccard_heatmap.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig3_top100_pair_jaccard_heatmap.tsv`]**

### Cross-dataset generalization

Cervical WTA 数据集被纳入后，TopoLink-CCI benchmark 从单一 Breast WTA 扩展到跨组织泛化。Cervical WTA 的 whole-dataset full common-db run 在 PDC 上完成，生成 **2,404,971 rows**，路径为 **`/cfs/klemming/scratch/h/hutaobo/pyxenium_cci_benchmark_2026-04/datasets/atera_cervical_wta/runs/full_common/pyxenium/pyxenium_standardized.tsv`**。其 top hit 为 **DSC2-DSG3 / Differentiating Tumor Cells -> Differentiating Tumor Cells**，与 Breast WTA 中 vascular/endothelial top axes 不同，提示 TopoLink-CCI 的最高优先级发现会随组织结构和肿瘤生态改变，而不是固定输出同一类 vascular program。

这种 cross-dataset 结果具有两层意义。首先，它支持 TopoLink-CCI 在更大规模 WTA 数据上的工程可扩展性：Cervical WTA 细胞数明显高于 Breast WTA，但 sparse-only prepare 与 full common-db run 能够完成。其次，它支持方法的生物学适应性：Breast WTA 的 top axes 偏向 vascular/stromal/immune niche，而 Cervical WTA 的 top axis 指向 differentiating tumor cell adhesion program。该差异更符合 tissue-context-specific CCI prioritization，而不是单一高表达基因列表的重复输出。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/preview_completed_20260511/figures/fig4_breast_cervical_topolink_comparison.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/preview_completed_20260511/source_data/fig4_breast_cervical_topolink_comparison.tsv`]**

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/preview_completed_20260511/figures/fig3b_cervical_topolink_top_axes.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/preview_completed_20260511/source_data/fig3b_cervical_topolink_top_axes.tsv`]**

## Scope of Benchmark and Technical Limitations

本 benchmark 的目标不是把每一个外部 CCI 方法都强行推进到 170k 或 717k cells 的 full run，而是为每个方法给出可审计的终端状态：full result、bounded subset result 或 reproducible failure card。这个原则对大规模 WTA 数据尤其重要，因为许多 CCI 方法最初并不是为数十万细胞、数千 interaction pairs 的 whole-dataset setting 设计的。

Giotto、SpaTalk、NICHES、CellNEST 和 CellAgentChat 因此被纳入 bounded appendix，而不是被简单归类为失败。Giotto pilot50k 完成 **1,151,351 rows**，说明真实 runner 和输出标准化可以工作；但 full 170k 遇到 R Matrix/TsparseMatrix **2^31-1** sparse index 限制，这反映的是 R sparse matrix backend 在超大规模 WTA CCI 推断中的结构性边界。SpaTalk smoke20k 完成 **66/66 chunks** 并产出 **37,532 rows**，证明真实方法链路可运行；pilot50k 没有有效输出，主要体现 SpaTalk 在更大 cell count 与 full common resource 下的 scalability/resource tradeoff。NICHES pilot50k 完成 **1,181,042 rows**，但 full170k 因 A100 memory safety gate 停止，说明其空间邻域/细胞间 niche 推断在高维 WTA 场景下内存开销较高。

CellNEST 和 CellAgentChat 代表更复杂的 GPU/source-workflow 类方法。CellNEST pilot50k 产生 **157,553 rows**，CellAgentChat pilot50k 产生 **1,143,543 rows**，但 full170k resource_exceeded。因此二者保留为 bounded evidence，用于说明复杂模型在 subset setting 下能否恢复可解释 CCI themes，而不被写入主 full comparison。FastCCC 在 PDC 上曾因 raw output standardization blocker 终止；A100 retry 修复 interaction ID mapping 后，完成 20k smoke 并产出 **1,319,600 rows**，因此升级为 bounded appendix analytic baseline。SCILD 在官方 source 被重新确认后，采用专用 Python 3.11 环境和严格 bounded gate 完成 **3000 cells × 20 CCI pairs** ligand-diffusion run，产生 **2,880 rows**，峰值内存约 **18.9GB**，因此从 failure card 升级为 bounded appendix。Copulacci 官方 source 安装和 import 成功后，进一步通过真实 `copulacci.model2.run_scc` workflow 完成 **20k cells × top 200 CCI pairs × top 80 celltype groups** 的 A100 bounded retry，产生 **13,981 rows**；50k 扩容因方法内部 dense spatial adjacency materialization 引发内存风险而停止，因此 Copulacci 被保留为 bounded appendix，而不是 full comparison。NicheNet 经过 clean R retry 后同样升级为 bounded appendix：A100 上官方 `saeyslab/nichenetr` 安装成功，并使用官方 Zenodo v2 ligand-target matrix 对 20k smoke 数据运行 downstream receiver-response support，产生 **215,542 rows**；该结果用于补充 receiver target-program support，不作为 direct spatial CCI ranker。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/preview_completed_20260511/figures/fig2_result_row_counts.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/preview_completed_20260511/source_data/fig2_result_row_counts.tsv`]**

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig5_overlap_with_topolink.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig5_overlap_with_topolink.tsv`]**

## Methodological Discussion

本研究采用的评价体系刻意避免直接比较不同方法的 raw score。不同 CCI/LR 方法的 score 定义差异很大：CellPhoneDB 和 Squidpy 更接近表达/置换统计，CellChat 使用 communication probability 与 pathway abstraction，SpatialDM/stLearn 更强调空间共表达或邻域约束，COMMOT 建模 transport/diffusion，TopoLink-CCI 则整合 Topology Anchor、expression specificity、structure bridge 和 local contact。因此 raw score 不是跨方法可比量，直接比较会产生错误的排名解释。

相应地，本 benchmark 选择 rank-based 和 evidence-based 指标：canonical recovery、top-k overlap、cross-method consistency、false-positive controls、Synthetic Truth metrics、bootstrap stability 和 cross-dataset generalization。F1、AUROC 和 AUPRC 只在 Synthetic Truth 或预定义 canonical truth set 中使用，因为这些场景存在明确 positive/negative definition。例如 PDC Synthetic Truth 中，TopoLink-CCI 的 **AUROC 0.9919** 与 **AUPRC 0.8333** 可以解释为植入 CCI axes 的恢复能力；但在真实 WTA 切片中，未被列为 canonical 的 CCI axis 并不等于 false positive，因此不能把所有 non-canonical axes 当作 negative class 计算全局 F1。

ARI 也不作为主指标。ARI 适用于比较 clustering assignment，而当前输出是 interaction-axis ranking，不是给每个细胞或每个 cell type pair 分配唯一 cluster label。尽管 top-k binary selection ARI heatmap 可以作为 exploratory consistency visualization，用来观察方法之间 top axis selection 的相似度，但它不能替代 biological recovery、spatial support 或 false-positive controls。换言之，TopoLink-CCI benchmark 的核心问题不是“方法是否产生相同的聚类”，而是“方法是否优先给出在空间、表达、拓扑和生物学上可解释的 CCI hypotheses”。

本研究建立的评估框架旨在为真实复杂组织切片中的 CCI 优先级排序提供可靠依据，而非单纯追求合成数据集上的指标拟合。Synthetic Truth 证明完整拓扑组合在受控条件下具有良好的 ranking behavior；Breast WTA full benchmark 展示其在真实肿瘤生态中的 biological interpretability；A100 supplement 与 bounded appendix 说明不同方法的工程可扩展性边界；Cervical WTA generalization 则证明 TopoLink-CCI 的发现会随 tissue context 改变，而不是固定输出某一类高表达轴。

因此，TopoLink-CCI 应被表述为一个 topology-guided spatial CCI hypothesis prioritization framework。它可以在 ligand-receptor-resource mode 下使用 ligand/receptor 列进行候选定义，但方法学定位是 CCI，而不是狭义 LR binding proof。所有结果均属于 computational evidence：它们可以提高候选 CCI axis 的可信度，支持后续蛋白水平、功能扰动或空间多组学验证设计，但不应被表述为单靠 CCI_score 已经证明蛋白结合、因果通讯或功能效应。

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig2_canonical_recovery_f1.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig2_canonical_recovery_f1.tsv`]**

> **[图表占位：此处引用 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/figures/fig4_top100_binary_selection_ari_heatmap.pdf`；对应 source_data 为 `benchmarking/cci_2026_atera/results/cross_method_comparison_20260511/source_data/fig4_top100_binary_selection_ari_heatmap.tsv`]**
