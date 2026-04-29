# TopoLink-CCI: a topology-guided local interaction score for spatial cell-cell interaction discovery

> **Naming update for manuscript use:** The recommended manuscript-level method name is now **TopoLink-CCI** (*Topology-guided Cell-Cell Interaction scoring*). The term **TopoLink-CCI** should be used only for the cell-cell interaction-resource mode of TopoLink-CCI, because several high-ranking axes are better described as molecular interaction or cell-cell interaction axes rather than strictly classical cell-cell interaction events.

**Chinese name:** 拓扑引导的局部互作配体-受体评分法
**Recommended abbreviation:** **TopoLink-CCI**
**Implementation:** `pyXenium.cci.cci_topology_analysis`

## English Manuscript-Style Method Description

### Title

**TopoLink-CCI: topology-guided local interaction scoring identifies spatially supported cell-cell interaction axes in Xenium WTA data**

### Short Title

TopoLink-CCI for spatial cell-cell interaction discovery

### Abstract

Spatial cell-cell interaction analysis in imaging-based transcriptomics requires more than ligand and receptor co-expression. A candidate communication axis should also be compatible with tissue topology, sender and receiver cellular niches, local spatial contact, and prior biological plausibility. We therefore developed **TopoLink-CCI**, a topology-guided local interaction score implemented in `pyXenium`. TopoLink-CCI ranks cell-cell interaction hypotheses by integrating six evidence components: ligand-to-sender topology anchoring, receptor-to-receiver topology anchoring, sender-receiver structural bridging, sender expression support, receiver expression support, and local contact support. A curated cell-cell interaction prior can optionally modulate the final score. The resulting `CCI_score` is a discovery score rather than a standalone proof of communication; downstream validation should use expression specificity, permutation nulls, spatial nulls, matched-gene controls, cross-method consensus, downstream target support, and bootstrap stability. On the Atera Xenium WTA breast cancer dataset, TopoLink-CCI generated 1,319,600 sender-receiver cell-cell interaction hypotheses and prioritized interpretable vascular, stromal, immune, and Notch axes, including `VWF-SELP`, `VWF-LRP1`, `MMRN2-CD93`, `DLL4-NOTCH3`, `CXCL12-CXCR4`, `CD48-CD2`, and `JAG1-NOTCH2`.

### Rationale

Most classical cell-cell communication methods begin with the same biological premise: ligands expressed by one cell population and receptors expressed by another population may indicate intercellular signaling. Methods such as CellPhoneDB, CellChat, Squidpy, LIANA, NicheNet, stLearn, SpatialDM, COMMOT, and SpaTalk then add different layers of evidence, including curated interaction databases, expression thresholds, permutation tests, spatial proximity, downstream target response, and cross-method consensus. TopoLink-CCI follows the same cautious principle: a high score is a hypothesis generator, and orthogonal evidence is required before interpreting a candidate axis as biologically credible.

TopoLink-CCI was designed for pyXenium analyses where a topology or structure map has already been inferred from Xenium WTA data. Instead of treating all expressed cell-cell interaction pairs equally, TopoLink-CCI asks whether the ligand is topologically anchored to the sender compartment, whether the receptor is anchored to the receiver compartment, whether the sender and receiver lie in a plausible tissue-structural relationship, whether both genes are expressed in the relevant populations, and whether local neighboring cells support the proposed contact.

### Method Name

We recommend naming the method **TopoLink-CCI**.

**Full name:** Topology-guided Local Interaction Cell-Cell Interaction scoring
**Chinese full name:** 拓扑引导的局部互作配体-受体评分法

The name captures the three defining properties of the method:

1. **Topo:** it uses topology-derived gene-to-cell and cell-to-cell structure information.
2. **Link:** it links ligand, receptor, sender, receiver, and local spatial contact into one hypothesis.
3. **LR:** it is designed specifically for cell-cell interaction discovery and prioritization.

### Inputs

TopoLink-CCI uses the following inputs:

1. A spatial expression matrix, ideally with gene symbols as feature names.
2. Cell metadata containing a sender/receiver grouping column, typically `cell_type` or `cluster`.
3. Spatial coordinates for every cell.
4. A topology output that maps genes to cell types and cell types to tissue structures.
5. A cell-cell interaction resource containing at minimum ligand and receptor gene symbols.
6. Optional prior confidence weights for cell-cell interaction pairs.

For the Atera Xenium WTA breast benchmark, the full dataset contained 170,057 cells, 18,028 RNA features, 20 annotated cell clusters, and a common cell-cell interaction resource of 3,299 pairs.

### Score Components

For a ligand \(l\), receptor \(r\), sender cell type \(s\), and receiver cell type \(t\), TopoLink-CCI computes the following components.

#### 1. Sender Anchor

The sender anchor measures whether the ligand is topologically close to the sender population:

\[
A_{\mathrm{sender}}(l,s) = 1 - D_{\mathrm{topology}}(l,s)
\]

where \(D_{\mathrm{topology}}(l,s)\) is the topology distance or dissimilarity between ligand \(l\) and sender cell type \(s\). Higher values indicate stronger ligand-sender anchoring.

#### 2. Receiver Anchor

The receiver anchor measures whether the receptor is topologically close to the receiver population:

\[
A_{\mathrm{receiver}}(r,t) = 1 - D_{\mathrm{topology}}(r,t)
\]

Higher values indicate stronger receptor-receiver anchoring.

#### 3. Structure Bridge

The structure bridge measures whether the sender and receiver compartments are compatible in the inferred tissue structure:

\[
B(s,t) = 1 - D_{\mathrm{structure}}(s,t)
\]

This term favors cell-cell interaction hypotheses whose sender and receiver cell types are topologically or structurally connected.

#### 4. Sender Expression Support

The sender expression component measures whether ligand \(l\) is expressed in sender cell type \(s\). pyXenium uses a pseudobulk-detection support score:

\[
E(l,s) = \mathrm{rowNorm}\left(P(l,s)\sqrt{F(l,s)}\right)
\]

where \(P(l,s)\) is the pseudobulk expression share of ligand \(l\) in sender \(s\), and \(F(l,s)\) is the detection fraction of ligand \(l\) in sender cells.

#### 5. Receiver Expression Support

The receiver expression component is computed analogously:

\[
E(r,t) = \mathrm{rowNorm}\left(P(r,t)\sqrt{F(r,t)}\right)
\]

This term requires the receptor to be detected in the receiver population.

#### 6. Local Contact Support

The local contact component measures whether neighboring sender and receiver cells jointly express the ligand and receptor:

\[
C(l,r,s,t) =
\mathbb{1}_{N_{s,t} \geq N_{\min}}
\sqrt{S_{\mathrm{norm}}(l,r,s,t) \cdot Q(l,r,s,t)}
\cdot W(N_{s,t})
\]

where \(N_{s,t}\) is the number of local sender-receiver edges, \(N_{\min}\) is the minimum edge threshold, \(S_{\mathrm{norm}}\) is the normalized mean cell-cell interaction edge strength, \(Q\) is the active-edge coverage fraction, and \(W\) is an edge-count support term. In the current implementation, local neighborhoods are constructed by K-nearest neighbors or radius-based spatial neighborhoods.

### Final Score

The final score is a prior-weighted geometric mean:

\[
\mathrm{TopoLink\mbox{-}LR}(l,r,s,t)
=
\pi(l,r)
\cdot
\mathrm{GM}
\left[
A_{\mathrm{sender}},
A_{\mathrm{receiver}},
B,
E_{\mathrm{sender}},
E_{\mathrm{receiver}},
C
\right]
\]

where \(\pi(l,r)\) is the optional prior confidence of the cell-cell interaction pair, and GM is the geometric mean. The geometric mean is used because a strong candidate should be supported by all major evidence components; one extremely high component should not fully compensate for missing topology, expression, or contact evidence.

### Output

TopoLink-CCI produces a ranked sender-receiver cell-cell interaction table. The main columns are:

| Column | Meaning |
|---|---|
| `ligand` | Ligand gene symbol |
| `receptor` | Receptor gene symbol |
| `sender` | Sender cell type |
| `receiver` | Receiver cell type |
| `CCI_score` | TopoLink-CCI discovery score |
| `sender_anchor` | Ligand-sender topology support |
| `receiver_anchor` | Receptor-receiver topology support |
| `structure_bridge` | Sender-receiver structural support |
| `sender_expr` | Ligand expression support in sender |
| `receiver_expr` | Receptor expression support in receiver |
| `local_contact` | Spatial contact support |
| `prior_confidence` | Ligand-receptor database prior |
| `cross_edge_count` | Number of local sender-receiver edges |

### Interpretation

TopoLink-CCI should be interpreted as a **spatially constrained ranking method**. A high `CCI_score` indicates that the cell-cell interaction pair is simultaneously supported by topology, expression, tissue structure, and local contact. It does not prove protein secretion, receptor binding, or downstream signaling by itself.

The most appropriate interpretation is:

> "This cell-cell interaction axis is a high-priority spatial communication hypothesis supported by Xenium WTA topology and local tissue organization."

### Benchmark Example

In the Atera Xenium WTA breast cancer benchmark, TopoLink-CCI produced 1,319,600 full common-database results. The top-ranked axis was:

| CCI pair | Sender -> Receiver | CCI_score | Interpretation |
|---|---|---:|---|
| `VWF-SELP` | Endothelial Cells -> Endothelial Cells | 0.791289 | Endothelial activation / Weibel-Palade body / vascular adhesion state |

For this axis, the score was supported by high sender anchor, high receiver anchor, maximal structure bridge, maximal expression support, and measurable endothelial-endothelial local contact:

| Component | Value |
|---|---:|
| `sender_anchor` | 0.955713 |
| `receiver_anchor` | 0.881913 |
| `structure_bridge` | 1.000000 |
| `sender_expr` | 1.000000 |
| `receiver_expr` | 1.000000 |
| `local_contact` | 0.291245 |
| `prior_confidence` | 1.000000 |

### Validation Strategy

Following the computational validation principles used in classical cell-cell interaction and cell-cell communication studies, TopoLink-CCI discoveries should be validated with orthogonal evidence layers:

1. **Expression specificity:** ligand enriched in sender and receptor enriched in receiver.
2. **Cell-label permutation:** sender-receiver specificity exceeds randomized cell-type labels.
3. **Spatial null control:** spatial contact exceeds randomized or permuted spatial neighborhoods.
4. **Matched-gene negative controls:** the axis outperforms expression-matched random gene pairs.
5. **Downstream target support:** receiver cells show compatible target or pathway activity.
6. **Cross-method consensus:** related biology appears in CellPhoneDB, LIANA, SpatialDM, stLearn, LARIS, Squidpy, CellChat, COMMOT, SpaTalk, or other methods.
7. **Component ablation:** the score is not driven by one isolated component.
8. **Bootstrap stability:** top axes remain stable under stratified cell subsampling.

In the current PDC clean validation run, seven biologically interpretable TopoLink-CCI axes were classified as having strong computational support: `VWF-SELP`, `VWF-LRP1`, `MMRN2-CD93`, `CD48-CD2`, `DLL4-NOTCH3`, `CXCL12-CXCR4`, and `JAG1-NOTCH2`.

### Strengths

TopoLink-CCI has several practical advantages for Xenium WTA data:

1. It explicitly uses spatial topology rather than only pseudobulk expression.
2. It penalizes cell-cell interaction hypotheses lacking local cell-cell contact.
3. It keeps component-level diagnostics, making biological interpretation transparent.
4. It works naturally with pyXenium topology, contour, pathway, and mechanostress analyses.
5. It can be benchmarked against both spatial and non-spatial CCI methods through a common standardized output schema.

### Limitations

TopoLink-CCI has important limitations:

1. RNA co-localization does not prove protein-level ligand secretion or receptor binding.
2. A high score may reflect a shared cellular state, especially for autocrine or same-cell-type axes.
3. The method depends on the quality of cell-type annotation and topology inference.
4. Genes with very strong cell-type specificity may score highly and still require null controls.
5. Prior database quality affects which cell-cell interaction pairs are tested.
6. Downstream functional validation remains necessary for causal claims.

### Recommended Reporting Language

For manuscripts, we recommend the following language:

> "We used TopoLink-CCI, a topology-guided local interaction score implemented in pyXenium, to prioritize spatial cell-cell interaction hypotheses. TopoLink-CCI integrates ligand-sender topology anchoring, receptor-receiver topology anchoring, sender-receiver structural bridging, sender and receiver expression support, and local spatial contact. Candidate axes were then evaluated using expression specificity, permutation nulls, spatial controls, matched-gene negative controls, cross-method consensus, downstream target support, ablation analysis, and bootstrap stability. We therefore interpret high-scoring axes as computationally supported spatial communication hypotheses rather than direct proof of protein-level signaling."

### References

1. Efremova M, et al. CellPhoneDB: inferring cell-cell communication from combined expression of multi-subunit cell-cell interaction complexes. *Nature Protocols*. 2020. https://www.nature.com/articles/s41596-020-0292-x
2. Jin S, et al. Inference and analysis of cell-cell communication using CellChat. *Nature Communications*. 2021. https://www.nature.com/articles/s41467-021-21246-9
3. Browaeys R, et al. NicheNet: modeling intercellular communication by linking ligands to target genes. *Nature Methods*. 2020. https://www.nature.com/articles/s41592-019-0667-5
4. Dimitrov D, et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-seq data. *Nature Communications*. 2022. https://www.nature.com/articles/s41467-022-30755-0
5. Palla G, et al. Squidpy: a scalable framework for spatial omics analysis. *Nature Methods*. 2022. https://www.nature.com/articles/s41592-021-01358-2
6. Pham D, et al. stLearn: integrating spatial location, tissue morphology and gene expression to find cell-cell interactions. *Nature Communications*. 2023. https://www.nature.com/articles/s41467-023-43120-6
7. Li H, et al. SpatialDM for spatially resolved transcriptomics cell-cell interaction inference. *Nature Communications*. 2023. https://www.nature.com/articles/s41467-023-39608-w
8. Cang Z, et al. Screening cell-cell communication in spatial transcriptomics via collective optimal transport. *Nature Methods*. 2023. https://www.nature.com/articles/s41592-022-01728-4
9. Shao X, et al. SpaTalk: inferring spatially resolved cell-cell communication. *Nature Communications*. 2022. https://www.nature.com/articles/s41467-022-32111-8

---

# TopoLink-CCI：用于空间配体-受体发现的拓扑引导局部互作评分法

**英文名称：** Topology-guided Local Interaction Cell-Cell Interaction scoring
**推荐缩写：** **TopoLink-CCI**
**实现函数：** `pyXenium.cci.cci_topology_analysis`

## 中文论文式方法说明

### 标题

**TopoLink-CCI：一种用于 Xenium WTA 空间配体-受体发现的拓扑引导局部互作评分方法**

### 短标题

TopoLink-CCI 空间配体-受体分析

### 摘要

在基于成像的空间转录组数据中，配体-受体分析不能只依赖配体和受体是否表达。一个可信的细胞通讯候选轴还应当同时满足组织拓扑合理性、sender 和 receiver 细胞生态位匹配、局部空间接触支持以及已有配体-受体知识库的生物学先验。为此，我们在 pyXenium 中开发了 **TopoLink-CCI**，即拓扑引导的局部互作配体-受体评分法。TopoLink-CCI 综合六类证据：配体与 sender 的拓扑锚定、受体与 receiver 的拓扑锚定、sender-receiver 结构桥接、sender 表达支持、receiver 表达支持以及局部空间接触支持，并可进一步引入配体-受体数据库先验。TopoLink-CCI 的 `CCI_score` 是候选发现评分，而不是单独证明细胞通讯真实存在的证据。因此，高分候选轴需要进一步通过表达特异性、置换检验、空间 null、表达匹配随机基因对、跨方法一致性、下游靶基因支持和重采样稳定性进行验证。在 Atera Xenium WTA 乳腺癌数据中，TopoLink-CCI 生成了 1,319,600 条 sender-receiver 配体-受体假设，并优先发现了具有明确生物学解释的血管、基质、免疫和 Notch 相关轴，例如 `VWF-SELP`、`VWF-LRP1`、`MMRN2-CD93`、`DLL4-NOTCH3`、`CXCL12-CXCR4`、`CD48-CD2` 和 `JAG1-NOTCH2`。

### 方法动机

经典细胞通讯方法通常从同一个生物学假设出发：如果一个细胞群表达配体，另一个细胞群表达相应受体，那么二者之间可能存在细胞间通讯。CellPhoneDB、CellChat、Squidpy、LIANA、NicheNet、stLearn、SpatialDM、COMMOT 和 SpaTalk 等方法会在这个基础上叠加不同的证据层，例如配体-受体知识库、表达阈值、置换检验、空间邻近性、下游靶基因响应和多方法一致性。TopoLink-CCI 采用同样谨慎的思想：高分结果首先是一个候选发现，真正的生物学解释需要独立证据支持。

TopoLink-CCI 面向已经完成 pyXenium 拓扑分析的 Xenium WTA 数据。它不仅判断配体和受体是否表达，还进一步询问：配体是否拓扑上锚定到 sender 细胞类型，受体是否拓扑上锚定到 receiver 细胞类型，sender 和 receiver 是否具有合理的组织结构关系，两者是否在局部空间邻域中实际相邻，以及局部邻接细胞是否共同支持该配体-受体轴。

### 方法命名

推荐方法名称为 **TopoLink-CCI**。

**英文全称：** Topology-guided Local Interaction Cell-Cell Interaction scoring
**中文全称：** 拓扑引导的局部互作配体-受体评分法

这个名称概括了方法的三个核心特征：

1. **Topo：** 方法使用基因-细胞类型和细胞类型-组织结构的拓扑信息。
2. **Link：** 方法把配体、受体、sender、receiver 和局部空间接触连接成一个完整假设。
3. **LR：** 方法专门用于 cell-cell interaction 候选轴发现和排序。

### 输入数据

TopoLink-CCI 需要以下输入：

1. 空间表达矩阵，推荐使用 gene symbol 作为特征名。
2. 细胞元数据，其中包含 sender/receiver 分组列，例如 `cell_type` 或 `cluster`。
3. 每个细胞的空间坐标。
4. pyXenium 拓扑分析结果，包括基因到细胞类型、细胞类型到组织结构的映射。
5. 配体-受体数据库，至少包含 ligand 和 receptor 基因名。
6. 可选的配体-受体先验置信度。

在 Atera Xenium WTA 乳腺癌 benchmark 中，完整数据包含 170,057 个细胞、18,028 个 RNA features、20 个细胞簇，以及 3,299 对 common cell-cell interaction pairs。

### 评分组件

对于配体 \(l\)、受体 \(r\)、sender 细胞类型 \(s\) 和 receiver 细胞类型 \(t\)，TopoLink-CCI 计算以下六个组件。

#### 1. Sender Anchor

sender anchor 衡量配体是否在拓扑上贴近 sender 细胞群：

\[
A_{\mathrm{sender}}(l,s) = 1 - D_{\mathrm{topology}}(l,s)
\]

其中 \(D_{\mathrm{topology}}(l,s)\) 表示配体 \(l\) 与 sender 细胞类型 \(s\) 之间的拓扑距离或不相似度。数值越高，说明配体越锚定到 sender。

#### 2. Receiver Anchor

receiver anchor 衡量受体是否在拓扑上贴近 receiver 细胞群：

\[
A_{\mathrm{receiver}}(r,t) = 1 - D_{\mathrm{topology}}(r,t)
\]

数值越高，说明受体越锚定到 receiver。

#### 3. Structure Bridge

structure bridge 衡量 sender 和 receiver 在组织结构中是否存在合理连接：

\[
B(s,t) = 1 - D_{\mathrm{structure}}(s,t)
\]

该项提高那些发生在结构上相邻、相连或组织生态位上相关的 sender-receiver 组合的得分。

#### 4. Sender Expression Support

sender expression support 衡量配体 \(l\) 是否在 sender 细胞群 \(s\) 中表达。pyXenium 使用 pseudobulk-detection 组合评分：

\[
E(l,s) = \mathrm{rowNorm}\left(P(l,s)\sqrt{F(l,s)}\right)
\]

其中 \(P(l,s)\) 是配体 \(l\) 在 sender \(s\) 中的 pseudobulk 表达份额，\(F(l,s)\) 是 sender 细胞中检测到该配体的细胞比例。

#### 5. Receiver Expression Support

receiver expression support 以同样方式计算：

\[
E(r,t) = \mathrm{rowNorm}\left(P(r,t)\sqrt{F(r,t)}\right)
\]

该项要求受体在 receiver 细胞群中有表达支持。

#### 6. Local Contact Support

local contact support 衡量局部相邻的 sender 和 receiver 细胞是否共同表达该配体-受体组合：

\[
C(l,r,s,t) =
\mathbb{1}_{N_{s,t} \geq N_{\min}}
\sqrt{S_{\mathrm{norm}}(l,r,s,t) \cdot Q(l,r,s,t)}
\cdot W(N_{s,t})
\]

其中 \(N_{s,t}\) 是 sender-receiver 局部边数量，\(N_{\min}\) 是最小边数阈值，\(S_{\mathrm{norm}}\) 是标准化后的局部边表达强度，\(Q\) 是活跃边覆盖比例，\(W\) 是边数支持项。当前实现可使用 K 近邻或半径邻域构建局部空间图。

### 最终得分

最终得分为先验加权的几何平均：

\[
\mathrm{TopoLink\mbox{-}LR}(l,r,s,t)
=
\pi(l,r)
\cdot
\mathrm{GM}
\left[
A_{\mathrm{sender}},
A_{\mathrm{receiver}},
B,
E_{\mathrm{sender}},
E_{\mathrm{receiver}},
C
\right]
\]

其中 \(\pi(l,r)\) 是配体-受体先验置信度，GM 为几何平均。使用几何平均的原因是：一个可信的 LR 轴应当同时得到多个证据组件支持；某一个组件极高不应完全补偿拓扑、表达或局部接触证据的缺失。

### 输出结果

TopoLink-CCI 输出一个按 sender-receiver 配体-受体候选轴排序的表格。主要列包括：

| 列名 | 含义 |
|---|---|
| `ligand` | 配体基因名 |
| `receptor` | 受体基因名 |
| `sender` | sender 细胞类型 |
| `receiver` | receiver 细胞类型 |
| `CCI_score` | TopoLink-CCI 发现评分 |
| `sender_anchor` | 配体-sender 拓扑支持 |
| `receiver_anchor` | 受体-receiver 拓扑支持 |
| `structure_bridge` | sender-receiver 结构支持 |
| `sender_expr` | sender 中配体表达支持 |
| `receiver_expr` | receiver 中受体表达支持 |
| `local_contact` | 局部空间接触支持 |
| `prior_confidence` | 配体-受体数据库先验 |
| `cross_edge_count` | 局部 sender-receiver 边数量 |

### 结果解释

TopoLink-CCI 应被理解为一种 **空间约束的候选排序方法**。高 `CCI_score` 表示该配体-受体轴同时获得拓扑、表达、组织结构和局部接触支持，但它本身不能证明蛋白分泌、受体结合或下游信号激活。

推荐的解释方式是：

> "该配体-受体轴是一个由 Xenium WTA 拓扑和局部组织结构支持的高优先级空间细胞通讯候选假设。"

### Benchmark 示例

在 Atera Xenium WTA 乳腺癌 benchmark 中，TopoLink-CCI 生成了 1,319,600 条 full common-database 结果。排名第一的轴为：

| CCI pair | Sender -> Receiver | CCI_score | 生物学解释 |
|---|---|---:|---|
| `VWF-SELP` | Endothelial Cells -> Endothelial Cells | 0.791289 | 内皮激活 / Weibel-Palade body / 血管黏附状态 |

该轴同时具有较高的 sender anchor、receiver anchor、structure bridge、表达支持和可测量的 endothelial-endothelial 局部接触：

| 组件 | 数值 |
|---|---:|
| `sender_anchor` | 0.955713 |
| `receiver_anchor` | 0.881913 |
| `structure_bridge` | 1.000000 |
| `sender_expr` | 1.000000 |
| `receiver_expr` | 1.000000 |
| `local_contact` | 0.291245 |
| `prior_confidence` | 1.000000 |

### 验证策略

参考经典 LR/CCC 方法论文中的计算型防假阳性思想，TopoLink-CCI 发现需要通过以下独立证据层验证：

1. **表达特异性：** 配体在 sender 中富集，受体在 receiver 中富集。
2. **细胞标签置换：** 真实 sender-receiver 特异性高于随机 cell-type label。
3. **空间 null：** 真实局部接触高于随机或置换后的空间邻域。
4. **表达匹配随机基因对：** 真实 LR 轴优于表达水平相近的随机 gene pairs。
5. **下游靶基因支持：** receiver 细胞具有兼容的靶基因或通路活性。
6. **跨方法一致性：** 相关生物学主题可被 CellPhoneDB、LIANA、SpatialDM、stLearn、LARIS、Squidpy、CellChat、COMMOT、SpaTalk 或其他方法支持。
7. **组件消融：** 高分结果不是由单一组件人为推高。
8. **重采样稳定性：** 在分层 bootstrap 子采样中，高分轴排名稳定。

在当前 PDC clean validation run 中，七条具有明确生物学解释的 TopoLink-CCI 轴被评为 strong computational support：`VWF-SELP`、`VWF-LRP1`、`MMRN2-CD93`、`CD48-CD2`、`DLL4-NOTCH3`、`CXCL12-CXCR4` 和 `JAG1-NOTCH2`。

### 方法优势

TopoLink-CCI 对 Xenium WTA 数据具有以下优势：

1. 显式利用空间拓扑，而不仅仅依赖 pseudobulk 表达。
2. 对缺乏局部细胞接触的 LR 假设进行惩罚。
3. 保留组件级诊断，使生物学解释更加透明。
4. 能自然结合 pyXenium 的 topology、contour、pathway 和 mechanostress 分析。
5. 可通过统一输出 schema 与空间和非空间 LR 方法进行 benchmark。

### 局限性

TopoLink-CCI 也有重要局限：

1. RNA 共定位不能证明蛋白水平分泌、受体结合或功能激活。
2. 高分同细胞类型轴可能反映共享细胞状态，而不一定是经典旁分泌信号。
3. 方法依赖细胞类型注释和拓扑推断质量。
4. 细胞类型特异性极强的高表达基因可能得到高分，因此仍需 null controls。
5. 配体-受体数据库质量会影响候选集合。
6. 若要提出因果机制，仍需要功能实验或蛋白层验证。

### 论文推荐表述

建议在论文中这样描述：

> "我们使用 TopoLink-CCI，一种在 pyXenium 中实现的拓扑引导局部互作评分方法，对空间配体-受体候选轴进行排序。TopoLink-CCI 综合配体-sender 拓扑锚定、受体-receiver 拓扑锚定、sender-receiver 结构桥接、sender 和 receiver 表达支持以及局部空间接触。随后，我们使用表达特异性、置换 null、空间 controls、表达匹配随机基因对、跨方法一致性、下游靶基因支持、组件消融和 bootstrap 稳定性对候选轴进行验证。因此，我们将高分轴解释为具有计算证据支持的空间细胞通讯假设，而不是蛋白水平信号传递的直接证明。"

### 参考文献

1. Efremova M, et al. CellPhoneDB: inferring cell-cell communication from combined expression of multi-subunit cell-cell interaction complexes. *Nature Protocols*. 2020. https://www.nature.com/articles/s41596-020-0292-x
2. Jin S, et al. Inference and analysis of cell-cell communication using CellChat. *Nature Communications*. 2021. https://www.nature.com/articles/s41467-021-21246-9
3. Browaeys R, et al. NicheNet: modeling intercellular communication by linking ligands to target genes. *Nature Methods*. 2020. https://www.nature.com/articles/s41592-019-0667-5
4. Dimitrov D, et al. Comparison of methods and resources for cell-cell communication inference from single-cell RNA-seq data. *Nature Communications*. 2022. https://www.nature.com/articles/s41467-022-30755-0
5. Palla G, et al. Squidpy: a scalable framework for spatial omics analysis. *Nature Methods*. 2022. https://www.nature.com/articles/s41592-021-01358-2
6. Pham D, et al. stLearn: integrating spatial location, tissue morphology and gene expression to find cell-cell interactions. *Nature Communications*. 2023. https://www.nature.com/articles/s41467-023-43120-6
7. Li H, et al. SpatialDM for spatially resolved transcriptomics cell-cell interaction inference. *Nature Communications*. 2023. https://www.nature.com/articles/s41467-023-39608-w
8. Cang Z, et al. Screening cell-cell communication in spatial transcriptomics via collective optimal transport. *Nature Methods*. 2023. https://www.nature.com/articles/s41592-022-01728-4
9. Shao X, et al. SpaTalk: inferring spatially resolved cell-cell communication. *Nature Communications*. 2022. https://www.nature.com/articles/s41467-022-32111-8
