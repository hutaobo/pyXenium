# Online Methods outline

## TopoLink-CCI score

TopoLink-CCI evaluates each candidate molecular interaction axis using six retained components: sender topology anchor, receiver topology anchor, structure bridge, sender expression, receiver expression and local contact. The discovery score is computed as a prior-weighted geometric mean of these components.

## Topology map and expression support

Topology anchors are derived from pyXenium topology outputs linking genes, cell groups and tissue structures. Expression support is evaluated within sender and receiver cell groups and exported as diagnostic fields.

## Local contact graph

Local contact support is computed on a spatial neighbor graph and records active-edge support between sender and receiver populations.

## Synthetic Truth benchmark

Synthetic Truth data preserve tissue topology while implanting known CCI axes. AUROC, AUPRC and top-k precision/recall are used only because positive and negative axes are defined by construction.

## Benchmark status tiers

Methods were terminalized as full result or bounded subset result in the expanded Breast WTA benchmark. Reproducible failure cards remain the predefined stopping rule, but no expanded Breast method currently remains in that class. Bounded methods are not treated as equivalent to whole-dataset full methods. NicheNet is analyzed as downstream receiver-response support and is not presented as a direct spatial CCI ranker.
