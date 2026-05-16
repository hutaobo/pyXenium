# Gene Component Validation Report

This report audits whether candidate H&E embedding axes track component genes from the WTA programs used in mTM. It is a component-gene sanity check, not independent protein or IHC validation.

## Summary

- breast / PLIP / luminal_estrogen_response: 9 genes, median effective image-gene n 157, median image-gene partial rho -0.560, image-gene sign-match fraction 1.00, program high-low positive fraction 1.00; strongest gene XBP1.
- breast / PLIP / unfolded_protein_response: 9 genes, median effective image-gene n 157, median image-gene partial rho 0.285, image-gene sign-match fraction 0.78, program high-low positive fraction 1.00; strongest gene XBP1.
- breast / PLIP / oxidative_phosphorylation: 8 genes, median effective image-gene n 157, median image-gene partial rho 0.341, image-gene sign-match fraction 0.88, program high-low positive fraction 1.00; strongest gene COX6C.

## Component Genes

- luminal_estrogen_response: XBP1 (-0.671), FOXA1 (-0.611), GATA3 (-0.611), TFF1 (-0.587), ESR1 (-0.560), AGR2 (-0.548), TFF3 (-0.537), SCUBE2 (-0.528), PGR (-0.389)
- unfolded_protein_response: XBP1 (0.576), HSPA5 (0.402), HERPUD1 (0.364), ERN1 (0.358), PDIA3 (0.285), ATF4 (0.234), ATF3 (-0.141), HSP90B1 (0.111), DDIT3 (-0.013)
- oxidative_phosphorylation: COX6C (0.452), ATP5F1A (0.368), ATP5MC1 (0.348), SDHB (0.345), COX5A (0.337), UQCRC1 (0.318), NDUFA1 (0.086), NDUFB8 (-0.034)
