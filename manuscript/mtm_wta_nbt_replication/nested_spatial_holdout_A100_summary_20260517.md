# Nested spatial holdout A100 summary, 2026-05-17

Each fold selected an embedding feature only on training spatial blocks, then evaluated the selected and locked features on the held-out block. This is a stricter stress test than the source-data spatial permutation and should be interpreted as fold-level stability, not a new discovery screen.

- breast / PLIP: 3 candidates; selected-feature sign-match median 0.750 (minimum 0.750); locked-feature sign-match median 0.812 (minimum 0.750); median selected-feature reuse 1.000.
- breast / UNI: 5 candidates; selected-feature sign-match median 0.812 (minimum 0.750); locked-feature sign-match median 0.875 (minimum 0.750); median selected-feature reuse 1.000.
- cervical / PLIP: 10 candidates; selected-feature sign-match median 0.688 (minimum 0.562); locked-feature sign-match median 0.688 (minimum 0.562); median selected-feature reuse 0.875.
- cervical / UNI: 5 candidates; selected-feature sign-match median 0.688 (minimum 0.375); locked-feature sign-match median 0.688 (minimum 0.625); median selected-feature reuse 0.750.

The primary breast PLIP candidates show held-out sign support in 75.0-93.8% of folds. Weaker cross-cancer/model stress-test candidates, especially cervical UNI EMT invasion, are intentionally reported as less stable rather than promoted as primary claims.
