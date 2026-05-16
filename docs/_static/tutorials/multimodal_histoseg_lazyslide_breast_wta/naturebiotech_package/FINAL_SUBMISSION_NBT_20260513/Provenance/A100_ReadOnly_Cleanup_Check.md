# A100 Read-Only Cleanup Check

Checked UTC: 2026-05-13T12:17:40Z

Remote host: `sscb-a100.scilifelab.se`

Remote roots:

- `/data/taobo.hu/pyxenium_lazyslide_breast_wta_20260507`
- `/data/taobo.hu/pyxenium_lazyslide_cervical_wta_20260511`

Result:

- Remote roots exist: yes.
- Running mTM/autopilot/LazySlide jobs detected: none.
- Temporary candidates searched: `*.tmp`, `core.*`, `*Traceback*`, `*traceback*`.
- Temporary candidates found within max depth 4: none.

No destructive remote cleanup was performed. The final submission package is self-contained locally and no longer depends on remote runtime files.
