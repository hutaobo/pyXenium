---
orphan: true
---

# Draft PR Text for `scverse/ecosystem-packages`

## Summary

This PR proposes adding `pyXenium` to the scverse ecosystem packages list.

Proposed metadata file:

- `packages/pyXenium/meta.yaml`

## Checklist for adding packages

### Mandatory

Name of the tool: `pyXenium`

Short description:

`pyXenium` is a Python toolkit for 10x Genomics Xenium data that combines Xenium I/O, multimodal RNA+protein analysis, contour-aware spatial profiling, topology workflows, GMI inference, mechanostress analysis, and optional external workflow bridges.

How does the package use scverse data structures (please describe in a few sentences):

`pyXenium` uses `anndata.AnnData` as its canonical analysis object for downstream RNA+protein and spatial workflows. Core loaders can return `AnnData` directly, and the package's multimodal, contour, pathway, benchmarking, and mechanostress modules operate on `AnnData` inputs or on `XeniumSlide`, a slide-level container whose main table is an `AnnData`. The project also keeps an optional interoperability bridge to `SpatialData` via `XeniumSlide.to_spatialdata()`, but does not require `spatialdata` as a core runtime dependency.

- [x] The code is publicly available under an [OSI-approved](https://opensource.org/licenses/alphabetical) license
- [x] The package provides versioned releases
- [x] The package can be installed from a standard registry (e.g. PyPI, conda-forge, bioconda)
- [x] Automated tests cover essential functions of the package and a reasonable range of inputs and conditions
- [x] Continuous integration (CI) automatically executes these tests on each push or pull request
- [x] The package provides API documentation via a website or README
- [x] The package uses scverse datastructures where appropriate (i.e. AnnData, MuData or SpatialData and their modality-specific extensions)
- [x] I am an author or maintainer of the tool and agree on listing the package on the scverse website

### Recommended

- [ ] Please announce this package on scverse communication channels (zulip, discourse, twitter)
- [ ] Please tag the author(s) these announcements. Handles (e.g. `@scverse_team`) to include are:
  - Zulip:
  - Discourse:
  - Mastodon:
  - Bluesky:
  - Twitter:

- [x] The package provides tutorials (or "vignettes") that help getting users started quickly
- [ ] The package uses the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse).

## Additional notes

- Source: https://github.com/hutaobo/pyXenium
- Documentation: https://pyxenium.readthedocs.io/
- Tutorials: https://pyxenium.readthedocs.io/page/tutorials/index.html
- PyPI: https://pypi.org/project/pyXenium/
- Maintained by: `SPATHO AB`

## Before submitting

- If a publication DOI becomes available, add it to `meta.yaml`.
