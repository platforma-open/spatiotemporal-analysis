# @platforma-open/milaboratories.spatiotemporal-analysis

## 1.2.1

### Patch Changes

- 99eab89: Force categorical metadata (subject, grouping, timepoint) to String at CSV read so polars never infers Float64 from numeric-looking values, and normalize values via JS-compatible canonicalization so backend output (e.g. `5.0`) matches the UI's `--timepoint-order` (`5`).

## 1.2.0

### Minor Changes

- 55a62fb: Modernize the block on BlockModelV3, vectorize the Python pipeline with Polars, and publish PColumn specs before computation completes.

  - **Early spec export.** Every `xsv.importFile` call now passes `splitDataAndSpec: true`; a shared `buildTracedPf` helper wraps each result through `pframes.pFrameBuilder` with trace injection. Downstream blocks (e.g. Lead Selection) can resolve anchored queries and populate their UI the moment specs land, instead of waiting for the full workflow.
  - **Remove `wf.prepare`.** The outer `wf.body` now runs immediately and delegates to a new ephemeral template `process.tpl.tengo` that awaits only `PColumnBundle` state (specs, not data). Output specs publish to the result pool shortly after block start instead of after the full Python analysis — `wf.prepare`'s `Final`-state await previously gated the whole body on upstream data availability, cancelling the benefit of `splitDataAndSpec`.
  - **BlockModelV3 migration.** Replace `BlockModel` V1 with `BlockModelV3` + `DataModelBuilder`, merge `args` and `uiState` into a single `BlockData` type, and keep existing projects working via `upgradeLegacy`. UI switches to `defineAppV3` and reads `app.model.data.*`.
  - **Polars vectorization.** Rewrite `compartment_analysis.py` hot paths — grouping, temporal kinetics, CLR normalization — as Polars expressions, eliminating per-row Python loops.
  - **CID stability for reruns.** Resolve spec gaps 2, 3, 4, 5: the heatmap, temporal-line, and prevalence-histogram PFrames now carry `blockId` in their axis and column domains, matching the main PFrame and preventing content-ID collisions across block instances.
  - **Python test suite.** Add 101 behavioral tests covering grouping restriction, temporal kinetics, cross-subject convergence, CLR normalization, and empty/single-element edge cases — 77 % coverage on `compartment_analysis.py`. Shared fixtures, strengthened assertions.
  - **UI polish.** Reorder sidebar to surface subject prevalence first, hide graph tabs whose driving variable is unset, include the subject name in chart subtitles, and guard conditional output resolves.
  - **Discrete-filter derivation.** Prefer the upstream `pl7.app/discreteValues` annotation on the grouping column (new samples-and-data attaches it) — this fast path requires no data scan and preserves early spec export. For legacy upstreams that lack the annotation, fall back to scanning the grouping column via `col.data.getDataAsJson()`; this blocks the inner template on data availability and delays spec publication for those projects, but keeps the dropdown working. Peak Timepoint uses `args.timepointOrder` directly. Lead Selection renders dropdowns listing the actual categories (e.g. `spleen`, `lung`, `mediastinal lymph node`) instead of free-text fields.

### Patch Changes

- Updated dependencies [55a62fb]
  - @platforma-open/milaboratories.spatiotemporal-analysis.workflow@0.3.0
  - @platforma-open/milaboratories.spatiotemporal-analysis.model@0.2.0
  - @platforma-open/milaboratories.spatiotemporal-analysis.ui@0.2.0

## 1.1.2

### Patch Changes

- 56e90e2: Fix rankingOrder annotations: use standard "decreasing"/"increasing" instead of non-standard "descending"/"ascending"

## 1.1.1

### Patch Changes

- 7960302: Guarantee deterministic CSV output ordering for backend deduplication

## 1.1.0

### Minor Changes

- 5c9a004: Rename block to Clonotype Distribution; hide Subject Prevalence when no subject variable; add tooltips to Grouping/Temporal dropdowns; default Min Subject Count to 1

## 1.0.3

### Patch Changes

- 5434b0e: Remove pseudo-count parameter: fold-changes use only detected timepoints where frequency is always nonzero

## 1.0.2

### Patch Changes

- Updated dependencies [72b5956]
  - @platforma-open/milaboratories.spatiotemporal-analysis.workflow@0.2.0

## 1.0.1

### Patch Changes

- 3222205: Release

## 1.0.0

### Major Changes

- f2de6b0: Release

### Minor Changes

- b738ac6: Initial release of Spatiotemporal Analysis block.

  Profiles clonal abundance across grouping variables (tissue compartments), temporal variables (timepoints), and subjects. Computes restriction index, dominant group, breadth, Log2 Peak Delta, Temporal Shift Index, Log2 Kinetic Delta, subject prevalence, and cross-subject convergence metrics. Supports population-level and intra-subject calculation modes with relative frequency and CLR normalization.
