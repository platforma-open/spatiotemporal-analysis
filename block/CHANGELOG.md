# @platforma-open/milaboratories.spatiotemporal-analysis

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
