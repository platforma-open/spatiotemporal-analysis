# Spatiotemporal Analysis Block -- Review Report

**Date:** 2026-04-09
**Spec:** `docs/text/work/projects/in-vivo-compartment-analysis/`
**Block:** `blocks/spatiotemporal-analysis/`

---

## Executive Summary

The block implements the core computation pipeline (spatial restriction, temporal kinetics, consensus metrics) with correct formulas for single-variable use cases. The architecture is sound: Python engine + Tengo workflow + TypeScript model + Vue UI following standard block patterns.

However, **three systemic gaps** span all layers and represent the largest delta between spec and implementation:

1. **Multiple grouping variables** -- spec requires an array of independent grouping variables; all layers implement exactly one
2. **Per-subject detail columns** -- spec requires `[elementId, subjectId]` axes output in intra-subject mode; not implemented anywhere
3. **`countDominantIn_*` dynamic columns** -- Python computes them, but workflow never imports them (silently dropped)

These are not bugs in the existing code -- the code that exists is largely correct. They are **unimplemented spec requirements** that need to be built.

---

## Issues by Severity

### Critical (blocks correctness or completeness)

| # | Component | Issue | Details |
|---|-----------|-------|---------|
| C1 | Python + Workflow | **Per-subject detail columns not output** | Spec R3 requires per-subject columns with axes `[elementId, subjectId]` in intra-subject mode. Python computes per-subject metrics internally but discards them. Workflow has no import specs for them. |
| C2 | All layers | **Multiple grouping variables not supported** | Spec R6 requires multiple independent grouping variables. Python accepts one `grouping` column. Model has `groupingColumnRef` (singular). UI has one dropdown. Workflow passes one column. |
| C3 | Workflow | **`countDominantIn_*` columns silently dropped** | Python outputs dynamic `countDominantIn_{category}` columns per R19, but workflow XSV import has no specs for them. They are lost during import. Requires dynamic column discovery or metadata passing from Python. |

### High (spec violations that affect output correctness)

| # | Component | Issue | Location |
|---|-----------|-------|----------|
| H1 | Model | **Missing `elementLevel` field** | `model/src/index.ts` -- no `'clonotype' | 'cluster'` field per spec. Block cannot distinguish element levels. |
| H2 | Model | **`minSubjectCount` default is 1, spec says 2** | `model/src/index.ts:66`. Spec R17b says default 2 to prevent noisy single-subject averages. Default of 1 disables the guard entirely. |
| H3 | Python | **Null subject values not filtered** | `compartment_analysis.py:62`. Missing/null subject metadata not excluded. Can count null as a distinct subject in prevalence, corrupting consensus metrics. |
| H4 | UI | **No missing metadata warnings** | `MainPage.vue` -- spec R7b requires UI warning listing excluded samples. No warning component exists. |

### Medium (spec deviations that affect naming, schema, or UX)

| # | Component | Issue | Location |
|---|-----------|-------|----------|
| M1 | Workflow | **Domain key namespace wrong** | Lines 115, 167-170, 271: uses `spatiotemporalAnalysis` instead of spec's `compartmentAnalysis`. Will cause Result Pool query mismatches with Lead Selection. |
| M2 | Workflow | **PColumn name mismatch** | Line 175: `pl7.app/vdj/restrictionIndex` vs spec `pl7.app/vdj/compartmentRestrictionIndex`. |
| M3 | Workflow | **Log2 Kinetic Delta visibility wrong** | Line 329: `"optional"` vs spec `"default"`. |
| M4 | Model | **Temporal variable structure differs from spec** | `model/src/index.ts:29-30`. Split into `temporalColumnRef` + `timepointOrder` instead of single `TemporalVariableConfig` object. |
| M5 | Model | **Title is static** | `model/src/index.ts:226`. Hardcoded "Clonotype Distribution" instead of dynamic per spec. |
| M6 | Model | **Missing `GroupingVariableConfig` / `TemporalVariableConfig` types** | These spec-defined types don't exist in the model. |
| M7 | UI | **No tabs/facets for multiple grouping variables in heatmap** | `HeatmapPage.vue` -- single `GraphMaker` with no switching mechanism. Blocked by C2 but needs design even after C2 is fixed. |
| M8 | UI | **No per-item timepoint deselect** | `MainPage.vue:237-250`. Only bulk reset available. Spec R7 requires individual deselection. |
| M9 | Python | **Log2 Peak Delta not clamped to >= 0** | `compartment_analysis.py:470`. With CLR normalization, values could theoretically go negative. Spec says "always >= 0". |
| M10 | Python | **Single-sample subjects not excluded from RI mean** | `compartment_analysis.py:252-276`. Spec R33: subjects with one sample should have undefined RI and be excluded from mean. |

### Low (cosmetic, naming, minor deviations)

| # | Component | Issue |
|---|-----------|-------|
| L1 | Workflow | Label "Breadth" vs spec "Breadth / {variableName}" (line 211) |
| L2 | Workflow | `splitDataAndSpec: true` not used in XSV imports |
| L3 | Workflow | Hardcoded axis index assumption (sampleId=0, elementId=1) at lines 45-46 |
| L4 | Workflow | No guard against missing output files from Python |
| L5 | Model | `subjectVariable` renamed to `subjectColumnRef` |
| L6 | Model | Output named `abundanceOptions` / `mainTable` instead of `datasetOptions` / `displayTable` |
| L7 | Model | Prevalence section conditionally hidden (may be intentional UX) |
| L8 | UI | Temporal line chart uses `scatterplot` type -- verify GraphMaker renders lines |
| L9 | Packaging | Block title "Clonotype Distribution" vs repo name "spatiotemporal-analysis" -- verify intentional |
| L10 | Packaging | `pyproject.toml` lists runtime deps under dev group |
| L11 | Python | No error handling for malformed input files |

---

## Test Coverage Gaps

The Python unit tests (`software/tests/test_compartment_analysis.py`, ~1180 lines) have good structural coverage but these gaps exist:

| Gap | Severity | Details |
|-----|----------|---------|
| **TSI exact formula value** | Medium | Only range [0,1] checked; no test pins to a manually calculated expected value |
| **Mean RI / StdDev exact values** | Medium | Only NaN threshold tested; no exact numerical assertion |
| **CLR transform correctness** | Medium | Only verifies it runs without error; doesn't check transformed values |
| **Replicate averaging correctness** | Medium | Only checks process completes; doesn't verify averaged frequency value |
| **Per-subject detail columns** | High | Not covered (not implemented in source either) |
| **Multiple grouping variables** | High | Not covered (not implemented in source either) |
| **Single-timepoint TSI = rank/(T-1)** | Low | Correct in code but untested |
| **Zero-total sample exclusion** | Low | Assertion only checks element presence, not exclusion mechanism |

The integration test (`test/src/wf.test.ts`) covers 7 configurations and validates output existence and basic properties. It is well-structured.

---

## What Works Well

- **Core formulas are correct**: RI, TSI, Log2PD, Log2KD, CLR normalization with multiplicative replacement -- all mathematically verified against spec
- **Sample averaging logic**: Correctly averages duplicate condition combinations before computation
- **Conditional output**: Temporal metrics correctly gated on >= 2 timepoints; consensus gated on subject variable
- **minAbundanceThreshold filtering**: Correctly excludes clones below threshold in all samples
- **Timepoint ordering**: Respects user-defined order, not alphabetical
- **Visualization PFrames**: Heatmap, temporal line, and prevalence histogram structures match spec
- **Provenance trace**: Correctly injected into main PFrame columns
- **Block packaging**: Well-structured, consistent SDK versions, correct build pipeline
- **Integration test**: 7 configurations covering main code paths

---

## Recommended Fix Priority

### Phase 1: Correctness fixes (no new features)

1. **H2**: Change `minSubjectCount` default from 1 to 2 in model
2. **H3**: Add null-subject filtering in Python before consensus computation
3. **M1**: Fix domain key namespace to `compartmentAnalysis`
4. **M2**: Fix PColumn name to `compartmentRestrictionIndex`
5. **M3**: Fix Log2 Kinetic Delta visibility to `"default"`
6. **M9**: Clamp Log2 Peak Delta to >= 0
7. **M10**: Exclude single-sample subjects from RI mean
8. **L1**: Fix Breadth label

### Phase 2: Missing features (spec gaps)

1. **C2**: Multiple grouping variables -- requires changes to all layers (model array type, UI add/remove, workflow loop, Python multi-column support)
2. **C1**: Per-subject detail columns -- requires Python to output per-subject CSV, workflow to import with [elementId, subjectId] axes
3. **C3**: Dynamic `countDominantIn_*` column import -- requires Python to output category list metadata or workflow to discover column names dynamically

### Phase 3: UX polish

1. **H4**: Add missing metadata warnings to UI
2. **M8**: Per-item timepoint deselect in UI
3. **M7**: Heatmap tabs/facets (blocked by C2)
4. **H1**: Add `elementLevel` field to model
5. **M5**: Dynamic title

### Phase 4: Test hardening

1. Pin exact TSI formula values in tests
2. Pin exact Mean RI / StdDev values
3. Verify CLR transformed values
4. Verify replicate averaging values
5. Add coverage for new features (multi-variable, per-subject detail)

---

## Domain/Naming Decision Needed

The spec uses `compartmentAnalysis` as the domain namespace, but the block is named `spatiotemporal-analysis`. The workflow currently uses `spatiotemporalAnalysis`. A decision is needed:

- **Option A**: Align everything to spec's `compartmentAnalysis` -- ensures Lead Selection integration works as specified
- **Option B**: Update spec to `spatiotemporalAnalysis` -- matches the actual block name

This affects PColumn names, domain keys, and any downstream block that queries for these columns. Must be decided before M1-M2 fixes.
