# Spec Gap Resolution Plan — Spatiotemporal Analysis

Spec: `docs/text/work/projects/in-vivo-compartment-analysis/README.md`
PColumn schema: `docs/text/work/projects/in-vivo-compartment-analysis/pcolumn-schema.md`

The TESTING_PLAN.md identified 5 spec gaps. This plan resolves each one with changes across all block layers (Python, workflow, model, UI) where applicable. Changes are designed to be compatible with the future Polars vectorization in OPTIMIZATION_PLAN.md.

---

## Gap 1: R6 — Multiple Grouping Variables

**Spec:** `groupingVariables: GroupingVariableConfig[]` — array of grouping variables with labels. R26: "Multiple Grouping Variable configurations produce independent sets of columns, disambiguated by the variable name."

**Current:** Single `groupingColumnRef?: SUniversalPColumnId` throughout.

### Resolution: Python Multi-Grouping Support

Run one Python invocation with multiple grouping columns in the input CSV. This avoids wasteful re-reads and is compatible with vectorized optimization (N sequential pivots, or parallel pivots in one pass).

#### Python Changes (`compartment_analysis.py`)

**New args:**
```
--grouping-columns '["grouping_0","grouping_1"]'   # CSV column names
--grouping-labels '["Organism Part","Immunophenotype"]'  # Human-readable labels
```

Replace `--has-grouping` flag. When `--grouping-columns` is empty/absent, no grouping metrics are computed.

**Processing:**
- Loop over `(column_name, label)` pairs
- For each: compute grouping metrics using `column_name` as the grouping column
- Output per-variable files: `result_grouping_0.csv`, `result_heatmap_0.csv`, etc.
- Main table: merge all grouping results, prefix column names with variable index
  - `ri_0`, `dominant_0`, `breadth_0`, `ri_1`, `dominant_1`, `breadth_1`, ...
  - Consensus columns (with subject): `consensusDominant_0`, `meanRi_0`, `stdRi_0`, ...
  - CountDominantIn: `countDominantIn_0_lung`, `countDominantIn_1_CD4`, ...

**Functions affected:**
- `parse_args()` — new `--grouping-columns`, `--grouping-labels` args
- `main()` — loop over grouping variables, merge results
- `compute_grouping_metrics()` — accept column name parameter (currently hardcoded `COL_GROUPING`)
- `build_heatmap_data()` — accept column name parameter

**Performance note:** Each variable is an independent computation. The vectorized pivot approach from OPTIMIZATION_PLAN.md handles this naturally — one pivot per variable. With N variables and M elements, it's N pivots of M rows, each fully vectorized.

#### Workflow Changes (`main.tpl.tengo`)

```go
// Build CSV: add each grouping column with indexed header
for i, ref in args.groupingColumnRefs {
    csvBuilder.add(columns.getColumn(ref), {header: "grouping_" + string(i)})
}

// Pass grouping info to Python
groupingCols := []
groupingLabels := []
for i, ref in args.groupingColumnRefs {
    groupingCols = append(groupingCols, "grouping_" + string(i))
    // Label from metadata spec
    spec := columns.getSpec(ref)
    label := spec.annotations["pl7.app/label"]
    if label == undefined { label = "Grouping " + string(i) }
    groupingLabels = append(groupingLabels, label)
}
runAnalysis = runAnalysis.
    arg("--grouping-columns").arg(string(json.encode(groupingCols))).
    arg("--grouping-labels").arg(string(json.encode(groupingLabels)))
```

**PFrame construction:** For each grouping variable, create PColumn specs with variable-specific domains:
```go
for i, label in groupingLabels {
    varDomain := {
        "pl7.app/vdj/spatiotemporalAnalysis/variable": label,
        "pl7.app/vdj/spatiotemporalAnalysis/blockId": blockId
    }
    // RI column for this variable
    mainImportColumns = append(mainImportColumns, {
        column: "ri_" + string(i),
        spec: { name: "pl7.app/vdj/restrictionIndex", domain: varDomain,
                annotations: {"pl7.app/label": "Restriction Index / " + label, ...} }
    })
    // ... same for dominant, breadth, consensus columns
}
```

**Heatmap PFrames:** One per variable. Save each `result_heatmap_{i}.csv` and import as separate PFrame.

**Output map:** Change from single `heatmapPf` to indexed: `heatmapPf_0`, `heatmapPf_1`, etc. Or use a single `heatmapPf` output key that routes to the active tab's variable.

#### Model Changes (`model/src/index.ts`)

```typescript
// New type
type BlockDataV2 = Omit<BlockData, 'groupingColumnRef'> & {
  groupingColumnRefs: SUniversalPColumnId[];
};

// Migration from v1
const dataModel = new DataModelBuilder()
  .from<BlockDataV2>('v2')
  .upgrade<BlockData>('v1', (v1) => ({
    ...v1,
    groupingColumnRefs: v1.groupingColumnRef ? [v1.groupingColumnRef] : [],
  }))
  ...
```

**Validation:** At least one of `groupingColumnRefs.length > 0` or `temporalColumnRef` required.

**Outputs:** Dynamic heatmap outputs based on `groupingColumnRefs.length`.

#### UI Changes (`MainPage.vue`)

- Replace single `PlDropdown` for grouping with a multi-select or add/remove list
- Pattern: "Add grouping variable" button → dropdown to select metadata column → adds to list
- Each list item shows the column label and a remove button
- Heatmap page: tabs per grouping variable (R28)

#### Test Changes

- Add tests for multi-variable grouping: 2 variables produce independent RI/dominant/breadth columns
- Verify column naming: `ri_0`, `dominant_0`, etc.
- Verify domain disambiguation per variable
- Verify heatmap data generated per variable

---

## Gap 2: R3 — Per-Subject Detail Columns — RESOLVED

**Spec (R3):** "The block outputs two sets of columns: (a) per-subject detail columns with axes `[elementId, subjectId]` for inspection in Table block, and (b) aggregated per-clone summary columns with axes `[elementId]` for Lead Selection."

**Status:** Resolved. `compute_grouping_metrics` and `compute_temporal_metrics` now return tuples `(aggregated, per_subject)`. `main()` writes `result_per_subject.csv` when `mode == "intra-subject"` and subject is present. Workflow builds `perSubjectPf` with axes `[elementId, subject]`; model exposes `perSubjectPf` + `perSubjectPCols` outputs conditional on intra-subject mode with subject variable.

### Resolution: Export Per-Subject Intermediate Data

The per-subject data already exists inside `compute_grouping_metrics()` (from `_compute_per_subject_grouping`) and `compute_temporal_metrics()` (intra-subject path). Export it before aggregation.

#### Python Changes

**New output file:** `result_per_subject.csv`

Columns: `elementId, subject, ri, dominant, breadth, [peakTimepoint, temporalShiftIndex, log2PeakDelta, log2KineticDelta]`

The temporal columns are only present when temporal variable is configured.

With multiple grouping variables (Gap 1): `ri_0, dominant_0, breadth_0, ri_1, dominant_1, breadth_1, ...`

**Changes to `compute_grouping_metrics()`:**
- Return both the aggregated DataFrame AND the per-subject DataFrame
- Currently `_compute_per_subject_grouping` already returns per-subject data — just save it

```python
def compute_grouping_metrics(...) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Returns (aggregated_metrics, per_subject_metrics)."""
    ...
    per_subject_metrics = _compute_per_subject_grouping(...)
    # Aggregate per_subject_metrics into summary (existing logic)
    aggregated = ...
    return aggregated, per_subject_metrics
```

**Changes to `compute_temporal_metrics()`:**
- In intra-subject mode, return both per-subject and aggregated DataFrames
- The `ps_df` variable (line 371) already holds per-subject temporal metrics — save it

**Changes to `main()`:**
- Collect per-subject DataFrames from grouping and temporal
- Join them on `[elementId, subject]`
- Write to `result_per_subject.csv`
- Only in intra-subject mode (population mode aggregates differently)

#### Workflow Changes

**New PFrame:** `perSubjectPf` with 2-axis import.

```go
if mode == "intra-subject" && hasSubject {
    perSubjectCsv := runAnalysis.getFile("result_per_subject.csv")

    // Subject axis: reuse original metadata column's spec
    subjectSpec := columns.getSpec(args.subjectColumnRef)
    subjectAxisSpec := {
        name: subjectSpec.axesSpec[0].name,  // or subjectSpec.name
        type: "String",
        domain: subjectSpec.domain,
        annotations: {
            "pl7.app/label": subjectSpec.annotations["pl7.app/label"],
            "pl7.app/vdj/spatiotemporalAnalysis/isSubjectAxis": "true"
        }
    }

    perSubjectPf = xsv.importFile(perSubjectCsv, "csv", {
        axes: [
            {column: "elementId", spec: elementAxisSpec},
            {column: "subject", spec: subjectAxisSpec}
        ],
        columns: [
            // Per-subject RI (with "per-subject" level domain)
            {column: "ri", spec: {
                name: "pl7.app/vdj/restrictionIndex", valueType: "Double",
                domain: {
                    "pl7.app/vdj/spatiotemporalAnalysis/variable": varLabel,
                    "pl7.app/vdj/spatiotemporalAnalysis/blockId": blockId,
                    "pl7.app/vdj/spatiotemporalAnalysis/level": "per-subject"
                },
                annotations: {"pl7.app/label": "RI (per subject)", "pl7.app/table/visibility": "optional"}
            }},
            // ... dominant, log2PeakDelta, log2KineticDelta, peakTimepoint
        ]
    })
}
```

**Condition:** Only generated in intra-subject mode with subject variable.

**Output:** Add `perSubjectPf` to workflow outputs and model.

#### Model Changes

- Add `perSubjectPf` output (conditional on intra-subject mode + subject variable)
- No UI changes needed — users inspect per-subject data in the Table block via Result Pool

#### Test Changes

- New test: verify `result_per_subject.csv` is written in intra-subject mode
- Verify it has `[elementId, subject]` columns + metric columns
- Verify per-subject RI values match expected hand-calculated values
- Verify temporal per-subject metrics present when temporal variable configured
- Verify NOT generated in population mode

---

## Gap 3: R7 — Timepoint Deselection — RESOLVED

**Spec (R7):** "Users can both reorder and deselect timepoints. Deselected timepoints exclude corresponding samples from all temporal computations."

**Status:** Resolved. The mechanism was already implemented across all layers (UI draggable list, model `timepointOrder`, workflow passes JSON, Python filters via `is_in(timepoint_order)` at `compartment_analysis.py:370` and `:537`). Added explicit test `test_deselected_timepoints_excluded` in `test_temporal_metrics.py` verifying Day7 data is excluded when only Day0 and Day14 are selected. Existing `test_out_of_order_timepoints_filtered` covers `build_temporal_line_data` path.

### Verification

The mechanism is complete across all layers:

1. **UI (`MainPage.vue:240`):** `PlElementList v-model:items="app.model.data.timepointOrder"` — draggable list that supports item removal. Removed items appear in `availableTimepointsToAdd` (line 105) and can be restored via "Reset to default" (line 110).

2. **Model (`index.ts:24`):** `timepointOrder: string[]` stores only selected timepoints in user-defined order.

3. **Workflow (`main.tpl.tengo:71`):** Passes order as JSON: `--timepoint-order '["Day0","Day7"]'`. Deselected timepoints are simply absent.

4. **Python (`compartment_analysis.py:336`):** `df = df.filter(pl.col(COL_TIMEPOINT).is_in(timepoint_order))` — excludes samples at deselected timepoints.

5. **Validation (`index.ts:95`):** `timepointOrder.length >= 2` required for temporal analysis — handles edge case of too few selected timepoints.

### Action: Close gap. Add one test to explicitly verify deselection behavior:

```python
# test_preprocessing.py or test_temporal_metrics.py
def test_deselected_timepoints_excluded():
    """R7: timepoints not in order are excluded from temporal computation."""
    df = pl.DataFrame({
        "sampleId": ["s1", "s2", "s3"],
        "elementId": ["a", "a", "a"],
        "frequency": [0.2, 0.5, 0.8],
        "timepoint": ["Day0", "Day7", "Day14"],
    })
    # Only Day0 and Day14 selected (Day7 deselected)
    result = compute_temporal_metrics(df, ["Day0", "Day14"], ...)
    # Day7 data should not participate in computation
    row = result.row(0, named=True)
    assert row["peakTimepoint"] == "Day14"
    # Log2KD = log2(0.8/0.2) — Day7 is absent
    assert row["log2KineticDelta"] == pytest.approx(math.log2(0.8 / 0.2))
```

---

## Gap 4: R18 — Consensus Dominant Tie-Breaking — RESOLVED

**Spec (R18):** "In case of tie, reports the group with the highest mean frequency across tied subjects."

**Status:** Resolved via Option A (implementation fixed to match spec).

### Resolution (Option A): Fix Implementation

#### Python Changes

**`_consensus_dominant` signature change:**
```python
def _consensus_dominant(
    dominants: list[str | None],
    group_mean_freqs: dict[str, float] | None = None,
) -> str | None:
    """Mode of dominants. Ties broken by highest mean frequency, then alphabetically."""
    dom_counts: dict[str, int] = {}
    for d in dominants:
        if d is not None:
            dom_counts[d] = dom_counts.get(d, 0) + 1
    if not dom_counts:
        return None
    max_count = max(dom_counts.values())
    tied = [k for k, v in dom_counts.items() if v == max_count]
    if len(tied) == 1:
        return tied[0]
    # Tie-break: highest mean frequency, then alphabetical
    if group_mean_freqs:
        tied.sort(key=lambda g: (-group_mean_freqs.get(g, 0.0), g))
        return tied[0]
    return sorted(tied)[0]
```

**Caller change in `compute_grouping_metrics`:**
```python
# Compute mean frequency per group across all subjects for this element
group_freqs: dict[str, list[float]] = {cat: [] for cat in categories}
for _, subj_row in group.iter_rows():  # or via Polars expressions
    dominant = subj_row["dominant"]
    # Accumulate per-group frequencies from per_subject_grouping data
    ...
group_mean_freqs = {cat: np.mean(freqs) if freqs else 0.0 for cat, freqs in group_freqs.items()}
consensus = _consensus_dominant(dominants, group_mean_freqs)
```

**Performance note for vectorized version:** In the Polars-native optimization, this can be done via a custom aggregation:
```python
# After computing per-subject dominant, for ties:
# Join back per-subject group frequencies, compute mean per group per element,
# then pick the tied group with highest mean frequency
```
This is more complex than `.mode().sort().first()` but feasible with additional join + when/then logic.

#### Test Changes

- Update `test_tie_broken_alphabetically` in `test_grouping_modes.py` to verify frequency-based tie-breaking
- Add new test: two groups with equal dominant counts but different mean frequencies → higher frequency wins
- Add new test: two groups with equal dominant counts AND equal mean frequencies → alphabetical wins (final fallback)

---

## Gap 5: Pooled Dominant Tie-Breaking (R12) — RESOLVED

**Spec (R12):** "Ties broken by alphabetical order of category name (deterministic)."

**Status:** Resolved. `_compute_pooled_grouping` now uses explicit alphabetical tie-breaking (same pattern as `_compute_per_subject_grouping`), no longer dependent on category list ordering.

### Resolution: Explicit Alphabetical Tie-Breaking

#### Python Changes

**In `_compute_pooled_grouping` (line 252-254):**

Replace:
```python
dominant_idx = int(np.argmax(freq_arr))
dominant = categories[dominant_idx] if freq_arr[dominant_idx] > 0 else None
```

With:
```python
max_freq = float(freq_arr.max())
if max_freq > 0:
    tied = [categories[i] for i in range(len(categories)) if freq_arr[i] == max_freq]
    dominant = sorted(tied)[0]
else:
    dominant = None
```

This matches the pattern already used in `_compute_per_subject_grouping` (line 286-289).

**Performance note:** In the vectorized version, this is handled by the when/then chain built in reverse-alphabetical order (see OPTIMIZATION_PLAN.md), which inherently picks the alphabetically first tied category.

#### Test Changes

- Existing `test_dominant_tie_alphabetical` already covers this behavior
- Add a comment documenting that this now uses explicit alphabetical tie-breaking (not implicit via sorted categories + argmax)

---

## Implementation Order

These gaps have dependencies and should be resolved in this order:

| Order | Gap | Scope | Effort | Dependency |
|-------|-----|-------|--------|------------|
| 1 | **Gap 5** — Pooled tie-breaking | Python only | Trivial (5 lines) | None |
| 2 | **Gap 4** — Consensus tie-breaking | Python + tests | Small (1 function + tests) | None |
| 3 | **Gap 3** — Timepoint deselection | Tests only | Trivial (1 test) | None |
| 4 | **Gap 2** — Per-subject detail columns | Python + workflow + model | Medium | None |
| 5 | **Gap 1** — Multiple grouping variables | All layers | Large | Gaps 2, 4, 5 done first |

### Rationale

- **Gaps 5, 4, 3** are pure correctness fixes in Python. Do them first — they're small, independent, and improve the foundation before larger changes.
- **Gap 2** adds a new output file and PFrame. Do it before Gap 1 because Gap 1's multi-variable support needs to work with per-subject detail columns too.
- **Gap 1** is the largest change spanning all layers. Do it last after the foundation is solid.

### Relationship to Performance Optimization

All gap fixes should use the current Python loop-based architecture (not vectorized). The performance optimization in OPTIMIZATION_PLAN.md is a separate pass applied afterward. The gap fixes are designed to be compatible:

- **Gap 1 (multi-grouping):** Vectorized version processes each variable as a separate pivot — trivially parallelizable.
- **Gap 2 (per-subject detail):** Vectorized version produces per-subject wide DataFrame as intermediate result — just save it before aggregating.
- **Gap 4 (consensus tie-breaking):** Vectorized version can use `group_by.agg()` with a custom expression chain instead of `.mode().sort().first()`.
- **Gap 5 (pooled tie-breaking):** Vectorized version's when/then chain in reverse-alpha order handles this automatically.

---

## Files Modified Per Gap

| File | Gap 1 | Gap 2 | Gap 3 | Gap 4 | Gap 5 |
|------|-------|-------|-------|-------|-------|
| `software/src/compartment_analysis.py` | Yes | Yes | — | Yes | Yes |
| `workflow/src/main.tpl.tengo` | Yes | Yes | — | — | — |
| `model/src/index.ts` | Yes | Yes | — | — | — |
| `ui/src/pages/MainPage.vue` | Yes | — | — | — | — |
| `ui/src/pages/HeatmapPage.vue` | Yes | — | — | — | — |
| `software/tests/test_grouping_modes.py` | Yes | — | — | Yes | Yes |
| `software/tests/test_temporal_metrics.py` | — | — | Yes | — | — |
| `software/tests/test_end_to_end.py` | Yes | Yes | — | — | — |
