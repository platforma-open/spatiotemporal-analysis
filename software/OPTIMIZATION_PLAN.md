# Performance Optimization Plan — compartment_analysis.py

Spec: `docs/text/work/projects/in-vivo-compartment-analysis/README.md`
Source: `software/src/compartment_analysis.py`

## Test Review Findings

All 101 tests pass. No test is incorrect. Issues found:

### Issue 1: Consensus Dominant Tie-Breaking Deviates From Spec (R18)
- **Spec:** ties broken by highest mean frequency across tied subjects
- **Implementation (`_consensus_dominant`):** ties broken alphabetically
- **Test (`test_grouping_modes.py:119`):** documents current alphabetical behavior
- **TESTING_PLAN.md** gap #4 already flagged this
- **Action needed:** decide whether to fix implementation or update spec

### Issue 2: Pooled Dominant Tie-Breaking Is Fragile (R12)
- **`_compute_pooled_grouping` (line 253):** uses `np.argmax(freq_arr)` which returns first index — happens to be alphabetical because `categories` is sorted, but not explicit
- **`_compute_per_subject_grouping` (line 286-289):** explicitly finds tied categories and picks `sorted(tied)[0]` — correct
- **Action needed:** add explicit alphabetical tie-breaking to pooled path

### Issue 3: CLR Intra-Subject Test Doesn't Verify Different Behavior
- **`test_normalization.py:137-151`:** uses 1 sample per subject, making population and intra-subject CLR produce identical results
- **Action needed:** use a scenario with different zero patterns per subject

### Issue 4: End-to-End Tests Missing Several Planned Tests
- Missing: `test_full_pipeline_intra_subject_mode` (R1, R3)
- Missing: `test_grouping_only_no_temporal` (R5a, R32)
- Missing: `test_temporal_only_no_grouping` (R5a, R32)
- Missing: `test_no_subject_skips_convergence` (R5)
- `_run_pipeline` hardcodes `"population"` mode — no end-to-end intra-subject coverage

### Issue 5: Mean RI Test Could Be More Precise
- **`test_grouping_modes.py:171-182`:** only checks `0.0 < meanRi < 1.0`, not exact arithmetic mean

---

## Current Architecture — Why It's Slow

Every metric function follows this pattern:
```python
for element_id, group in df.group_by("elementId"):
    freq_map = dict(zip(group[COL].to_list(), group["freq"].to_list()))
    freq_arr = np.array([freq_map.get(cat, 0.0) for cat in categories])
    result = some_formula(freq_arr)
    results.append({"elementId": eid, "metric": result})
return pl.DataFrame(results)
```

O(N) Python iterations where N = number of clonotypes (10K-100K+). Each iteration crosses the Python-Polars boundary, builds dicts, and calls numpy on tiny arrays (3-10 elements).

### Bottleneck Locations

| Location | Lines | Pattern | Scale |
|----------|-------|---------|-------|
| `_compute_pooled_grouping` | 247-268 | `group_by("elementId")` loop | N elements |
| `_compute_per_subject_grouping` | 278-304 | `group_by(["elementId", subject])` loop | N × S |
| `compute_grouping_metrics` | 198-236 | `group_by("elementId")` loop over per-subject results | N elements |
| `compute_temporal_metrics` (population) | 344-352 | `group_by("elementId")` loop | N elements |
| `compute_temporal_metrics` (intra-subject) | 361-391 | Two nested `group_by` loops | N × S + N |
| `build_temporal_line_data` | 515-532 | `group_by("elementId")` scoring loop | N elements |
| `_apply_clr_to_group` | 138-149 | `group_by("sampleId").map_groups()` | S samples |

---

## Optimization Strategy: Pivot + Vectorized Expressions

Replace all Python for-loops with Polars-native operations. Core technique: **pivot** long-form data to wide-form, then compute metrics as **columnar expressions** across all rows simultaneously.

Polars v1.39.3 features verified: `pivot()`, `sum_horizontal()`, `max_horizontal()`, `.log(base=2)`, `.mode()`, `.std(ddof=1)`, `when/then/otherwise` chains.

---

### Priority 1: Grouping Metrics (High Impact)

**Functions:** `_compute_pooled_grouping`, `_compute_per_subject_grouping`, `compute_grouping_metrics`

#### Step 1a: Vectorize `_compute_pooled_grouping`

Pivot to wide format (rows=elementId, columns=categories, values=meanFreq), then compute all metrics as column expressions:

```python
wide = per_grouping.pivot(on=COL_GROUPING, index="elementId", values="meanFreq").fill_null(0.0)
cat_cols = sorted([c for c in wide.columns if c != "elementId"])

# Normalize to proportions
row_sum = pl.sum_horizontal(*[pl.col(c) for c in cat_cols])
# ... p_i columns, then:

# Shannon entropy: H = -sum(p_i * log2(p_i)) for p_i > 0
h_terms = [pl.when(pl.col(f"p_{c}") > 0)
             .then(-pl.col(f"p_{c}") * pl.col(f"p_{c}").log(base=2))
             .otherwise(0.0) for c in cat_cols]
H = pl.sum_horizontal(*h_terms)

# RI = 1 - H/log2(N) with edge cases
n_nonzero = pl.sum_horizontal(*[pl.when(pl.col(c) > 0).then(1).otherwise(0) for c in cat_cols])
ri = (pl.when(n_nonzero == 0).then(float("nan"))
        .when(n_nonzero == 1).then(1.0)
        .otherwise(1.0 - H / n_nonzero.cast(pl.Float64).log(base=2)))

# Dominant: when/then chain in reverse-alpha order (last applied = first evaluated)
max_val = pl.max_horizontal(*[pl.col(c) for c in cat_cols])
dominant_expr = pl.lit(None).cast(pl.String)
for c in sorted(cat_cols, reverse=True):
    dominant_expr = pl.when(pl.col(c) == max_val).then(pl.lit(c)).otherwise(dominant_expr)

# Breadth: count above threshold
breadth = pl.sum_horizontal(*[pl.when(pl.col(c) > threshold).then(1).otherwise(0) for c in cat_cols])
```

**Edge cases:** `fill_null(0.0)` handles absent categories (R31). N=0→NaN, N=1→1.0 per spec.

#### Step 1b: Vectorize `_compute_per_subject_grouping`

Same pattern with `index=["elementId", COL_SUBJECT]`.

#### Step 1c: Vectorize `compute_grouping_metrics` Aggregation

Replace element loop with `group_by("elementId").agg(...)`:

```python
per_subject_metrics.group_by("elementId").agg(
    pl.col("ri").mean().alias("meanRi"),
    pl.col("ri").std(ddof=1).alias("stdRi"),
    pl.col("dominant").mode().sort().first().alias("consensusDominant"),
    pl.col("breadth").mean().round(0).cast(pl.Int32).alias("breadth"),
    pl.len().alias("nSubjects"),
    *[pl.col("dominant").eq(cat).sum().alias(f"countDominantIn_{cat}") for cat in categories],
)
```

Then apply `minSubjectCount` threshold via `with_columns(when/then)`.

---

### Priority 2: Temporal Metrics (High Impact)

**Functions:** `compute_temporal_metrics`, `_compute_temporal_for_element`

Pivot to wide format (rows=elementId, columns=timepoints), compute all metrics:

```python
wide = per_tp.pivot(on=COL_TIMEPOINT, index="elementId", values="meanFreq").fill_null(0.0)
tp_cols = [c for c in timepoint_order if c in wide.columns]

# Peak: when/then chain (reversed tp_cols so first timepoint wins ties)
max_freq = pl.max_horizontal(*[pl.col(tp) for tp in tp_cols])
peak_expr = pl.lit(None).cast(pl.String)
for tp in reversed(tp_cols):
    peak_expr = pl.when(pl.col(tp) == max_freq).then(pl.lit(tp)).otherwise(peak_expr)

# TSI: weighted mean of positions
total_freq = pl.sum_horizontal(*[pl.col(tp) for tp in tp_cols])
weighted_sum = pl.sum_horizontal(*[pl.col(tp) * float(i) for i, tp in enumerate(tp_cols)])
tsi = pl.when(total_freq > 0).then(weighted_sum / (total_freq * float(len(tp_cols) - 1))).otherwise(float("nan"))

# First/last detected indices via when/then scan
first_idx = pl.lit(None).cast(pl.Int32)
for i, tp in enumerate(tp_cols):
    first_idx = pl.when(first_idx.is_null() & (pl.col(tp) > 0)).then(i).otherwise(first_idx)
last_idx = pl.lit(None).cast(pl.Int32)
for i, tp in enumerate(tp_cols):
    last_idx = pl.when(pl.col(tp) > 0).then(i).otherwise(last_idx)

# Extract freq at computed index via when/then
first_freq = pl.lit(0.0)
for i, tp in enumerate(tp_cols):
    first_freq = pl.when(first_idx == i).then(pl.col(tp)).otherwise(first_freq)
# Same for peak_freq, last_freq

# Fold changes
n_detected = pl.sum_horizontal(*[pl.when(pl.col(tp) > 0).then(1).otherwise(0) for tp in tp_cols])
log2pd = pl.when(n_detected <= 1).then(0.0).otherwise((peak_freq / first_freq).log(base=2))
log2kd = pl.when(n_detected <= 1).then(0.0).otherwise((last_freq / first_freq).log(base=2))
```

Intra-subject: same pivot with `index=["elementId", COL_SUBJECT]`, then `group_by("elementId").agg(mean, mode)`.

---

### Priority 3: CLR Transform (Medium Impact)

**Functions:** `compute_clr`, `_apply_clr_to_group`

Replace `map_groups(python_func)` with window expressions:

```python
# Zero replacement
delta = 0.65 * pl.col("min_nz") / pl.col("D")
df = df.with_columns(
    pl.when(pl.col("frequency") == 0).then(delta).otherwise(pl.col("frequency")).alias("freq_replaced")
)

# Renormalize per sample
df = df.with_columns(
    (pl.col("freq_replaced") / pl.col("freq_replaced").sum().over("sampleId")).alias("freq_norm")
)

# CLR = log(freq_norm) - mean(log(freq_norm)) per sample
df = df.with_columns(
    (pl.col("freq_norm").log() - pl.col("freq_norm").log().mean().over("sampleId")).alias("frequency")
)
```

Scoping: global min_nz/D for population mode, per-subject for intra-subject mode (via group_by + join).

---

### Priority 4: Temporal Line Scoring (Low Impact)

**Function:** `build_temporal_line_data` (lines 515-532)

Same pivot pattern as Priority 2. Only scores subset of data, so lower impact.

---

## Implementation Order

| Step | What | Difficulty | Impact |
|------|------|------------|--------|
| 1a | Vectorize `_compute_pooled_grouping` | Medium | High |
| 1b | Vectorize `_compute_per_subject_grouping` | Low | High |
| 1c | Vectorize `compute_grouping_metrics` aggregation | Low | Medium |
| 2a | Vectorize population temporal | Medium-High | High |
| 2b | Vectorize intra-subject temporal | Medium | High |
| 3 | Vectorize CLR | Low-Medium | Medium |
| 4 | Vectorize temporal line scoring | Low | Low |
| 5 | Fix pooled dominant tie-breaking (Issue 2) | Trivial | Correctness |

## Verification

After each step:
1. `uv run pytest tests/ -v` — all 101 tests must pass
2. `uv run pytest --cov=compartment_analysis --cov-report=term-missing tests/`

After all steps:
- Synthetic benchmark: 100K elements, 5 groups, 4 timepoints, 10 subjects
- Edge cases: single group, single timepoint, all-zero elements, NaN handling

## Functions Removable After Optimization

| Function | Absorbed Into |
|----------|---------------|
| `_compute_temporal_for_element()` | Vectorized temporal pivot |
| `_apply_clr_to_group()` | Vectorized CLR window expressions |

Keep `shannon_entropy()` and `restriction_index()` as tested utility functions (imported by tests) but remove from hot path.
