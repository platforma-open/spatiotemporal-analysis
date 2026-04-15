# Performance Optimization Plan — compartment_analysis.py

Spec: `docs/text/work/projects/in-vivo-compartment-analysis/README.md`
Source: `software/src/compartment_analysis.py`

## Status Context

All correctness gaps resolved or intentionally descoped (see `SPEC_GAPS_PLAN.md`):
- **Gap 2** (R3 per-subject detail), **Gap 3** (R7 deselection), **Gap 4** (R18 consensus tie-breaking), **Gap 5** (R12 pooled tie-breaking) — RESOLVED.
- **Gap 1** (R6 multiple grouping variables, R26 disambiguation, R28 tabs) — WON'T FIX.

112 tests pass, block builds clean. This plan now covers **performance only** — the correctness work is done.

### Why Gap 1's descoping simplifies this plan

The earlier draft mentioned that the vectorized pivot approach would "handle [multi-variable] naturally — one pivot per variable." With multi-variable descoped, the plan reduces to a **single pivot per computation**. No outer loop over variables, no result merging, no per-variable domain tags. The remaining bottlenecks are purely the per-clone Python iteration patterns — exactly what vectorization targets.

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

O(N) Python iterations where N = number of clonotypes (10K–100K+). Each iteration crosses the Python-Polars boundary, builds dicts, and calls numpy on tiny arrays (3–10 elements).

### Bottleneck Locations

| Location | Pattern | Scale |
|----------|---------|-------|
| `_compute_pooled_grouping` | `group_by("elementId")` loop | N elements |
| `_compute_per_subject_grouping` | `group_by(["elementId", subject])` loop | N × S |
| `compute_grouping_metrics` aggregation | `group_by("elementId")` loop over per-subject results | N elements |
| `compute_temporal_metrics` (population) | `group_by("elementId")` loop | N elements |
| `compute_temporal_metrics` (intra-subject) | Two nested `group_by` loops | N × S + N |
| `build_temporal_line_data` | `group_by("elementId")` scoring loop | N elements |
| `_apply_clr_to_group` | `group_by("sampleId").map_groups()` | S samples |

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

# Dominant: when/then chain in reverse-alpha order (last applied = first evaluated).
# Gap 5 resolved: the Python version uses explicit sorted(tied)[0]; this when/then
# chain is the vectorized equivalent.
max_val = pl.max_horizontal(*[pl.col(c) for c in cat_cols])
dominant_expr = pl.lit(None).cast(pl.String)
for c in sorted(cat_cols, reverse=True):
    dominant_expr = pl.when(pl.col(c) == max_val).then(pl.lit(c)).otherwise(dominant_expr)

# Breadth: count above threshold
breadth = pl.sum_horizontal(*[pl.when(pl.col(c) > threshold).then(1).otherwise(0) for c in cat_cols])
```

**Edge cases:** `fill_null(0.0)` handles absent categories (R31). N=0→NaN, N=1→1.0 per spec.

#### Step 1b: Vectorize `_compute_per_subject_grouping`

Same pattern with `index=["elementId", COL_SUBJECT]`. The result is the per-subject wide DataFrame — this is **also the `per_subject_out` DataFrame returned by `compute_grouping_metrics` for Gap 2**, so we save the pivot result before aggregating.

#### Step 1c: Vectorize `compute_grouping_metrics` Aggregation

Replace element loop with `group_by("elementId").agg(...)`:

```python
per_subject_metrics.group_by("elementId").agg(
    pl.col("ri").mean().alias("meanRi"),
    pl.col("ri").std(ddof=1).alias("stdRi"),
    # Consensus dominant: highest-mean-frequency among tied-mode groups (R18, Gap 4).
    # `.mode().sort().first()` alone only handles alphabetical ties — for frequency-
    # weighted ties we need a join-back pattern (see below) or a post-agg polars
    # expression that consumes the per-element group means.
    pl.col("breadth").mean().round(0).cast(pl.Int32).alias("breadth"),
    pl.len().alias("nSubjects"),
    *[pl.col("dominant").eq(cat).sum().alias(f"countDominantIn_{cat}") for cat in categories],
)
```

**Consensus dominant vectorization (Gap 4 interaction):**

The current Python implementation pre-computes per-element per-group mean frequencies for tie-breaking. The vectorized equivalent:
1. Compute `dom_counts = group_by([elementId, dominant]).len()` — how often each category is dominant per element.
2. Compute `max_count = dom_counts.select(pl.col("len").max().over("elementId"))`.
3. Filter to tied categories, join with per-element-per-group mean frequency, sort by `(-mean_freq, alpha_name)`, pick first per elementId.

This is more code than `.mode().sort().first()` but preserves the spec-compliant tie-breaking. Alternative: keep consensus dominant in a small Python loop (N elements × ≤|categories| tied items — negligible after the bulk pivot work).

Apply `minSubjectCount` threshold via `with_columns(when/then)`.

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

**Intra-subject path (Gap 2 interaction):** Same pivot with `index=["elementId", COL_SUBJECT]`. The resulting per-subject wide DataFrame **is the `per_subject_out` returned by `compute_temporal_metrics`** — save it before aggregating across subjects. Final aggregation via `group_by("elementId").agg(mean, mode)`.

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

**Scoping:** global `min_nz` / `D` for population mode; per-subject scope for intra-subject mode (via `group_by(subject).agg` + join back, or `.min().over(subject)` window expressions).

---

### Priority 4: Temporal Line Scoring (Low Impact)

**Function:** `build_temporal_line_data`

Same pivot pattern as Priority 2 for the per-clone ranking step. Only scores a subset of clones (for the line plot), so lower overall impact. Can share the pivot with `compute_temporal_metrics` if we refactor to compute both in one pass.

---

## Implementation Order

| Step | What | Difficulty | Impact |
|------|------|------------|--------|
| 1a | Vectorize `_compute_pooled_grouping` | Medium | High |
| 1b | Vectorize `_compute_per_subject_grouping` (also produces Gap 2 per-subject output) | Low | High |
| 1c | Vectorize `compute_grouping_metrics` aggregation | Low–Medium | Medium |
| 2a | Vectorize population temporal | Medium–High | High |
| 2b | Vectorize intra-subject temporal (also produces Gap 2 per-subject output) | Medium | High |
| 3 | Vectorize CLR | Low–Medium | Medium |
| 4 | Vectorize temporal line scoring | Low | Low |

**Note on consensus dominant (Gap 4):** The vectorized aggregation in Step 1c may keep a small Python fallback for the frequency-weighted tie-break if the full Polars expression gets too complex. The per-element work is tiny (just tied categories), so this isn't a bottleneck even if left in Python.

## Verification

After each step:
1. `uv run pytest tests/ -v` — all **112 tests** must pass.
2. `uv run pytest --cov=compartment_analysis --cov-report=term-missing tests/` — coverage should not decrease.
3. Build the block: `pnpm run build:dev` from the block root — must succeed.

After all steps:
- Synthetic benchmark: 100K elements, 5 groups, 4 timepoints, 10 subjects — measure before/after wall time.
- Edge cases to verify: single group, single timepoint, all-zero elements, NaN handling, subject-less mode, intra-subject + population parity where expected.
- Spot-check per-subject CSV output still matches pre-optimization values (Gap 2 regression guard).

## Functions Removable After Optimization

| Function | Absorbed Into |
|----------|---------------|
| `_compute_temporal_for_element()` | Vectorized temporal pivot |
| `_apply_clr_to_group()` | Vectorized CLR window expressions |

Keep `shannon_entropy()` and `restriction_index()` as tested utility functions (imported by `test_restriction_index.py`), but don't call them in the hot path.

Keep `_consensus_dominant()` if we retain the small Python fallback for frequency-weighted tie-breaking (Gap 4); otherwise it can be folded into the vectorized aggregation.
