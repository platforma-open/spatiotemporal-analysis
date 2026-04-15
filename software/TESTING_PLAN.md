# Python Testing Plan — compartment_analysis.py

Spec: `docs/text/work/projects/in-vivo-compartment-analysis/README.md`

## Approach

All tests are **behavioral**: construct a minimal synthetic DataFrame, call the function, assert on output values. No mocking of internal functions. Every test traces to a spec requirement (R-number) or an edge case from the spec's edge case table.

Tests use `pytest` with `polars` DataFrames constructed inline. Coverage target: all public functions and all spec-defined formulas.

---

## Test Modules

### 1. `tests/test_normalization.py` — Normalization pipeline

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_relative_frequency_basic` | R8 | Two samples, known abundances → frequency = abundance / sample_total |
| `test_relative_frequency_sums_to_one` | R8 | Each sample's frequencies sum to 1.0 |
| `test_relative_frequency_excludes_zero_total_samples` | R8, edge | Sample with total abundance 0 is dropped, not division-by-zero |
| `test_clr_basic_no_zeros` | R9 | Known frequencies → CLR values sum to ~0 per sample (compositional closure) |
| `test_clr_zero_replacement` | R9 | Zero frequency → replaced with δ = 0.65 * min(nonzero) / D, then CLR |
| `test_clr_population_mode_is_global` | R9 | Population mode applies CLR across all samples globally |
| `test_clr_intra_subject_is_per_subject` | R9 | Intra-subject mode applies CLR independently per subject — different subjects get different transforms |
| `test_clr_values_sum_to_zero_per_sample` | R9 | CLR is a centered transform: values within each sample sum to ~0 |

### 2. `tests/test_restriction_index.py` — Grouping metrics (R11-R13)

| Test | Spec | What it verifies |
|------|------|-----------------|
| **Shannon entropy** | | |
| `test_entropy_uniform_distribution` | R11 | Equal frequencies → H = log2(N) |
| `test_entropy_single_nonzero` | R11 | One nonzero → H = 0 |
| `test_entropy_empty` | R11 | No positive values → H = 0 |
| **Restriction Index** | | |
| `test_ri_uniform_equals_zero` | R11 | Equal freq across all groups → RI = 0.0 |
| `test_ri_single_group_equals_one` | R11, R34 | Clone in exactly one group → RI = 1.0 |
| `test_ri_all_zero_returns_nan` | R11, edge | Clone absent from all groups → RI = NaN |
| `test_ri_two_groups_hand_calculated` | R11 | 2 groups, known frequencies → hand-calculate H(p)/log2(2), verify RI |
| `test_ri_three_groups_skewed` | R11 | 3 groups with highly skewed distribution → RI close to 1.0 |
| **Dominant Group** | | |
| `test_dominant_is_argmax` | R12 | Highest frequency group is dominant |
| `test_dominant_tie_broken_alphabetically` | R12 | Two groups with equal frequency → alphabetically first wins |
| **Breadth Score** | | |
| `test_breadth_counts_above_threshold` | R13 | Presence threshold 0 → count all nonzero groups |
| `test_breadth_with_nonzero_threshold` | R13 | Presence threshold 0.01 → only groups above 0.01 counted |

### 3. `tests/test_grouping_modes.py` — Population vs intra-subject grouping (R2, R3, R17-R21)

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_pooled_grouping_no_subject` | R2 | Without subject: metrics computed from pooled samples directly |
| `test_population_with_subject_averages_per_subject` | R2 | Population mode with subject: per-subject RI computed then averaged across subjects |
| `test_consensus_dominant_is_mode` | R18 | Mode of per-subject dominants → consensus dominant |
| `test_consensus_dominant_tie_alphabetical` | R18 | Tied mode → alphabetically first |
| `test_count_dominant_in_per_category` | R19 | Count of subjects with each dominant group |
| `test_mean_ri_across_subjects` | R20 | Arithmetic mean of per-subject RI values |
| `test_std_ri_across_subjects` | R21 | Standard deviation (ddof=1) of per-subject RI values |
| `test_min_subject_count_threshold` | R17b | Clone in fewer subjects than minSubjectCount → mean RI = NaN |
| `test_single_subject_std_ri_nan` | R21, edge | Single subject → StdDev = NaN |

### 4. `tests/test_temporal_metrics.py` — Temporal kinetics (R14-R16a)

| Test | Spec | What it verifies |
|------|------|-----------------|
| **Peak Timepoint** | | |
| `test_peak_is_max_frequency` | R14 | Clone with frequencies [0.1, 0.5, 0.3] → peak = timepoint[1] |
| `test_peak_tie_uses_argmax` | R14 | Equal frequencies at two timepoints → first in order (numpy argmax) |
| **Temporal Shift Index** | | |
| `test_tsi_all_at_first_equals_zero` | R15 | All abundance at first timepoint → TSI = 0.0 |
| `test_tsi_all_at_last_equals_one` | R15 | All abundance at last timepoint → TSI = 1.0 |
| `test_tsi_uniform_equals_point_five` | R15 | Equal abundance at all timepoints → TSI = 0.5 (for odd T) |
| `test_tsi_hand_calculated` | R15 | 4 timepoints with known frequencies → TSI = sum(i*f_i) / (sum(f_i) * (T-1)) |
| `test_tsi_single_timepoint_detected` | R15, edge | Clone at one timepoint only → TSI = position/(T-1) |
| **Log2 Peak Delta** | | |
| `test_log2pd_basic_expansion` | R16 | freq_peak > freq_first → Log2PD = log2(peak/first) > 0 |
| `test_log2pd_first_is_peak` | R16 | Peak = first detected → Log2PD = 0.0 |
| `test_log2pd_single_timepoint` | R16, edge | Clone at one timepoint → Log2PD = 0.0 |
| `test_log2pd_always_non_negative` | R16 | Peak >= first by definition → verify Log2PD >= 0 |
| **Log2 Kinetic Delta** | | |
| `test_log2kd_expansion` | R16a | freq_last > freq_first → Log2KD > 0 |
| `test_log2kd_contraction` | R16a | freq_last < freq_first → Log2KD < 0 |
| `test_log2kd_single_timepoint` | R16a, edge | Clone at one timepoint → Log2KD = 0.0 |
| `test_log2kd_no_detection` | edge | Clone not detected at any timepoint → Log2KD = 0.0 |
| **Less than 2 timepoints** | | |
| `test_temporal_metrics_empty_with_one_timepoint` | R15, edge | T=1 → returns empty DataFrame (metrics not computed) |

### 5. `tests/test_temporal_modes.py` — Population vs intra-subject temporal (R2, R3)

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_population_temporal_averages_across_subjects` | R2 | Population mode: averages frequency across subjects per timepoint, then computes metrics |
| `test_intra_subject_temporal_per_subject_then_average` | R3 | Intra-subject mode: computes metrics per subject, then averages across subjects |
| `test_intra_subject_min_subject_count` | R17b | Intra-subject temporal metrics NaN below minSubjectCount |
| `test_intra_subject_peak_timepoint_is_mode` | R14 | Consensus peak timepoint across subjects = mode |

### 6. `tests/test_subject_prevalence.py` — Convergence metrics (R17, R17a)

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_prevalence_counts_distinct_subjects` | R17 | 3 subjects, clone in 2 → prevalence = 2 |
| `test_prevalence_fraction` | R17a | prevalence / total_subjects |
| `test_prevalence_without_subject_counts_samples` | edge | No subject column → counts distinct sampleIds |
| `test_prevalence_histogram_shape` | R30 | Histogram buckets: x = prevalence count, y = number of clones |

### 7. `tests/test_preprocessing.py` — Input handling and replicate averaging (R7a-R7c)

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_min_abundance_filter` | R7c | Clone with peak abundance below threshold excluded from all samples |
| `test_min_abundance_filter_keeps_clone_above_in_one_sample` | R7c | Clone above threshold in ANY sample → kept everywhere |
| `test_null_abundance_dropped` | pre-R8 | NaN/null abundance rows removed |
| `test_replicate_averaging` | R7a | Two samples mapping to same condition → abundance averaged |
| `test_no_replicates_passes_through` | R7a | No duplicate conditions → DataFrame unchanged |
| `test_replicate_synthetic_sampleid` | R7a | Averaged rows get deterministic synthetic sampleId from condition columns |
| `test_missing_grouping_excluded` | R7b, R35 | Null/empty grouping values excluded from grouping metrics |
| `test_missing_timepoint_excluded` | R7b, R35 | Null/empty timepoint values excluded from temporal metrics |

### 8. `tests/test_visualization_data.py` — Output PFrames for visualizations (R28-R30)

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_heatmap_shape_and_columns` | R28 | Output has columns [elementId, groupCategory, normalizedFrequency] |
| `test_heatmap_top_n_by_ri` | R28 | Only top N clones by RI included |
| `test_heatmap_without_grouping_metrics` | R28 | No RI available → all clones included |
| `test_temporal_line_shape` | R29 | Output has columns [elementId, timepointValue, frequency] |
| `test_temporal_line_top_n_by_log2pd` | R29 | Top N clones ranked by absolute Log2 Peak Delta |
| `test_temporal_line_fewer_than_two_timepoints` | R29, edge | T < 2 → returns all clones without ranking |
| `test_prevalence_histogram_bins` | R30 | Correct bin counts for known prevalence distribution |

### 9. `tests/test_end_to_end.py` — Integration / main() pipeline

| Test | Spec | What it verifies |
|------|------|-----------------|
| `test_full_pipeline_population_mode` | R1-R30 | Write CSV to tmp_path, run main logic, verify all output CSVs exist and have correct columns |
| `test_full_pipeline_intra_subject_mode` | R1, R3 | Intra-subject mode produces correct output files |
| `test_grouping_only_no_temporal` | R5a, R32 | Only grouping variable → no temporal output files |
| `test_temporal_only_no_grouping` | R5a, R32 | Only temporal variable → no grouping output files |
| `test_no_subject_skips_convergence` | R5 | No subject variable → no prevalence files |

---

## Spec Gaps Identified During Review

These are areas where the current implementation deviates from or doesn't fully cover the spec:

1. **R6 — Multiple Grouping Variables:** The spec calls for multiple grouping variables (`groupingVariables: GroupingVariableConfig[]`). The implementation supports only a single `groupingColumnRef`. Not testable in current Python code — this would require workflow/model changes.

2. **R3 — Per-subject detail columns (RESOLVED):** `compute_grouping_metrics` and `compute_temporal_metrics` now return tuples `(aggregated, per_subject)`. `main()` writes `result_per_subject.csv` in intra-subject mode with subject. Workflow builds `perSubjectPf` with axes `[elementId, subject]`. Model exposes `perSubjectPf` + `perSubjectPCols` outputs.

3. **R7 — Timepoint deselection (RESOLVED):** The implementation filters by `timepoint_order` at `compartment_analysis.py:370` (`compute_temporal_metrics`) and `:537` (`build_temporal_line_data`). UI-side deselection handled via `PlElementList` in `MainPage.vue`. Added explicit `test_deselected_timepoints_excluded` test verifying behavior.

4. **R18 — Consensus tie-breaking (RESOLVED):** `_consensus_dominant` now breaks ties by highest mean frequency across subjects (with alphabetical as final fallback when frequencies are equal). Matches spec.

5. **_compute_pooled_grouping dominant tie-breaking (RESOLVED):** Both `_compute_pooled_grouping` and `_compute_per_subject_grouping` now use explicit alphabetical tie-breaking per R12. No longer fragile to category list ordering.

---

## Priority

1. **High** — Formula correctness: `test_restriction_index.py`, `test_temporal_metrics.py`, `test_normalization.py`
2. **High** — Mode correctness: `test_grouping_modes.py`, `test_temporal_modes.py`
3. **Medium** — Preprocessing: `test_preprocessing.py`, `test_subject_prevalence.py`
4. **Medium** — Visualization: `test_visualization_data.py`
5. **Lower** — End-to-end: `test_end_to_end.py` (partially covered by block integration test)

---

## Running

```bash
cd software/
uv sync
uv run pytest tests/ -v
uv run pytest --cov=compartment_analysis --cov-report=term-missing tests/
```
