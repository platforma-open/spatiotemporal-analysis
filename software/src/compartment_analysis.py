"""
Spatiotemporal analysis: computes grouping restriction, temporal kinetics,
and cross-subject convergence metrics for clonal/cluster abundance data.

Input: CSV with columns [sampleId, elementId, abundance, subject?, grouping?, timepoint?]
Output: Multiple CSV files with computed metrics.
"""

import argparse
import json
import math

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Spatiotemporal analysis")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("--calculation-mode", choices=["population", "intra-subject"], default="population")
    parser.add_argument("--normalization", choices=["relative-frequency", "clr"], default="relative-frequency")
    parser.add_argument("--has-grouping", action="store_true")
    parser.add_argument("--has-timepoint", action="store_true")
    parser.add_argument("--has-subject", action="store_true")
    parser.add_argument("--timepoint-order", type=str, default="[]")
    parser.add_argument("--presence-threshold", type=float, default=0.0)
    parser.add_argument("--min-abundance-threshold", type=float, default=0.0)
    parser.add_argument("--min-subject-count", type=int, default=2)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output-prefix", type=str, default="output")
    return parser.parse_args()


# Hardcoded column names by convention
COL_SUBJECT = "subject"
COL_GROUPING = "grouping"
COL_TIMEPOINT = "timepoint"

ABUNDANCE_NULL_VALUES = ["", "NaN", "nan", "NA", "na", "null", "None"]


def read_input(path: str, has_grouping: bool, has_timepoint: bool, min_abundance_threshold: float) -> pl.DataFrame:
    """Read input CSV with proper type handling."""
    df = pl.read_csv(path, null_values=ABUNDANCE_NULL_VALUES, infer_schema_length=10000)

    # Ensure abundance is Float64, drop null
    df = df.with_columns(pl.col("abundance").cast(pl.Float64))
    df = df.filter(pl.col("abundance").is_not_null())

    # Apply minimum abundance filter: exclude clones whose peak abundance across
    # all samples is below the threshold (R7c). A clone is kept if it exceeds
    # the threshold in at least one sample.
    if min_abundance_threshold > 0:
        df = df.filter(pl.col("abundance").max().over("elementId") >= min_abundance_threshold)

    # Force categorical columns to String and exclude missing metadata (R7b)
    if has_grouping and COL_GROUPING in df.columns:
        df = df.with_columns(pl.col(COL_GROUPING).cast(pl.String))
    if has_timepoint and COL_TIMEPOINT in df.columns:
        df = df.with_columns(pl.col(COL_TIMEPOINT).cast(pl.String))
    if COL_SUBJECT in df.columns:
        df = df.with_columns(pl.col(COL_SUBJECT).cast(pl.String))
        # Exclude samples with missing/empty subject (R7b) — they still
        # participate in non-subject computations via the pre-filter copy,
        # but must not corrupt prevalence or consensus metrics.
        df = df.filter(pl.col(COL_SUBJECT).is_not_null() & (pl.col(COL_SUBJECT) != ""))

    return df


def average_replicates(df: pl.DataFrame, has_subject: bool, has_grouping: bool, has_timepoint: bool) -> pl.DataFrame:
    """Average abundance when multiple samples map to same condition combination."""
    group_cols = ["elementId"]
    if has_subject:
        group_cols.append(COL_SUBJECT)
    if has_grouping:
        group_cols.append(COL_GROUPING)
    if has_timepoint:
        group_cols.append(COL_TIMEPOINT)

    # Check if there are true replicates by counting unique sampleIds per condition combo
    combo_counts = df.group_by(group_cols).agg(pl.col("sampleId").n_unique().alias("nSamples"))
    if combo_counts["nSamples"].max() <= 1:
        return df  # No replicates, nothing to average

    # Average abundance across replicate samples for each condition combo.
    # Build a deterministic synthetic sampleId from condition columns only
    # (excluding elementId) so all clones in the same condition share one
    # sampleId — required for correct per-sample normalization (R7a/R8).
    averaged = df.group_by(group_cols).agg(pl.col("abundance").mean().alias("abundance"))
    condition_cols = [c for c in group_cols if c != "elementId"]
    if condition_cols:
        averaged = averaged.with_columns(
            pl.concat_str([pl.col(c).cast(pl.String) for c in condition_cols], separator="|").alias("sampleId")
        )
    else:
        averaged = averaged.with_columns(pl.lit("__all__").alias("sampleId"))
    return averaged


def compute_relative_frequency(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-sample relative frequency from abundance."""
    sample_totals = df.group_by("sampleId").agg(pl.col("abundance").sum().alias("sampleTotal"))
    df = df.join(sample_totals, on="sampleId")
    df = df.filter(pl.col("sampleTotal") > 0)
    df = df.with_columns((pl.col("abundance") / pl.col("sampleTotal")).alias("frequency")).drop("sampleTotal")
    return df


def compute_clr(df: pl.DataFrame, mode: str, has_subject: bool) -> pl.DataFrame:
    """Compute centered log-ratio transform.
    Global in population mode, per-subject in intra-subject mode.

    Vectorized: uses Polars window expressions instead of map_groups callbacks."""
    sample_totals = df.group_by("sampleId").agg(pl.col("abundance").sum().alias("sampleTotal"))
    df = df.join(sample_totals, on="sampleId")
    df = df.filter(pl.col("sampleTotal") > 0)
    df = df.with_columns((pl.col("abundance") / pl.col("sampleTotal")).alias("frequency")).drop("sampleTotal")

    # Compute min nonzero frequency (scope depends on mode)
    if mode == "intra-subject" and has_subject:
        # Per-subject min nonzero
        min_nz_expr = (
            pl.when(pl.col("frequency") > 0)
            .then(pl.col("frequency"))
            .otherwise(None)
            .min()
            .over(COL_SUBJECT)
            .fill_null(1e-10)
        )
    else:
        # Global min nonzero
        min_val = df.filter(pl.col("frequency") > 0)["frequency"].min()
        if min_val is None or min_val <= 0:
            min_val = 1e-10
        min_nz_expr = pl.lit(min_val)

    # D = number of components per sample; delta = 0.65 * min_nonzero / D
    d_expr = pl.len().over("sampleId").cast(pl.Float64)
    delta_expr = 0.65 * min_nz_expr / d_expr

    # Multiplicative replacement for zeros
    df = df.with_columns(
        pl.when(pl.col("frequency") == 0).then(delta_expr).otherwise(pl.col("frequency")).alias("_replaced")
    )

    # Renormalize per sample
    df = df.with_columns(
        (pl.col("_replaced") / pl.col("_replaced").sum().over("sampleId")).alias("_normed")
    )

    # CLR: log(normed) - mean(log(normed)) per sample
    df = df.with_columns(
        (pl.col("_normed").log() - pl.col("_normed").log().mean().over("sampleId")).alias("frequency")
    )

    return df.drop(["_replaced", "_normed"])


def shannon_entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return -float(np.sum(p * np.log2(p)))


def restriction_index(freq_by_group: np.ndarray) -> float:
    """RI = 1 - H(p) / log2(N)"""
    nonzero = freq_by_group[freq_by_group > 0]
    n = len(nonzero)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0
    h = shannon_entropy(nonzero)
    return 1.0 - h / math.log2(n)


# ---------------------------------------------------------------------------
# Vectorized grouping metrics
# ---------------------------------------------------------------------------


def _compute_ri_dominant_breadth(
    freq_matrix: np.ndarray,
    categories: list[str],
    presence_threshold: float,
) -> tuple[np.ndarray, list[str | None], np.ndarray]:
    """Vectorized RI, dominant, breadth from a [N x C] frequency matrix.

    categories must be sorted alphabetically so that np.argmax gives
    alphabetical tie-breaking (returns first occurrence of the max).
    """
    n_rows = len(freq_matrix)

    # Normalize rows to probability distribution
    row_sums = freq_matrix.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums == 0, 1.0, row_sums)
    p = freq_matrix / safe_sums

    # Count nonzero groups per row
    n_nonzero = (freq_matrix > 0).sum(axis=1)

    # Shannon entropy per row: H = -sum(p_i * log2(p_i)) for p_i > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_p = np.where(p > 0, np.log2(p), 0.0)
    entropy = -np.sum(p * log2_p, axis=1)

    # RI = 1 - H / log2(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_n = np.where(n_nonzero > 1, np.log2(n_nonzero.astype(np.float64)), 1.0)
    ri = np.where(n_nonzero == 0, np.nan, np.where(n_nonzero == 1, 1.0, 1.0 - entropy / log2_n))

    # Dominant: argmax per row (alphabetical tie-breaking via sorted categories)
    dominant_idx = np.argmax(freq_matrix, axis=1)
    max_vals = freq_matrix[np.arange(n_rows), dominant_idx]
    dominant: list[str | None] = [
        categories[idx] if max_vals[row] > 0 else None for row, idx in enumerate(dominant_idx)
    ]

    # Breadth: count of groups above presence threshold
    breadth = (freq_matrix > presence_threshold).sum(axis=1).astype(np.int64)

    return ri, dominant, breadth


def compute_grouping_metrics(
    df: pl.DataFrame,
    has_subject: bool,
    mode: str,
    presence_threshold: float,
    min_subject_count: int,
) -> pl.DataFrame:
    """Compute RI, dominant group, breadth for the grouping variable."""
    # Exclude samples with missing grouping
    df = df.filter(pl.col(COL_GROUPING).is_not_null() & (pl.col(COL_GROUPING) != ""))
    categories = sorted(df[COL_GROUPING].unique().to_list())

    if len(categories) == 0:
        return pl.DataFrame()

    if not has_subject:
        return _compute_pooled_grouping(df, categories, presence_threshold).sort("elementId")

    per_subject_grouping = df.group_by(["elementId", COL_SUBJECT, COL_GROUPING]).agg(
        pl.col("frequency").mean().alias("meanFreq")
    )

    per_subject_metrics = _compute_per_subject_grouping(per_subject_grouping, categories, presence_threshold)

    if per_subject_metrics.is_empty():
        return pl.DataFrame()

    # --- Vectorized consensus aggregation ---

    # Only subjects where the clone is actually detected (breadth > 0) count
    # toward _nSubjects for the minSubjectCount threshold. Subjects with 0
    # frequency in all groups are absent, not present (R17b uses "present in").
    present_subjects = per_subject_metrics.filter(pl.col("breadth") > 0)

    # 1. Subject count and breadth from present subjects
    all_subject_agg = present_subjects.group_by("elementId").agg(
        [
            pl.len().alias("_nSubjects"),
            pl.col("breadth").mean().alias("_meanBreadth"),
        ]
    )

    # 2. Mean/std RI from multi-group subjects only (R33: subjects sampled from
    # only one group have trivial RI=1.0 — an artifact of sampling, not
    # biology — and must be excluded from consensus RI statistics)
    ri_eligible = present_subjects.filter(pl.col("breadth") > 1)
    ri_agg_raw = ri_eligible.group_by("elementId").agg(
        [
            pl.col("ri").drop_nulls().mean().alias("_meanRi"),
            pl.col("ri").drop_nulls().std(ddof=1).alias("_stdRi"),
            pl.col("ri").drop_nulls().count().alias("_riCount"),
        ]
    )

    ri_agg = all_subject_agg.join(ri_agg_raw, on="elementId", how="left")

    ri_agg = ri_agg.with_columns(
        [
            pl.when(
                (pl.col("_nSubjects") >= min_subject_count)
                & pl.col("_meanRi").is_not_null()
            )
            .then(pl.col("_meanRi"))
            .otherwise(float("nan"))
            .alias("meanRi"),
            pl.when(
                (pl.col("_nSubjects") >= min_subject_count)
                & (pl.col("_riCount").fill_null(0) > 1)
            )
            .then(pl.col("_stdRi"))
            .otherwise(float("nan"))
            .alias("stdRi"),
            pl.col("_meanBreadth").fill_null(0).round(0).cast(pl.Int64).alias("breadth"),
        ]
    ).select(["elementId", "meanRi", "stdRi", "breadth"])

    # 2. Consensus dominant: mode of per-subject dominant, ties broken by mean freq then alphabetical
    dom_counts = (
        per_subject_metrics.filter(pl.col("dominant").is_not_null())
        .group_by(["elementId", "dominant"])
        .agg(pl.len().alias("_domCount"))
    )

    mean_freqs = per_subject_grouping.group_by(["elementId", COL_GROUPING]).agg(
        pl.col("meanFreq").mean().alias("_meanFreq")
    )

    consensus = (
        dom_counts.join(
            mean_freqs, left_on=["elementId", "dominant"], right_on=["elementId", COL_GROUPING], how="left"
        )
        .with_columns(pl.col("_meanFreq").fill_null(0.0))
        .sort(["elementId", "_domCount", "_meanFreq", "dominant"], descending=[False, True, True, False])
        .group_by("elementId", maintain_order=True)
        .first()
        .select(["elementId", pl.col("dominant").alias("consensusDominant")])
    )

    # 3. Count dominant per category (pivot)
    count_dom_raw = (
        per_subject_metrics.filter(pl.col("dominant").is_not_null())
        .group_by(["elementId", "dominant"])
        .agg(pl.len().alias("_count"))
    )

    if len(count_dom_raw) > 0:
        count_dom = count_dom_raw.pivot(on="dominant", index="elementId", values="_count").fill_null(0)
        for cat in categories:
            if cat in count_dom.columns:
                count_dom = count_dom.rename({cat: f"countDominantIn_{cat}"})
            else:
                count_dom = count_dom.with_columns(pl.lit(0).alias(f"countDominantIn_{cat}"))
        count_cols = ["elementId"] + [f"countDominantIn_{cat}" for cat in categories]
        count_dom = count_dom.select(count_cols)
    else:
        data: dict = {"elementId": ri_agg["elementId"].to_list()}
        for cat in categories:
            data[f"countDominantIn_{cat}"] = [0] * len(ri_agg)
        count_dom = pl.DataFrame(data)

    # 4. Join everything together
    result = ri_agg.join(consensus, on="elementId", how="left")
    result = result.join(count_dom, on="elementId", how="left")

    # Add summary columns (ri = meanRi, dominant = consensusDominant)
    result = result.with_columns(
        [
            pl.col("meanRi").alias("ri"),
            pl.col("consensusDominant").alias("dominant"),
        ]
    )

    # Ensure column order matches expected output
    output_cols = ["elementId", "ri", "dominant", "breadth", "consensusDominant", "meanRi", "stdRi"]
    output_cols += [f"countDominantIn_{cat}" for cat in categories]
    result = result.select(output_cols)

    return result.sort("elementId")


def _compute_pooled_grouping(
    df: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Vectorized grouping metrics without subject dimension."""
    per_grouping = df.group_by(["elementId", COL_GROUPING]).agg(pl.col("frequency").mean().alias("meanFreq"))

    wide = per_grouping.pivot(on=COL_GROUPING, index="elementId", values="meanFreq").fill_null(0.0)
    for cat in categories:
        if cat not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(cat))

    if wide.is_empty():
        return pl.DataFrame()

    element_ids = wide["elementId"].to_list()
    freq_matrix = wide.select(categories).to_numpy().astype(np.float64)

    ri, dominant, breadth = _compute_ri_dominant_breadth(freq_matrix, categories, presence_threshold)

    return pl.DataFrame(
        {
            "elementId": element_ids,
            "ri": ri.tolist(),
            "dominant": dominant,
            "breadth": breadth.tolist(),
        }
    )


def _compute_per_subject_grouping(
    per_subject_grouping: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Vectorized per-subject RI, dominant, breadth."""
    wide = per_subject_grouping.pivot(
        on=COL_GROUPING, index=["elementId", COL_SUBJECT], values="meanFreq"
    ).fill_null(0.0)

    for cat in categories:
        if cat not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(cat))

    if wide.is_empty():
        return pl.DataFrame()

    element_ids = wide["elementId"].to_list()
    subjects = wide[COL_SUBJECT].to_list()
    freq_matrix = wide.select(categories).to_numpy().astype(np.float64)

    ri, dominant, breadth = _compute_ri_dominant_breadth(freq_matrix, categories, presence_threshold)

    return pl.DataFrame(
        {
            "elementId": element_ids,
            COL_SUBJECT: subjects,
            "ri": ri.tolist(),
            "dominant": dominant,
            "breadth": breadth.tolist(),
        }
    )


# ---------------------------------------------------------------------------
# Vectorized temporal metrics
# ---------------------------------------------------------------------------


def _compute_temporal_matrix(
    wide: pl.DataFrame,
    timepoint_order: list[str],
    t_count: int,
) -> pl.DataFrame:
    """Vectorized temporal metrics from a pivoted wide DataFrame.

    wide must have columns: [id_cols...] + timepoint columns.
    Returns a DataFrame with the same id columns + temporal metric columns.
    """
    for tp in timepoint_order:
        if tp not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(tp))

    id_cols = [c for c in wide.columns if c not in timepoint_order]
    freq_matrix = wide.select(timepoint_order).to_numpy().astype(np.float64)
    n_rows = len(freq_matrix)

    if n_rows == 0:
        schema: dict = {c: wide.schema[c] for c in id_cols}
        schema.update(
            {
                "peakTimepoint": pl.String,
                "temporalShiftIndex": pl.Float64,
                "log2KineticDelta": pl.Float64,
                "log2PeakDelta": pl.Float64,
            }
        )
        return pl.DataFrame(schema=schema)

    # Peak timepoint: argmax per row
    peak_indices = np.argmax(freq_matrix, axis=1)
    peak_timepoints = [timepoint_order[i] for i in peak_indices]

    # TSI = sum(i * freq_i) / (sum(freq_i) * (T - 1))
    positions = np.arange(t_count, dtype=np.float64)
    total_freq = freq_matrix.sum(axis=1)
    numerator = (freq_matrix * positions[np.newaxis, :]).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsi = np.where((total_freq > 0) & (t_count > 1), numerator / (total_freq * (t_count - 1)), np.nan)

    # First and last nonzero detection indices
    nonzero_mask = freq_matrix > 0
    n_nonzero = nonzero_mask.sum(axis=1)
    first_idx = np.argmax(nonzero_mask, axis=1)
    last_idx = t_count - 1 - np.argmax(nonzero_mask[:, ::-1], axis=1)

    row_idx = np.arange(n_rows)
    first_freq = freq_matrix[row_idx, first_idx]
    last_freq = freq_matrix[row_idx, last_idx]
    peak_freq = freq_matrix[row_idx, peak_indices]

    # Log2 Kinetic Delta: log2(last_detected / first_detected)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2kd = np.where(n_nonzero >= 2, np.log2(last_freq / first_freq), 0.0)

    # Log2 Peak Delta: log2(peak / first_detected), clamped to >= 0 per spec
    with np.errstate(divide="ignore", invalid="ignore"):
        log2pd = np.where(
            (n_nonzero >= 1) & (first_freq > 0) & (peak_freq > 0), np.log2(peak_freq / first_freq), 0.0
        )
        log2pd = np.maximum(log2pd, 0.0)

    result_data: dict = {c: wide[c].to_list() for c in id_cols}
    result_data.update(
        {
            "peakTimepoint": peak_timepoints,
            "temporalShiftIndex": tsi.tolist(),
            "log2KineticDelta": log2kd.tolist(),
            "log2PeakDelta": log2pd.tolist(),
        }
    )

    return pl.DataFrame(result_data)


def compute_temporal_metrics(
    df: pl.DataFrame,
    timepoint_order: list[str],
    has_subject: bool,
    mode: str,
    min_subject_count: int,
) -> pl.DataFrame:
    """Compute peak timepoint, TSI, log2 kinetic delta, log2 peak delta.

    Vectorized: pivots to wide format and computes metrics via numpy matrix ops.
    """
    if len(timepoint_order) < 2:
        return pl.DataFrame()

    # Exclude samples with missing timepoint
    df = df.filter(pl.col(COL_TIMEPOINT).is_not_null() & (pl.col(COL_TIMEPOINT) != ""))
    df = df.filter(pl.col(COL_TIMEPOINT).is_in(timepoint_order))

    t_count = len(timepoint_order)

    if not has_subject or mode == "population":
        if has_subject:
            # Two-stage aggregation: per-subject mean first, then cross-subject mean.
            per_tp = (
                df.group_by(["elementId", COL_SUBJECT, COL_TIMEPOINT])
                .agg(pl.col("frequency").mean().alias("meanFreq"))
                .group_by(["elementId", COL_TIMEPOINT])
                .agg(pl.col("meanFreq").mean().alias("meanFreq"))
            )
        else:
            per_tp = df.group_by(["elementId", COL_TIMEPOINT]).agg(pl.col("frequency").mean().alias("meanFreq"))

        wide = per_tp.pivot(on=COL_TIMEPOINT, index="elementId", values="meanFreq").fill_null(0.0)
        result = _compute_temporal_matrix(wide, timepoint_order, t_count)

        if result.is_empty():
            return pl.DataFrame()
        return result.sort("elementId")

    else:
        # Intra-subject: per-subject metrics then average across subjects
        per_tp = df.group_by(["elementId", COL_SUBJECT, COL_TIMEPOINT]).agg(
            pl.col("frequency").mean().alias("meanFreq")
        )

        wide = per_tp.pivot(on=COL_TIMEPOINT, index=["elementId", COL_SUBJECT], values="meanFreq").fill_null(0.0)
        ps_df = _compute_temporal_matrix(wide, timepoint_order, t_count)

        if ps_df.is_empty():
            return pl.DataFrame()

        # Consensus peak timepoint: mode with alphabetical tie-breaking
        peak_consensus = (
            ps_df.group_by(["elementId", "peakTimepoint"])
            .agg(pl.len().alias("_count"))
            .sort(["elementId", "_count", "peakTimepoint"], descending=[False, True, False])
            .group_by("elementId", maintain_order=True)
            .first()
            .select(["elementId", "peakTimepoint"])
        )

        # Mean metrics with minSubjectCount threshold
        temporal_agg = ps_df.group_by("elementId").agg(
            [
                pl.len().alias("_nSubjects"),
                pl.col("temporalShiftIndex").mean().alias("_tsi"),
                pl.col("log2KineticDelta").mean().alias("_log2kd"),
                pl.col("log2PeakDelta").mean().alias("_log2pd"),
            ]
        )

        temporal_agg = temporal_agg.with_columns(
            [
                pl.when(pl.col("_nSubjects") >= min_subject_count)
                .then(pl.col("_tsi"))
                .otherwise(float("nan"))
                .alias("temporalShiftIndex"),
                pl.when(pl.col("_nSubjects") >= min_subject_count)
                .then(pl.col("_log2kd"))
                .otherwise(float("nan"))
                .alias("log2KineticDelta"),
                pl.when(pl.col("_nSubjects") >= min_subject_count)
                .then(pl.col("_log2pd"))
                .otherwise(float("nan"))
                .alias("log2PeakDelta"),
            ]
        ).select(["elementId", "temporalShiftIndex", "log2KineticDelta", "log2PeakDelta"])

        result = temporal_agg.join(peak_consensus, on="elementId", how="left")
        return (
            result.select(["elementId", "peakTimepoint", "temporalShiftIndex", "log2KineticDelta", "log2PeakDelta"])
            .sort("elementId")
        )


def compute_subject_prevalence(df: pl.DataFrame, has_subject: bool) -> pl.DataFrame:
    """Count distinct subjects per element. Returns subjectPrevalence and fraction."""
    if not has_subject:
        # Without subject: count distinct samples
        prevalence = (
            df.filter(pl.col("abundance") > 0)
            .group_by("elementId")
            .agg(pl.col("sampleId").n_unique().alias("subjectPrevalence"))
        )
        total = df["sampleId"].n_unique()
    else:
        prevalence = (
            df.filter(pl.col("abundance") > 0)
            .group_by("elementId")
            .agg(pl.col(COL_SUBJECT).n_unique().alias("subjectPrevalence"))
        )
        total = df[COL_SUBJECT].n_unique()

    prevalence = prevalence.with_columns(
        (pl.col("subjectPrevalence").cast(pl.Float64) / float(total)).alias("subjectPrevalenceFraction")
    )
    return prevalence.sort("elementId")


def build_heatmap_data(
    df: pl.DataFrame,
    grouping_metrics: pl.DataFrame | None,
    top_n: int = 50,
) -> pl.DataFrame:
    """Build heatmap data for top N clones by RI (most restricted)."""
    df = df.filter(pl.col(COL_GROUPING).is_not_null() & (pl.col(COL_GROUPING) != ""))
    heatmap = df.group_by(["elementId", COL_GROUPING]).agg(pl.col("frequency").mean().alias("normalizedFrequency"))

    # Filter to top N clones by Restriction Index
    if grouping_metrics is not None and "ri" in grouping_metrics.columns and len(grouping_metrics) > 0:
        top_elements = (
            grouping_metrics.filter(pl.col("ri").is_not_null())
            .sort(["ri", "elementId"], descending=[True, False])
            .head(top_n)["elementId"]
            .to_list()
        )
        heatmap = heatmap.filter(pl.col("elementId").is_in(top_elements))

    return heatmap.rename({COL_GROUPING: "groupCategory"}).sort("elementId", "groupCategory")


def build_temporal_line_data(
    df: pl.DataFrame,
    timepoint_order: list[str],
    top_n: int,
) -> pl.DataFrame:
    """Build temporal line plot data for top N clones ranked by Log2 Peak Delta.

    Vectorized: pivots to wide format for scoring instead of per-element loop.
    """
    df = df.filter(
        pl.col(COL_TIMEPOINT).is_not_null()
        & (pl.col(COL_TIMEPOINT) != "")
        & pl.col(COL_TIMEPOINT).is_in(timepoint_order)
    )

    t_count = len(timepoint_order)
    if t_count < 2:
        line_data = df.group_by(["elementId", COL_TIMEPOINT]).agg(pl.col("frequency").mean().alias("frequency"))
        return line_data.rename({COL_TIMEPOINT: "timepointValue"}).sort("elementId", "timepointValue")

    per_tp = df.group_by(["elementId", COL_TIMEPOINT]).agg(pl.col("frequency").mean().alias("meanFreq"))

    # Pivot to wide and compute Log2PD scores in one vectorized pass
    wide = per_tp.pivot(on=COL_TIMEPOINT, index="elementId", values="meanFreq").fill_null(0.0)
    for tp in timepoint_order:
        if tp not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(tp))

    freq_matrix = wide.select(timepoint_order).to_numpy().astype(np.float64)
    n_rows = len(freq_matrix)

    if n_rows == 0:
        return pl.DataFrame()

    peak_indices = np.argmax(freq_matrix, axis=1)
    nonzero_mask = freq_matrix > 0
    has_any = nonzero_mask.any(axis=1)
    first_indices = np.argmax(nonzero_mask, axis=1)

    row_idx = np.arange(n_rows)
    first_freq = freq_matrix[row_idx, first_indices]
    peak_freq = freq_matrix[row_idx, peak_indices]

    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.where(has_any & (first_freq > 0) & (peak_freq > 0), np.log2(peak_freq / first_freq), 0.0)

    scores_df = (
        pl.DataFrame({"elementId": wide["elementId"].to_list(), "score": scores.tolist()})
        .sort(["score", "elementId"], descending=[True, False])
        .head(top_n)
    )
    top_elements = scores_df["elementId"].to_list()

    # Filter to top N elements
    line_data = (
        df.filter(pl.col("elementId").is_in(top_elements))
        .group_by(["elementId", COL_TIMEPOINT])
        .agg(pl.col("frequency").mean().alias("frequency"))
    )
    return line_data.rename({COL_TIMEPOINT: "timepointValue"}).sort("elementId", "timepointValue")


def build_prevalence_histogram(prevalence_df: pl.DataFrame) -> pl.DataFrame:
    return (
        prevalence_df.group_by("subjectPrevalence")
        .agg(pl.len().alias("cloneCount"))
        .rename({"subjectPrevalence": "prevalenceCount"})
        .sort("prevalenceCount")
    )


def main():
    args = parse_args()

    df = read_input(args.input_file, args.has_grouping, args.has_timepoint, args.min_abundance_threshold)
    timepoint_order = json.loads(args.timepoint_order)
    mode = args.calculation_mode
    prefix = args.output_prefix
    has_subject = args.has_subject and COL_SUBJECT in df.columns

    # Automatic sample averaging for replicates
    df = average_replicates(df, has_subject, args.has_grouping, args.has_timepoint)

    # Normalize
    if args.normalization == "clr":
        df = compute_clr(df, mode, has_subject)
    else:
        df = compute_relative_frequency(df)

    # Subject prevalence (only when subject variable is set)
    prevalence = None
    if has_subject:
        prevalence = compute_subject_prevalence(df, has_subject)
        prevalence.write_csv(f"{prefix}_prevalence.csv")

        histogram = build_prevalence_histogram(prevalence)
        histogram.write_csv(f"{prefix}_prevalence_histogram.csv")

    # Grouping metrics
    grouping = None
    if args.has_grouping:
        grouping = compute_grouping_metrics(df, has_subject, mode, args.presence_threshold, args.min_subject_count)
        if len(grouping) > 0:
            grouping.write_csv(f"{prefix}_grouping.csv")

        heatmap = build_heatmap_data(
            df, grouping if grouping is not None and len(grouping) > 0 else None, top_n=args.top_n
        )
        if len(heatmap) > 0:
            heatmap.write_csv(f"{prefix}_heatmap.csv")

    # Temporal metrics
    temporal = None
    if args.has_timepoint and len(timepoint_order) >= 2:
        temporal = compute_temporal_metrics(df, timepoint_order, has_subject, mode, args.min_subject_count)
        if len(temporal) > 0:
            temporal.write_csv(f"{prefix}_temporal.csv")

        line_data = build_temporal_line_data(df, timepoint_order, args.top_n)
        if len(line_data) > 0:
            line_data.write_csv(f"{prefix}_temporal_line.csv")

    # Combined main table — join in-memory DataFrames (no disk re-read)
    if prevalence is not None:
        main_table = prevalence
    else:
        main_table = df.select("elementId").unique().sort("elementId")

    if grouping is not None and len(grouping) > 0:
        main_table = main_table.join(grouping, on="elementId", how="left")

    if temporal is not None and len(temporal) > 0:
        main_table = main_table.join(temporal, on="elementId", how="left")

    main_table = main_table.sort("elementId")
    main_table.write_csv(f"{prefix}_main.csv")

    print(f"Analysis complete. Mode: {mode}, Normalization: {args.normalization}")
    print(f"Results written with prefix: {prefix}")


if __name__ == "__main__":
    main()
