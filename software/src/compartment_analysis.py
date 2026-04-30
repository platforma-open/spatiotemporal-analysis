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


def _js_str(v):
    """Match JS Number.prototype.toString() so '5.0' -> '5', '6.5' -> '6.5',
    non-numeric strings pass through unchanged. Needed because the block UI
    serializes numeric metadata via JS String() ('5'), while polars sink_csv
    serializes Float64 5.0 as '5.0'; without normalization the two never match."""
    if v is None or v == "":
        return v
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (ValueError, TypeError):
        return v


def _normalize_categorical(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Apply _js_str via a unique-value mapping. Metadata cardinality is small
    (subject/grouping/timepoint typically <100 unique values) while the frame
    has one row per clonotype × sample, so doing the Python work per-row with
    map_elements would be orders of magnitude slower."""
    if col not in df.columns:
        return df
    uniques = df[col].unique().drop_nulls().to_list()
    mapping = {u: _js_str(u) for u in uniques}
    return df.with_columns(pl.col(col).replace(mapping))


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
    # Force categorical columns to String at read time so polars doesn't infer
    # Float64 from numeric-looking values (which would later cast to '5.0' and
    # mismatch the UI's '5').
    schema_overrides = {}
    if has_grouping:
        schema_overrides[COL_GROUPING] = pl.String
    if has_timepoint:
        schema_overrides[COL_TIMEPOINT] = pl.String
    schema_overrides[COL_SUBJECT] = pl.String
    df = pl.read_csv(
        path,
        null_values=ABUNDANCE_NULL_VALUES,
        infer_schema_length=10000,
        schema_overrides=schema_overrides,
    )

    # Ensure abundance is Float64, drop null
    df = df.with_columns(pl.col("abundance").cast(pl.Float64))
    df = df.filter(pl.col("abundance").is_not_null())

    # Apply minimum abundance filter: exclude clones whose peak abundance across
    # all samples is below the threshold (R7c). A clone is kept if it exceeds
    # the threshold in at least one sample.
    if min_abundance_threshold > 0:
        df = df.filter(pl.col("abundance").max().over("elementId") >= min_abundance_threshold)

    # Normalize numeric-looking categorical values so they match the JS String()
    # representation the UI sends in --timepoint-order (e.g. CSV '5.0' -> '5').
    if has_grouping:
        df = _normalize_categorical(df, COL_GROUPING)
    if has_timepoint:
        df = _normalize_categorical(df, COL_TIMEPOINT)
    df = _normalize_categorical(df, COL_SUBJECT)

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
    """Compute centered log-ratio transform (vectorized).

    Global scope in population mode, per-subject scope in intra-subject mode
    (affects where ``min(nonzero)`` is computed). ``D`` is always per-sample
    (number of components within the sample).
    """
    sample_totals = df.group_by("sampleId").agg(pl.col("abundance").sum().alias("sampleTotal"))
    df = df.join(sample_totals, on="sampleId")
    df = df.filter(pl.col("sampleTotal") > 0)
    df = df.with_columns((pl.col("abundance") / pl.col("sampleTotal")).alias("frequency")).drop("sampleTotal")

    if df.is_empty():
        return df

    # Scope-bound min of nonzero frequencies — this sets delta in the
    # multiplicative zero replacement (Martín-Fernández et al.).
    intra = mode == "intra-subject" and has_subject
    nonzero_freq = (
        pl.when(pl.col("frequency") > 0).then(pl.col("frequency")).otherwise(None)
    )
    if intra:
        min_nz_expr = nonzero_freq.min().over(COL_SUBJECT)
    else:
        # Global min across the whole DataFrame.
        min_nz_scalar = df.filter(pl.col("frequency") > 0)["frequency"].min()
        if min_nz_scalar is None or min_nz_scalar <= 0:
            min_nz_scalar = 1e-10
        min_nz_expr = pl.lit(float(min_nz_scalar))

    df = df.with_columns(
        pl.when((min_nz_expr.is_null()) | (min_nz_expr <= 0))
        .then(pl.lit(1e-10))
        .otherwise(min_nz_expr)
        .alias("_min_nz"),
        pl.len().over("sampleId").cast(pl.Float64).alias("_D"),
    )

    df = df.with_columns(
        (0.65 * pl.col("_min_nz") / pl.col("_D")).alias("_delta")
    ).with_columns(
        pl.when(pl.col("frequency") == 0)
        .then(pl.col("_delta"))
        .otherwise(pl.col("frequency"))
        .alias("_freq_rep")
    )

    # Renormalize per sample so zero-replaced frequencies still sum to 1.
    df = df.with_columns(
        (pl.col("_freq_rep") / pl.col("_freq_rep").sum().over("sampleId")).alias("_freq_norm")
    )

    # CLR = log(f) - mean(log(f)) within each sample.
    df = df.with_columns(
        (pl.col("_freq_norm").log() - pl.col("_freq_norm").log().mean().over("sampleId")).alias("frequency")
    )

    return df.drop(["_min_nz", "_D", "_delta", "_freq_rep", "_freq_norm"])


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


def compute_grouping_metrics(
    df: pl.DataFrame,
    has_subject: bool,
    mode: str,
    presence_threshold: float,
    min_subject_count: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Compute RI, dominant group, breadth for the grouping variable.

    R3: Returns (aggregated_metrics, per_subject_metrics_or_none).
    per_subject_metrics holds [elementId, subject, ri, dominant, breadth] for
    Table-block inspection. It is None when has_subject=False or no data.
    """
    # Exclude samples with missing grouping
    df = df.filter(pl.col(COL_GROUPING).is_not_null() & (pl.col(COL_GROUPING) != ""))
    categories = sorted(df[COL_GROUPING].unique().to_list())

    if len(categories) == 0:
        return pl.DataFrame(), None

    if not has_subject:
        # No subject: treat all samples as pooled
        return _compute_pooled_grouping(df, categories, presence_threshold).sort("elementId"), None

    per_subject_grouping = df.group_by(["elementId", COL_SUBJECT, COL_GROUPING]).agg(
        pl.col("frequency").mean().alias("meanFreq")
    )

    per_subject_metrics = _compute_per_subject_grouping(per_subject_grouping, categories, presence_threshold)
    per_subject_out = (
        per_subject_metrics.sort(["elementId", COL_SUBJECT]) if len(per_subject_metrics) > 0 else None
    )

    if per_subject_metrics.is_empty():
        return pl.DataFrame(), per_subject_out

    # Vectorized aggregation across subjects: mean/std RI, breadth, nSubjects,
    # and per-category dominant counts (R19–R21).
    nan = float("nan")
    count_dominant_exprs = [
        pl.col("dominant").eq(cat).sum().cast(pl.Int64).alias(f"countDominantIn_{cat}")
        for cat in categories
    ]
    agg = (
        per_subject_metrics.group_by("elementId")
        .agg(
            pl.col("ri").mean().alias("meanRi"),
            pl.col("ri").std(ddof=1).alias("stdRi"),
            pl.col("breadth").mean().alias("_breadthMean"),
            pl.len().alias("nSubjects"),
            *count_dominant_exprs,
        )
        .with_columns(
            # R17b: insufficient subjects → NaN for mean/std RI.
            pl.when(pl.col("nSubjects") >= min_subject_count)
            .then(pl.col("meanRi"))
            .otherwise(pl.lit(nan))
            .fill_null(pl.lit(nan))
            .alias("meanRi"),
            pl.when(pl.col("nSubjects") >= min_subject_count)
            .then(pl.col("stdRi"))
            .otherwise(pl.lit(nan))
            .fill_null(pl.lit(nan))
            .alias("stdRi"),
            pl.col("_breadthMean").round(0).cast(pl.Int64).alias("breadth"),
        )
        .drop("_breadthMean")
    )

    # Consensus dominant (R18): mode across subjects; ties broken by highest
    # per-element-per-group mean frequency, then alphabetically.
    per_element_group_freq = per_subject_grouping.group_by(["elementId", COL_GROUPING]).agg(
        pl.col("meanFreq").mean().alias("elGroupFreq")
    )
    consensus = (
        per_subject_metrics.filter(pl.col("dominant").is_not_null())
        .group_by(["elementId", "dominant"])
        .agg(pl.len().alias("_dcount"))
        .with_columns(pl.col("_dcount").max().over("elementId").alias("_max"))
        .filter(pl.col("_dcount") == pl.col("_max"))
        .join(
            per_element_group_freq.rename({COL_GROUPING: "dominant"}),
            on=["elementId", "dominant"],
            how="left",
        )
        .with_columns(pl.col("elGroupFreq").fill_null(0.0))
        .sort(["elementId", "elGroupFreq", "dominant"], descending=[False, True, False])
        .unique(subset="elementId", keep="first")
        .select("elementId", pl.col("dominant").alias("consensusDominant"))
    )

    agg = agg.join(consensus, on="elementId", how="left").with_columns(
        pl.col("meanRi").alias("ri"),
        pl.col("consensusDominant").alias("dominant"),
    )

    final_cols = [
        "elementId",
        "ri",
        "dominant",
        "breadth",
        "consensusDominant",
        "meanRi",
        "stdRi",
        *[f"countDominantIn_{cat}" for cat in categories],
    ]
    return agg.select(final_cols).sort("elementId"), per_subject_out


def _grouping_metrics_from_wide(
    wide: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
    index_cols: list[str],
) -> pl.DataFrame:
    """Compute ri/dominant/breadth as Polars expressions across a wide DataFrame.

    wide has one row per (index_cols) and one column per category (meanFreq,
    zero-filled). Implements R11 (RI), R12 (dominant with alphabetical tie-break),
    and R13 (breadth).
    """
    # Ensure every category exists as a column (pivot omits categories absent
    # from all rows). Add zero columns for missing ones.
    missing = [c for c in categories if c not in wide.columns]
    if missing:
        wide = wide.with_columns(*[pl.lit(0.0).alias(c) for c in missing])

    total = pl.sum_horizontal(*[pl.col(c) for c in categories])

    # Count nonzero categories per row (N in RI formula).
    n_nonzero = pl.sum_horizontal(
        *[pl.when(pl.col(c) > 0).then(1).otherwise(0) for c in categories]
    )

    # Shannon entropy on normalized proportions: H = -sum(p_i * log2(p_i)) for p_i > 0.
    h_terms = [
        pl.when(pl.col(c) > 0)
        .then(-(pl.col(c) / total) * (pl.col(c) / total).log(base=2))
        .otherwise(0.0)
        for c in categories
    ]
    h_expr = pl.sum_horizontal(*h_terms)

    # RI = 1 - H/log2(N); edge cases: N=0→NaN, N=1→1.0.
    ri_expr = (
        pl.when(n_nonzero == 0)
        .then(pl.lit(float("nan")))
        .when(n_nonzero == 1)
        .then(pl.lit(1.0))
        .otherwise(1.0 - h_expr / n_nonzero.cast(pl.Float64).log(base=2))
    )

    # Dominant: alphabetically first column equal to row max (R12).
    # Build chain in reverse alphabetical order so the alphabetically-first
    # category ends up as the outermost (first-evaluated) when-branch.
    max_val = pl.max_horizontal(*[pl.col(c) for c in categories])
    dominant_expr = pl.lit(None).cast(pl.String)
    for c in sorted(categories, reverse=True):
        dominant_expr = pl.when(pl.col(c) == max_val).then(pl.lit(c)).otherwise(dominant_expr)
    dominant_expr = pl.when(max_val > 0).then(dominant_expr).otherwise(pl.lit(None).cast(pl.String))

    breadth_expr = pl.sum_horizontal(
        *[pl.when(pl.col(c) > presence_threshold).then(1).otherwise(0) for c in categories]
    ).cast(pl.Int64)

    return wide.select(
        *index_cols,
        ri_expr.alias("ri"),
        dominant_expr.alias("dominant"),
        breadth_expr.alias("breadth"),
    )


def _compute_pooled_grouping(
    df: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Compute grouping metrics without subject dimension (vectorized)."""
    per_grouping = df.group_by(["elementId", COL_GROUPING]).agg(
        pl.col("frequency").mean().alias("meanFreq")
    )
    if per_grouping.is_empty():
        return pl.DataFrame()

    wide = per_grouping.pivot(on=COL_GROUPING, index="elementId", values="meanFreq").fill_null(0.0)
    return _grouping_metrics_from_wide(wide, categories, presence_threshold, index_cols=["elementId"])


def _compute_per_subject_grouping(
    per_subject_grouping: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Compute per-subject RI, dominant, breadth (vectorized)."""
    if per_subject_grouping.is_empty():
        return pl.DataFrame()

    wide = per_subject_grouping.pivot(
        on=COL_GROUPING, index=["elementId", COL_SUBJECT], values="meanFreq"
    ).fill_null(0.0)
    return _grouping_metrics_from_wide(
        wide, categories, presence_threshold, index_cols=["elementId", COL_SUBJECT]
    )


def _consensus_dominant(
    dominants: list,
    group_mean_freqs: dict[str, float] | None = None,
) -> str | None:
    """Mode of dominants. R18: ties broken by highest mean frequency, then alphabetically."""
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
    # Tie-break: highest mean frequency (R18), then alphabetical as final fallback
    if group_mean_freqs:
        tied.sort(key=lambda g: (-group_mean_freqs.get(g, 0.0), g))
        return tied[0]
    return sorted(tied)[0]


def _compute_temporal_for_element(
    element_id: str,
    tp_freq: dict[str, float],
    timepoint_order: list[str],
    t_count: int,
) -> dict:
    """Reference implementation of temporal metrics for a single element.

    The hot path uses the vectorized ``_temporal_metrics_from_wide``; this
    function stays for spec-traceable unit tests and as the canonical formula
    reference.

    All metrics use standard frequencies. Fold-changes (Log2 Peak Delta,
    Log2 Kinetic Delta) are computed between detected timepoints where
    frequency is always nonzero by definition.
    """
    freqs = np.array([tp_freq.get(tp, 0.0) for tp in timepoint_order])

    peak_idx = int(np.argmax(freqs))
    peak_tp = timepoint_order[peak_idx]

    total_freq = freqs.sum()
    if total_freq > 0 and t_count > 1:
        positions = np.arange(t_count, dtype=np.float64)
        tsi = float(np.sum(positions * freqs) / (total_freq * (t_count - 1)))
    else:
        tsi = float("nan")

    nonzero_indices = np.where(freqs > 0)[0]
    if len(nonzero_indices) >= 2:
        first_freq = float(freqs[nonzero_indices[0]])
        last_freq = float(freqs[nonzero_indices[-1]])
        log2kd = math.log2(last_freq / first_freq)
    else:
        log2kd = 0.0

    if len(nonzero_indices) >= 1:
        first_freq = float(freqs[nonzero_indices[0]])
        peak_freq = float(freqs[peak_idx])
        log2pd = math.log2(peak_freq / first_freq) if first_freq > 0 and peak_freq > 0 else 0.0
    else:
        log2pd = 0.0

    return {
        "elementId": element_id,
        "peakTimepoint": peak_tp,
        "temporalShiftIndex": tsi,
        "log2KineticDelta": log2kd,
        "log2PeakDelta": log2pd,
    }


def _temporal_metrics_from_wide(
    wide: pl.DataFrame,
    timepoint_order: list[str],
    index_cols: list[str],
) -> pl.DataFrame:
    """Compute temporal metrics as Polars expressions over a wide DataFrame.

    wide has one row per (index_cols) and one column per timepoint (meanFreq,
    zero-filled). Implements R14 (peak timepoint, first-occurrence tie-break),
    R15 (TSI), R16 (Log2PD), R16a (Log2KD).
    """
    # Ensure every timepoint exists as a column (pivot omits timepoints absent
    # from all rows). Add zero columns for missing ones.
    missing = [tp for tp in timepoint_order if tp not in wide.columns]
    if missing:
        wide = wide.with_columns(*[pl.lit(0.0).alias(tp) for tp in missing])

    t = len(timepoint_order)
    tp_cols = timepoint_order

    # Max frequency per row — also the peak frequency by definition.
    max_freq = pl.max_horizontal(*[pl.col(tp) for tp in tp_cols])

    # Peak timepoint: first tp (in order) whose value equals max. Build the
    # when/then chain in reverse order so index 0 ends up outermost.
    peak_tp_expr = pl.lit(None).cast(pl.String)
    for i in reversed(range(t)):
        peak_tp_expr = (
            pl.when(pl.col(tp_cols[i]) == max_freq)
            .then(pl.lit(tp_cols[i]))
            .otherwise(peak_tp_expr)
        )

    # TSI = sum(i * f_i) / (sum(f_i) * (T-1)); NaN when total is zero.
    total_freq = pl.sum_horizontal(*[pl.col(tp) for tp in tp_cols])
    weighted_sum = pl.sum_horizontal(*[pl.col(tp) * float(i) for i, tp in enumerate(tp_cols)])
    tsi_expr = (
        pl.when(total_freq > 0)
        .then(weighted_sum / (total_freq * float(t - 1)))
        .otherwise(pl.lit(float("nan")))
    )

    # First-detected index: scan in reverse so earliest match ends up outermost.
    first_idx_expr = pl.lit(None).cast(pl.Int64)
    for i in reversed(range(t)):
        first_idx_expr = (
            pl.when(pl.col(tp_cols[i]) > 0)
            .then(pl.lit(i, dtype=pl.Int64))
            .otherwise(first_idx_expr)
        )

    # Last-detected index: scan forward so latest match ends up outermost.
    last_idx_expr = pl.lit(None).cast(pl.Int64)
    for i in range(t):
        last_idx_expr = (
            pl.when(pl.col(tp_cols[i]) > 0)
            .then(pl.lit(i, dtype=pl.Int64))
            .otherwise(last_idx_expr)
        )

    # Resolve first/last freq by index.
    first_freq_expr = pl.lit(0.0)
    for i in range(t):
        first_freq_expr = (
            pl.when(first_idx_expr == i).then(pl.col(tp_cols[i])).otherwise(first_freq_expr)
        )
    last_freq_expr = pl.lit(0.0)
    for i in range(t):
        last_freq_expr = (
            pl.when(last_idx_expr == i).then(pl.col(tp_cols[i])).otherwise(last_freq_expr)
        )

    # Detected timepoint count.
    n_detected = pl.sum_horizontal(
        *[pl.when(pl.col(tp) > 0).then(1).otherwise(0) for tp in tp_cols]
    )

    # Log2PD: 0 when nothing detected; otherwise log2(peak/first). With any
    # detection first_freq > 0 and peak_freq >= first_freq, so safe.
    log2pd_expr = (
        pl.when(n_detected == 0)
        .then(pl.lit(0.0))
        .otherwise((max_freq / first_freq_expr).log(base=2))
    )
    # Log2KD: 0 with fewer than two detected tps; otherwise log2(last/first).
    log2kd_expr = (
        pl.when(n_detected < 2)
        .then(pl.lit(0.0))
        .otherwise((last_freq_expr / first_freq_expr).log(base=2))
    )

    return wide.select(
        *index_cols,
        peak_tp_expr.alias("peakTimepoint"),
        tsi_expr.alias("temporalShiftIndex"),
        log2kd_expr.alias("log2KineticDelta"),
        log2pd_expr.alias("log2PeakDelta"),
    )


def compute_temporal_metrics(
    df: pl.DataFrame,
    timepoint_order: list[str],
    has_subject: bool,
    mode: str,
    min_subject_count: int,
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Compute peak timepoint, TSI, log2 kinetic delta, log2 peak delta.

    Fold-changes use standard frequencies at detected timepoints (always nonzero).

    R3: Returns (aggregated_metrics, per_subject_metrics_or_none). per_subject
    metrics hold per-(elementId, subject) temporal values for Table-block
    inspection and are returned only in intra-subject mode with has_subject.
    """
    if len(timepoint_order) < 2:
        return pl.DataFrame(), None

    # Exclude samples with missing timepoint
    df = df.filter(pl.col(COL_TIMEPOINT).is_not_null() & (pl.col(COL_TIMEPOINT) != ""))
    df = df.filter(pl.col(COL_TIMEPOINT).is_in(timepoint_order))

    if not has_subject or mode == "population":
        per_tp = df.group_by(["elementId", COL_TIMEPOINT]).agg(
            pl.col("frequency").mean().alias("meanFreq")
        )
        if per_tp.is_empty():
            return pl.DataFrame(), None
        wide = per_tp.pivot(on=COL_TIMEPOINT, index="elementId", values="meanFreq").fill_null(0.0)
        result = _temporal_metrics_from_wide(wide, timepoint_order, index_cols=["elementId"])
        return result.sort("elementId"), None

    # Intra-subject: per-subject pivot, then aggregate across subjects.
    per_tp = df.group_by(["elementId", COL_SUBJECT, COL_TIMEPOINT]).agg(
        pl.col("frequency").mean().alias("meanFreq")
    )
    if per_tp.is_empty():
        return pl.DataFrame(), None

    wide = per_tp.pivot(
        on=COL_TIMEPOINT, index=["elementId", COL_SUBJECT], values="meanFreq"
    ).fill_null(0.0)
    ps_df = _temporal_metrics_from_wide(
        wide, timepoint_order, index_cols=["elementId", COL_SUBJECT]
    ).sort(["elementId", COL_SUBJECT])

    if ps_df.is_empty():
        return pl.DataFrame(), ps_df

    nan = float("nan")
    agg = (
        ps_df.group_by("elementId")
        .agg(
            # Deterministic mode: alphabetically first among most frequent.
            pl.col("peakTimepoint").mode().sort().first().alias("peakTimepoint"),
            pl.col("temporalShiftIndex").mean().alias("temporalShiftIndex"),
            pl.col("log2KineticDelta").mean().alias("log2KineticDelta"),
            pl.col("log2PeakDelta").mean().alias("log2PeakDelta"),
            pl.len().alias("_nSubjects"),
        )
        .with_columns(
            pl.when(pl.col("_nSubjects") >= min_subject_count)
            .then(pl.col("temporalShiftIndex"))
            .otherwise(pl.lit(nan))
            .fill_null(pl.lit(nan))
            .alias("temporalShiftIndex"),
            pl.when(pl.col("_nSubjects") >= min_subject_count)
            .then(pl.col("log2KineticDelta"))
            .otherwise(pl.lit(nan))
            .fill_null(pl.lit(nan))
            .alias("log2KineticDelta"),
            pl.when(pl.col("_nSubjects") >= min_subject_count)
            .then(pl.col("log2PeakDelta"))
            .otherwise(pl.lit(nan))
            .fill_null(pl.lit(nan))
            .alias("log2PeakDelta"),
        )
        .drop("_nSubjects")
        .select("elementId", "peakTimepoint", "temporalShiftIndex", "log2KineticDelta", "log2PeakDelta")
    )
    return agg.sort("elementId"), ps_df


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
    """Build temporal line plot data for top N clones ranked by |Log2 Peak Delta|."""
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
    if per_tp.is_empty():
        return pl.DataFrame()

    # Reuse the vectorized temporal metrics pivot to compute Log2PD for ranking.
    wide = per_tp.pivot(on=COL_TIMEPOINT, index="elementId", values="meanFreq").fill_null(0.0)
    metrics = _temporal_metrics_from_wide(wide, timepoint_order, index_cols=["elementId"])

    scores_df = (
        metrics.select("elementId", pl.col("log2PeakDelta").abs().alias("score"))
        .sort(["score", "elementId"], descending=[True, False])
        .head(top_n)
    )
    top_elements = scores_df["elementId"].to_list()

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
    timepoint_order = [_js_str(t) for t in json.loads(args.timepoint_order)]
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
    per_subject_grouping = None
    if args.has_grouping:
        grouping, per_subject_grouping = compute_grouping_metrics(
            df, has_subject, mode, args.presence_threshold, args.min_subject_count
        )
        if len(grouping) > 0:
            grouping.write_csv(f"{prefix}_grouping.csv")

        heatmap = build_heatmap_data(
            df, grouping if grouping is not None and len(grouping) > 0 else None, top_n=args.top_n
        )
        if len(heatmap) > 0:
            heatmap.write_csv(f"{prefix}_heatmap.csv")

    # Temporal metrics
    per_subject_temporal = None
    if args.has_timepoint and len(timepoint_order) >= 2:
        temporal, per_subject_temporal = compute_temporal_metrics(
            df, timepoint_order, has_subject, mode, args.min_subject_count
        )
        if len(temporal) > 0:
            temporal.write_csv(f"{prefix}_temporal.csv")

        line_data = build_temporal_line_data(df, timepoint_order, args.top_n)
        if len(line_data) > 0:
            line_data.write_csv(f"{prefix}_temporal_line.csv")

    # R3: Per-subject detail export — only in intra-subject mode with subject
    if mode == "intra-subject" and has_subject:
        per_subject = None
        if per_subject_grouping is not None and len(per_subject_grouping) > 0:
            per_subject = per_subject_grouping
        if per_subject_temporal is not None and len(per_subject_temporal) > 0:
            if per_subject is None:
                per_subject = per_subject_temporal
            else:
                per_subject = per_subject.join(
                    per_subject_temporal, on=["elementId", COL_SUBJECT], how="full", coalesce=True
                )
        if per_subject is not None and len(per_subject) > 0:
            per_subject = per_subject.sort(["elementId", COL_SUBJECT])
            per_subject.write_csv(f"{prefix}_per_subject.csv")

    # Combined main table — start from prevalence if available, else unique elementIds
    if prevalence is not None:
        main_table = prevalence
    else:
        main_table = df.select("elementId").unique().sort("elementId")

    if args.has_grouping:
        try:
            g = pl.read_csv(f"{prefix}_grouping.csv")
            main_table = main_table.join(g, on="elementId", how="left")
        except Exception:
            pass

    if args.has_timepoint and len(timepoint_order) >= 2:
        try:
            t = pl.read_csv(f"{prefix}_temporal.csv")
            main_table = main_table.join(t, on="elementId", how="left")
        except Exception:
            pass

    main_table = main_table.sort("elementId")
    main_table.write_csv(f"{prefix}_main.csv")

    print(f"Analysis complete. Mode: {mode}, Normalization: {args.normalization}")
    print(f"Results written with prefix: {prefix}")


if __name__ == "__main__":
    main()
