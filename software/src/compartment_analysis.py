"""
Spatiotemporal analysis: computes grouping restriction, temporal kinetics,
and cross-subject convergence metrics for clonal/cluster abundance data.

Input: CSV with columns [sampleId, elementId, abundance, subject?, grouping?, timepoint?]
Output: Multiple CSV files with computed metrics.
"""

import argparse
import json
import math
import sys

import numpy as np
import polars as pl


def parse_args():
    parser = argparse.ArgumentParser(description="Spatiotemporal analysis")
    parser.add_argument("input_file", help="Input CSV file")
    parser.add_argument("--calculation-mode", choices=["population", "intra-subject"],
                        default="population")
    parser.add_argument("--normalization", choices=["relative-frequency", "clr"],
                        default="relative-frequency")
    parser.add_argument("--has-grouping", action="store_true")
    parser.add_argument("--has-timepoint", action="store_true")
    parser.add_argument("--has-subject", action="store_true")
    parser.add_argument("--timepoint-order", type=str, default="[]")
    parser.add_argument("--presence-threshold", type=float, default=0.0)
    parser.add_argument("--pseudo-count", type=float, default=1.0)
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


def read_input(path: str, has_grouping: bool, has_timepoint: bool,
               min_abundance_threshold: float) -> pl.DataFrame:
    """Read input CSV with proper type handling."""
    df = pl.read_csv(path, null_values=ABUNDANCE_NULL_VALUES, infer_schema_length=10000)

    # Ensure abundance is Float64, drop null
    df = df.with_columns(pl.col("abundance").cast(pl.Float64))
    df = df.filter(pl.col("abundance").is_not_null())

    # Apply minimum abundance filter
    if min_abundance_threshold > 0:
        df = df.filter(pl.col("abundance") >= min_abundance_threshold)

    # Force categorical columns to String
    if has_grouping and COL_GROUPING in df.columns:
        df = df.with_columns(pl.col(COL_GROUPING).cast(pl.String))
    if has_timepoint and COL_TIMEPOINT in df.columns:
        df = df.with_columns(pl.col(COL_TIMEPOINT).cast(pl.String))
    if COL_SUBJECT in df.columns:
        df = df.with_columns(pl.col(COL_SUBJECT).cast(pl.String))

    return df


def average_replicates(df: pl.DataFrame, has_subject: bool, has_grouping: bool,
                       has_timepoint: bool) -> pl.DataFrame:
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

    # Average abundance across replicate samples for each condition combo
    averaged = df.group_by(group_cols).agg(
        pl.col("abundance").mean().alias("abundance"),
        pl.col("sampleId").first().alias("sampleId"),
    )
    return averaged


def compute_relative_frequency(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-sample relative frequency from abundance."""
    sample_totals = df.group_by("sampleId").agg(
        pl.col("abundance").sum().alias("sampleTotal")
    )
    df = df.join(sample_totals, on="sampleId")
    df = df.filter(pl.col("sampleTotal") > 0)
    df = df.with_columns(
        (pl.col("abundance") / pl.col("sampleTotal")).alias("frequency")
    ).drop("sampleTotal")
    return df


def compute_clr(df: pl.DataFrame, mode: str, has_subject: bool) -> pl.DataFrame:
    """Compute centered log-ratio transform.
    Global in population mode, per-subject in intra-subject mode."""
    sample_totals = df.group_by("sampleId").agg(
        pl.col("abundance").sum().alias("sampleTotal")
    )
    df = df.join(sample_totals, on="sampleId")
    df = df.filter(pl.col("sampleTotal") > 0)
    df = df.with_columns(
        (pl.col("abundance") / pl.col("sampleTotal")).alias("frequency")
    ).drop("sampleTotal")

    # Multiplicative replacement: delta = 0.65 * min(nonzero) / D
    # D = number of components (distinct elementIds per group)
    if mode == "intra-subject" and has_subject:
        # Per-subject CLR
        result_dfs = []
        for subject, subject_df in df.group_by(COL_SUBJECT):
            result_dfs.append(_apply_clr_to_group(subject_df))
        if result_dfs:
            df = pl.concat(result_dfs)
    else:
        # Global CLR
        df = _apply_clr_to_group(df)

    return df


def _apply_clr_to_group(df: pl.DataFrame) -> pl.DataFrame:
    """Apply CLR transform to a group of samples."""
    min_nonzero = df.filter(pl.col("frequency") > 0)["frequency"].min()
    if min_nonzero is None or min_nonzero <= 0:
        min_nonzero = 1e-10

    def clr_transform(group: pl.DataFrame) -> pl.DataFrame:
        freq = group["frequency"].to_numpy().astype(np.float64)
        D = len(freq)
        delta = 0.65 * float(min_nonzero) / D if D > 0 else 1e-10
        freq = np.where(freq == 0, delta, freq)
        freq = freq / freq.sum()  # renormalize
        log_freq = np.log(freq)
        geo_mean = np.mean(log_freq)
        clr_vals = log_freq - geo_mean
        return group.with_columns(pl.Series("frequency", clr_vals))

    return df.group_by("sampleId", maintain_order=True).map_groups(clr_transform)


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
    if n <= 1:
        return 1.0
    h = shannon_entropy(nonzero)
    return 1.0 - h / math.log2(n)


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
        # No subject: treat all samples as pooled
        return _compute_pooled_grouping(df, categories, presence_threshold)

    per_subject_grouping = (
        df.group_by(["elementId", COL_SUBJECT, COL_GROUPING])
        .agg(pl.col("frequency").mean().alias("meanFreq"))
    )

    per_subject_metrics = _compute_per_subject_grouping(
        per_subject_grouping, categories, presence_threshold
    )

    results = []
    for element_id, group in per_subject_metrics.group_by("elementId"):
        eid = element_id[0] if isinstance(element_id, tuple) else element_id
        ris = group["ri"].drop_nulls().to_numpy()
        dominants = group["dominant"].to_list()
        n_subjects = len(group)

        enough_subjects = n_subjects >= min_subject_count
        mean_ri = float(np.mean(ris)) if enough_subjects and len(ris) > 0 else float("nan")
        std_ri = float(np.std(ris, ddof=1)) if enough_subjects and len(ris) > 1 else float("nan")

        # Consensus dominant: mode, ties broken alphabetically
        consensus_dominant = _consensus_dominant(dominants)

        # Count dominant per category
        count_dominant = {cat: 0 for cat in categories}
        for d in dominants:
            if d is not None and d in count_dominant:
                count_dominant[d] += 1

        breadths = group["breadth"].to_numpy()
        mean_breadth = int(round(float(np.mean(breadths)))) if len(breadths) > 0 else 0

        row: dict = {
            "elementId": eid,
            "ri": mean_ri,
            "dominant": consensus_dominant,
            "breadth": mean_breadth,
        }
        if mode == "population" and has_subject:
            row["consensusDominant"] = consensus_dominant
            row["meanRi"] = mean_ri
            row["stdRi"] = std_ri
        for cat in categories:
            row[f"countDominantIn_{cat}"] = count_dominant[cat]
        results.append(row)

    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results)


def _compute_pooled_grouping(
    df: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Compute grouping metrics without subject dimension."""
    per_grouping = (
        df.group_by(["elementId", COL_GROUPING])
        .agg(pl.col("frequency").mean().alias("meanFreq"))
    )
    results = []
    for element_id, group in per_grouping.group_by("elementId"):
        eid = element_id[0] if isinstance(element_id, tuple) else element_id
        freq_map = dict(zip(group[COL_GROUPING].to_list(), group["meanFreq"].to_list()))
        freq_arr = np.array([freq_map.get(cat, 0.0) for cat in categories])

        ri = restriction_index(freq_arr)
        dominant_idx = int(np.argmax(freq_arr))
        dominant = categories[dominant_idx] if freq_arr[dominant_idx] > 0 else None
        breadth = int(np.sum(freq_arr > presence_threshold))

        results.append({
            "elementId": eid,
            "ri": ri,
            "dominant": dominant,
            "breadth": breadth,
        })

    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results)


def _compute_per_subject_grouping(
    per_subject_grouping: pl.DataFrame,
    categories: list[str],
    presence_threshold: float,
) -> pl.DataFrame:
    """Compute per-subject RI, dominant, breadth."""
    results = []
    for (element_id, subject), group in per_subject_grouping.group_by(
        ["elementId", COL_SUBJECT]
    ):
        freq_map = dict(zip(group[COL_GROUPING].to_list(), group["meanFreq"].to_list()))
        freq_arr = np.array([freq_map.get(cat, 0.0) for cat in categories])

        ri = restriction_index(freq_arr)
        dominant_idx = int(np.argmax(freq_arr))
        dominant = categories[dominant_idx] if freq_arr[dominant_idx] > 0 else None
        # Tie-breaking: alphabetical
        max_freq = freq_arr[dominant_idx]
        if max_freq > 0:
            tied = [categories[i] for i in range(len(categories)) if freq_arr[i] == max_freq]
            dominant = sorted(tied)[0]
        breadth = int(np.sum(freq_arr > presence_threshold))

        results.append({
            "elementId": element_id,
            COL_SUBJECT: subject,
            "ri": ri,
            "dominant": dominant,
            "breadth": breadth,
        })

    if not results:
        return pl.DataFrame()
    return pl.DataFrame(results)


def _consensus_dominant(dominants: list) -> str | None:
    """Mode of dominants, ties broken alphabetically."""
    dom_counts: dict[str, int] = {}
    for d in dominants:
        if d is not None:
            dom_counts[d] = dom_counts.get(d, 0) + 1
    if not dom_counts:
        return None
    max_count = max(dom_counts.values())
    tied = sorted(k for k, v in dom_counts.items() if v == max_count)
    return tied[0]


def compute_temporal_metrics(
    df: pl.DataFrame,
    timepoint_order: list[str],
    has_subject: bool,
    mode: str,
    pseudo_count: float,
    min_subject_count: int,
) -> pl.DataFrame:
    """Compute peak timepoint, TSI, log2 kinetic delta, log2 peak delta."""
    if len(timepoint_order) < 2:
        return pl.DataFrame()

    # Exclude samples with missing timepoint
    df = df.filter(pl.col(COL_TIMEPOINT).is_not_null() & (pl.col(COL_TIMEPOINT) != ""))
    df = df.filter(pl.col(COL_TIMEPOINT).is_in(timepoint_order))

    t_count = len(timepoint_order)

    if not has_subject or mode == "population":
        if has_subject:
            # Average frequency across subjects at each timepoint
            per_tp = (
                df.group_by(["elementId", COL_TIMEPOINT])
                .agg(pl.col("frequency").mean().alias("meanFreq"))
            )
        else:
            per_tp = (
                df.group_by(["elementId", COL_TIMEPOINT])
                .agg(pl.col("frequency").mean().alias("meanFreq"))
            )

        results = []
        for element_id, group in per_tp.group_by("elementId"):
            eid = element_id[0] if isinstance(element_id, tuple) else element_id
            tp_freq = dict(zip(group[COL_TIMEPOINT].to_list(), group["meanFreq"].to_list()))
            row = _compute_temporal_for_element(eid, tp_freq, timepoint_order, t_count, pseudo_count)
            results.append(row)

        if not results:
            return pl.DataFrame()
        return pl.DataFrame(results)

    else:
        # Intra-subject: per-subject metrics then average
        per_tp = (
            df.group_by(["elementId", COL_SUBJECT, COL_TIMEPOINT])
            .agg(pl.col("frequency").mean().alias("meanFreq"))
        )

        per_subject_temporal = []
        for (element_id, subject), group in per_tp.group_by(["elementId", COL_SUBJECT]):
            eid = element_id if not isinstance(element_id, tuple) else element_id
            tp_freq = dict(zip(group[COL_TIMEPOINT].to_list(), group["meanFreq"].to_list()))
            row = _compute_temporal_for_element(eid, tp_freq, timepoint_order, t_count, pseudo_count)
            row[COL_SUBJECT] = subject
            per_subject_temporal.append(row)

        if not per_subject_temporal:
            return pl.DataFrame()

        ps_df = pl.DataFrame(per_subject_temporal)
        results = []
        for element_id, group in ps_df.group_by("elementId"):
            eid = element_id[0] if isinstance(element_id, tuple) else element_id
            n_subjects = len(group)
            enough = n_subjects >= min_subject_count
            row = {
                "elementId": eid,
                "peakTimepoint": group["peakTimepoint"].mode().to_list()[0] if len(group) > 0 else None,
                "temporalShiftIndex": float(group["temporalShiftIndex"].mean()) if enough else float("nan"),
                "log2KineticDelta": float(group["log2KineticDelta"].mean()) if enough else float("nan"),
                "log2PeakDelta": float(group["log2PeakDelta"].mean()) if enough else float("nan"),
            }
            results.append(row)

        if not results:
            return pl.DataFrame()
        return pl.DataFrame(results)


def _compute_temporal_for_element(
    element_id: str,
    tp_freq: dict[str, float],
    timepoint_order: list[str],
    t_count: int,
    pseudo_count: float,
) -> dict:
    """Compute temporal metrics for a single element."""
    freqs = np.array([tp_freq.get(tp, 0.0) for tp in timepoint_order])

    # Peak timepoint
    peak_idx = int(np.argmax(freqs))
    peak_tp = timepoint_order[peak_idx]

    # TSI = sum(rank_i * freq_i) / (sum(freq_i) * (T - 1)), 0-indexed ranks
    total_freq = freqs.sum()
    if total_freq > 0 and t_count > 1:
        ranks = np.arange(t_count, dtype=np.float64)
        tsi = float(np.sum(ranks * freqs) / (total_freq * (t_count - 1)))
    else:
        tsi = float("nan")

    # Log2 kinetic delta: last-to-first
    nonzero_indices = np.where(freqs > 0)[0]
    if len(nonzero_indices) >= 2:
        first_freq = float(freqs[nonzero_indices[0]])
        last_freq = float(freqs[nonzero_indices[-1]])
        log2kd = math.log2((last_freq + pseudo_count) / (first_freq + pseudo_count))
    elif len(nonzero_indices) == 1:
        log2kd = 0.0
    else:
        log2kd = 0.0

    # Log2 peak delta: peak-to-first
    if len(nonzero_indices) >= 1:
        first_freq = float(freqs[nonzero_indices[0]])
        peak_freq = float(freqs[peak_idx])
        if peak_freq > 0 and first_freq > 0:
            log2pd = math.log2((peak_freq + pseudo_count) / (first_freq + pseudo_count))
        else:
            log2pd = 0.0
    else:
        log2pd = 0.0

    return {
        "elementId": element_id,
        "peakTimepoint": peak_tp,
        "temporalShiftIndex": tsi,
        "log2KineticDelta": log2kd,
        "log2PeakDelta": log2pd,
    }


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
    return prevalence


def build_heatmap_data(
    df: pl.DataFrame,
    grouping_metrics: pl.DataFrame | None,
    prevalence: pl.DataFrame | None,
    top_n: int = 50,
) -> pl.DataFrame:
    """Build heatmap for top N convergent multi-compartment clones.

    Selection: breadth >= 2 (present in multiple groups), then ranked by
    subject prevalence (descending) and total abundance (descending).
    These are the clones most relevant for downstream lead selection.
    """
    df = df.filter(pl.col(COL_GROUPING).is_not_null() & (pl.col(COL_GROUPING) != ""))
    heatmap = (
        df.group_by(["elementId", COL_GROUPING])
        .agg(pl.col("frequency").mean().alias("normalizedFrequency"))
    )

    if grouping_metrics is not None and "breadth" in grouping_metrics.columns and len(grouping_metrics) > 0:
        # Start with multi-compartment clones (breadth >= 2)
        candidates = grouping_metrics.filter(pl.col("breadth") >= 2)

        # If too few, fall back to all clones
        if len(candidates) < top_n:
            candidates = grouping_metrics

        # Join prevalence for ranking
        if prevalence is not None and "subjectPrevalence" in prevalence.columns:
            candidates = candidates.join(prevalence.select(["elementId", "subjectPrevalence"]),
                                         on="elementId", how="left")
            candidates = candidates.with_columns(
                pl.col("subjectPrevalence").fill_null(0)
            )
        else:
            candidates = candidates.with_columns(pl.lit(0).alias("subjectPrevalence"))

        # Join total abundance for secondary ranking
        total_abundance = (
            df.group_by("elementId")
            .agg(pl.col("frequency").sum().alias("totalAbundance"))
        )
        candidates = candidates.join(total_abundance, on="elementId", how="left")
        candidates = candidates.with_columns(
            pl.col("totalAbundance").fill_null(0.0)
        )

        top_elements = (
            candidates.sort(
                ["subjectPrevalence", "totalAbundance"],
                descending=[True, True],
            )
            .head(top_n)["elementId"]
            .to_list()
        )
        heatmap = heatmap.filter(pl.col("elementId").is_in(top_elements))

    return heatmap.rename({COL_GROUPING: "groupCategory"})


def build_temporal_line_data(
    df: pl.DataFrame,
    timepoint_order: list[str],
    top_n: int,
    pseudo_count: float,
) -> pl.DataFrame:
    """Build temporal line plot data for top N clones ranked by Log2 Peak Delta."""
    df = df.filter(
        pl.col(COL_TIMEPOINT).is_not_null()
        & (pl.col(COL_TIMEPOINT) != "")
        & pl.col(COL_TIMEPOINT).is_in(timepoint_order)
    )

    t_count = len(timepoint_order)
    if t_count < 2:
        line_data = (
            df.group_by(["elementId", COL_TIMEPOINT])
            .agg(pl.col("frequency").mean().alias("frequency"))
        )
        return line_data.rename({COL_TIMEPOINT: "timepointValue"})

    # Compute Log2 Peak Delta per element for ranking
    per_tp = (
        df.group_by(["elementId", COL_TIMEPOINT])
        .agg(pl.col("frequency").mean().alias("meanFreq"))
    )

    element_scores = []
    for element_id, group in per_tp.group_by("elementId"):
        eid = element_id[0] if isinstance(element_id, tuple) else element_id
        tp_freq = dict(zip(group[COL_TIMEPOINT].to_list(), group["meanFreq"].to_list()))
        freqs = np.array([tp_freq.get(tp, 0.0) for tp in timepoint_order])
        peak_idx = int(np.argmax(freqs))
        nonzero = np.where(freqs > 0)[0]
        if len(nonzero) >= 1:
            first_freq = float(freqs[nonzero[0]])
            peak_freq = float(freqs[peak_idx])
            log2pd = abs(math.log2((peak_freq + pseudo_count) / (first_freq + pseudo_count))) if first_freq > 0 and peak_freq > 0 else 0.0
        else:
            log2pd = 0.0
        element_scores.append({"elementId": eid, "score": log2pd})

    if not element_scores:
        return pl.DataFrame()

    scores_df = pl.DataFrame(element_scores).sort("score", descending=True).head(top_n)
    top_elements = set(scores_df["elementId"].to_list())

    # Filter to top N elements
    line_data = (
        df.filter(pl.col("elementId").is_in(list(top_elements)))
        .group_by(["elementId", COL_TIMEPOINT])
        .agg(pl.col("frequency").mean().alias("frequency"))
    )
    return line_data.rename({COL_TIMEPOINT: "timepointValue"})


def build_prevalence_histogram(prevalence_df: pl.DataFrame) -> pl.DataFrame:
    return (
        prevalence_df.group_by("subjectPrevalence")
        .agg(pl.len().alias("cloneCount"))
        .rename({"subjectPrevalence": "prevalenceCount"})
        .sort("prevalenceCount")
    )


def main():
    args = parse_args()

    df = read_input(args.input_file, args.has_grouping, args.has_timepoint,
                    args.min_abundance_threshold)
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

    # Subject prevalence (always computed)
    prevalence = compute_subject_prevalence(df, has_subject)
    prevalence.write_csv(f"{prefix}_prevalence.csv")

    # Prevalence histogram
    histogram = build_prevalence_histogram(prevalence)
    histogram.write_csv(f"{prefix}_prevalence_histogram.csv")

    # Grouping metrics
    grouping = None
    if args.has_grouping:
        grouping = compute_grouping_metrics(df, has_subject, mode,
                                            args.presence_threshold, args.min_subject_count)
        if len(grouping) > 0:
            grouping.write_csv(f"{prefix}_grouping.csv")

        heatmap = build_heatmap_data(
            df,
            grouping if grouping is not None and len(grouping) > 0 else None,
            prevalence if len(prevalence) > 0 else None,
            top_n=args.top_n,
        )
        if len(heatmap) > 0:
            heatmap.write_csv(f"{prefix}_heatmap.csv")

    # Temporal metrics
    if args.has_timepoint and len(timepoint_order) >= 2:
        temporal = compute_temporal_metrics(df, timepoint_order, has_subject, mode,
                                           args.pseudo_count, args.min_subject_count)
        if len(temporal) > 0:
            temporal.write_csv(f"{prefix}_temporal.csv")

        line_data = build_temporal_line_data(df, timepoint_order, args.top_n,
                                            args.pseudo_count)
        if len(line_data) > 0:
            line_data.write_csv(f"{prefix}_temporal_line.csv")

    # Combined main table
    main_table = prevalence

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

    main_table.write_csv(f"{prefix}_main.csv")

    print(f"Analysis complete. Mode: {mode}, Normalization: {args.normalization}")
    print(f"Results written with prefix: {prefix}")


if __name__ == "__main__":
    main()
