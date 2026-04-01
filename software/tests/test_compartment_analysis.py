"""Behavioral tests for compartment_analysis.py.

All tests focus on observable behavior (inputs -> outputs) rather than
implementation details. Synthetic data is constructed inline for each test.

Original specs for this block are located in 'docs/text/work/projects/in-vivo-compartment-analysis/'

Run from software/:
    uv sync
    uv run pytest tests/
"""

import math
import os

import numpy as np
import polars as pl
import pytest

from compartment_analysis import (
    average_replicates,
    build_heatmap_data,
    build_prevalence_histogram,
    build_temporal_line_data,
    compute_clr,
    compute_grouping_metrics,
    compute_relative_frequency,
    compute_subject_prevalence,
    compute_temporal_metrics,
    read_input,
    restriction_index,
    shannon_entropy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(df: pl.DataFrame, tmp_path: str) -> str:
    path = os.path.join(tmp_path, "input.csv")
    df.write_csv(path)
    return path


def _simple_df(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Input reading & preprocessing
# ---------------------------------------------------------------------------


class TestReadInput:
    def test_reads_csv_and_casts_abundance(self, tmp_path):
        df = pl.DataFrame(
            {
                "sampleId": ["s1", "s1", "s2"],
                "elementId": ["c1", "c2", "c1"],
                "abundance": ["100", "200", "150"],
            }
        )
        path = _write_csv(df, str(tmp_path))
        result = read_input(path, has_grouping=False, has_timepoint=False, min_abundance_threshold=0.0)
        assert result["abundance"].dtype == pl.Float64
        assert len(result) == 3

    def test_drops_null_abundance(self, tmp_path):
        df = pl.DataFrame(
            {
                "sampleId": ["s1", "s1"],
                "elementId": ["c1", "c2"],
                "abundance": ["100", "NaN"],
            }
        )
        path = _write_csv(df, str(tmp_path))
        result = read_input(path, has_grouping=False, has_timepoint=False, min_abundance_threshold=0.0)
        assert len(result) == 1
        assert result["elementId"][0] == "c1"

    def test_min_abundance_filter_keeps_clone_above_threshold_in_any_sample(self, tmp_path):
        df = pl.DataFrame(
            {
                "sampleId": ["s1", "s2", "s1", "s2"],
                "elementId": ["c1", "c1", "c2", "c2"],
                "abundance": [5.0, 100.0, 3.0, 4.0],
            }
        )
        path = _write_csv(df, str(tmp_path))
        result = read_input(path, has_grouping=False, has_timepoint=False, min_abundance_threshold=10.0)
        assert set(result["elementId"].to_list()) == {"c1"}

    def test_min_abundance_zero_keeps_all(self, tmp_path):
        df = pl.DataFrame(
            {
                "sampleId": ["s1", "s2"],
                "elementId": ["c1", "c2"],
                "abundance": [1.0, 2.0],
            }
        )
        path = _write_csv(df, str(tmp_path))
        result = read_input(path, has_grouping=False, has_timepoint=False, min_abundance_threshold=0.0)
        assert len(result) == 2

    def test_integer_metadata_works_in_downstream_grouping(self, tmp_path):
        # Grouping and timepoint columns provided as integers should work end-to-end
        df = pl.DataFrame(
            {
                "sampleId": ["s1", "s2"],
                "elementId": ["c1", "c1"],
                "abundance": [80.0, 20.0],
                "grouping": [1, 2],
                "timepoint": [7, 14],
            }
        )
        path = _write_csv(df, str(tmp_path))
        result = read_input(path, has_grouping=True, has_timepoint=True, min_abundance_threshold=0.0)
        freq_df = compute_relative_frequency(result)
        grouping = compute_grouping_metrics(
            freq_df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        assert len(grouping) == 1
        assert grouping["dominant"][0] == "1"


# ---------------------------------------------------------------------------
# 2. Sample averaging
# ---------------------------------------------------------------------------


class TestAverageReplicates:
    def test_averages_replicate_samples(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "subject": "m1", "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 200.0, "subject": "m1", "grouping": "lung"},
            ]
        )
        result = average_replicates(df, has_subject=True, has_grouping=True, has_timepoint=False)
        assert len(result) == 1
        assert result["abundance"][0] == pytest.approx(150.0)

    def test_no_replicates_passes_through(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "subject": "m1", "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 200.0, "subject": "m1", "grouping": "spleen"},
            ]
        )
        result = average_replicates(df, has_subject=True, has_grouping=True, has_timepoint=False)
        assert len(result) == 2

    def test_averaged_replicates_normalize_correctly(self):
        # Two replicates for the same condition: abundances 100 and 200 → averaged to 150
        # Second condition has abundance 50. After averaging, normalization should use 150+50=200.
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "subject": "m1", "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 200.0, "subject": "m1", "grouping": "lung"},
                {"sampleId": "s1", "elementId": "c2", "abundance": 50.0, "subject": "m1", "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c2", "abundance": 50.0, "subject": "m1", "grouping": "lung"},
            ]
        )
        averaged = average_replicates(df, has_subject=True, has_grouping=True, has_timepoint=False)
        result = compute_relative_frequency(averaged)
        freq_map = dict(zip(result["elementId"].to_list(), result["frequency"].to_list()))
        # c1 averaged = 150, c2 averaged = 50, total = 200
        assert freq_map["c1"] == pytest.approx(0.75)
        assert freq_map["c2"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 3. Normalization
# ---------------------------------------------------------------------------


class TestRelativeFrequency:
    def test_frequencies_sum_to_one_per_sample(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0},
                {"sampleId": "s1", "elementId": "c2", "abundance": 300.0},
                {"sampleId": "s2", "elementId": "c1", "abundance": 50.0},
                {"sampleId": "s2", "elementId": "c2", "abundance": 50.0},
            ]
        )
        result = compute_relative_frequency(df)
        for sid in ["s1", "s2"]:
            sample = result.filter(pl.col("sampleId") == sid)
            assert sample["frequency"].sum() == pytest.approx(1.0)

    def test_correct_proportions(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 25.0},
                {"sampleId": "s1", "elementId": "c2", "abundance": 75.0},
            ]
        )
        result = compute_relative_frequency(df)
        freq_map = dict(zip(result["elementId"].to_list(), result["frequency"].to_list()))
        assert freq_map["c1"] == pytest.approx(0.25)
        assert freq_map["c2"] == pytest.approx(0.75)

    def test_zero_total_sample_excluded(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0},
            ]
        )
        result = compute_relative_frequency(df)
        # s2 has total 0, should be excluded
        assert result["sampleId"].to_list() == ["s1"]


class TestCLR:
    def test_clr_values_sum_to_zero_per_sample(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0},
                {"sampleId": "s1", "elementId": "c2", "abundance": 200.0},
                {"sampleId": "s1", "elementId": "c3", "abundance": 300.0},
            ]
        )
        result = compute_clr(df, mode="population", has_subject=False)
        assert result["frequency"].sum() == pytest.approx(0.0, abs=1e-10)

    def test_clr_handles_zero_abundance_via_replacement(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0},
                {"sampleId": "s1", "elementId": "c2", "abundance": 0.0},
                {"sampleId": "s1", "elementId": "c3", "abundance": 200.0},
            ]
        )
        result = compute_clr(df, mode="population", has_subject=False)
        # Should not have any NaN/inf values
        assert result["frequency"].is_nan().sum() == 0
        assert result["frequency"].is_infinite().sum() == 0

    def test_clr_per_subject_in_intra_subject_mode(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "subject": "m1"},
                {"sampleId": "s1", "elementId": "c2", "abundance": 200.0, "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 50.0, "subject": "m2"},
                {"sampleId": "s2", "elementId": "c2", "abundance": 150.0, "subject": "m2"},
            ]
        )
        result = compute_clr(df, mode="intra-subject", has_subject=True)
        # CLR values should sum to ~0 within each sample
        for sid in ["s1", "s2"]:
            sample = result.filter(pl.col("sampleId") == sid)
            assert sample["frequency"].sum() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 4. Grouping metrics
# ---------------------------------------------------------------------------


class TestRestrictionIndex:
    def test_single_group_returns_one(self):
        assert restriction_index(np.array([1.0])) == 1.0

    def test_uniform_distribution_returns_zero(self):
        assert restriction_index(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.0)

    def test_all_in_one_group_returns_one(self):
        assert restriction_index(np.array([1.0, 0.0, 0.0])) == 1.0

    def test_two_groups_half_half(self):
        ri = restriction_index(np.array([0.5, 0.5]))
        assert ri == pytest.approx(0.0)

    def test_skewed_distribution(self):
        ri = restriction_index(np.array([0.9, 0.1]))
        assert 0.0 < ri < 1.0

    def test_all_zeros_returns_nan(self):
        ri = restriction_index(np.array([0.0, 0.0, 0.0]))
        assert math.isnan(ri)


class TestShannonEntropy:
    def test_single_element(self):
        assert shannon_entropy(np.array([1.0])) == 0.0

    def test_uniform_two_elements(self):
        assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)

    def test_empty_array(self):
        assert shannon_entropy(np.array([])) == 0.0


class TestGroupingMetrics:
    def _build_grouped_df(self, rows):
        """Build a frequency-ready DataFrame with grouping column."""
        df = _simple_df(rows)
        return compute_relative_frequency(df)

    def test_dominant_group_is_highest_frequency(self):
        df = self._build_grouped_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen"},
                {"sampleId": "s1", "elementId": "c2", "abundance": 10.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c2", "abundance": 90.0, "grouping": "spleen"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        c2 = result.filter(pl.col("elementId") == "c2")
        assert c1["dominant"][0] == "lung"
        assert c2["dominant"][0] == "spleen"

    def test_dominant_group_tie_broken_alphabetically_pooled(self):
        # Equal frequencies in both groups — alphabetical first wins
        df = self._build_grouped_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 50.0, "grouping": "brain"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 50.0, "grouping": "lung"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        # With equal mean freq, argmax picks the first in the array order.
        # Categories are sorted alphabetically: ["brain", "lung"]
        # Equal freq → argmax gives index 0 → "brain"
        assert result["dominant"][0] == "brain"

    def test_breadth_counts_groups_above_threshold(self):
        df = self._build_grouped_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 1.0, "grouping": "spleen"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 0.0, "grouping": "brain"},
                # Need other elements so samples have nonzero totals
                {"sampleId": "s3", "elementId": "c_other", "abundance": 100.0, "grouping": "brain"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        # c1 present in lung and spleen but not brain (0 abundance)
        assert c1["breadth"][0] == 2

    def test_breadth_with_nonzero_threshold(self):
        df = self._build_grouped_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 1.0, "grouping": "spleen"},
                {"sampleId": "s2", "elementId": "c_other", "abundance": 99.0, "grouping": "spleen"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.05, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        # spleen frequency for c1 = 1/100 = 0.01, below threshold of 0.05
        assert c1["breadth"][0] == 1

    def test_ri_one_for_single_group_clone(self):
        df = self._build_grouped_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "grouping": "spleen"},
                {"sampleId": "s2", "elementId": "c_other", "abundance": 100.0, "grouping": "spleen"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["ri"][0] == pytest.approx(1.0)

    def test_empty_grouping_values_excluded(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.5, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.5, "grouping": ""},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["dominant"][0] == "lung"


# ---------------------------------------------------------------------------
# 5. Temporal metrics
# ---------------------------------------------------------------------------


class TestTemporalMetrics:
    def _build_temporal_df(self, rows):
        df = _simple_df(rows)
        return compute_relative_frequency(df)

    def test_peak_timepoint_correct(self):
        # Background element ensures per-sample frequencies vary meaningfully
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0, "timepoint": "D0"},
                {"sampleId": "s1", "elementId": "bg", "abundance": 90.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 100.0, "timepoint": "D7"},
                {"sampleId": "s2", "elementId": "bg", "abundance": 100.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 30.0, "timepoint": "D14"},
                {"sampleId": "s3", "elementId": "bg", "abundance": 70.0, "timepoint": "D14"},
            ]
        )
        result = compute_temporal_metrics(
            df, ["D0", "D7", "D14"], has_subject=False, mode="population", min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["peakTimepoint"][0] == "D7"

    def test_tsi_zero_when_all_mass_at_first_timepoint(self):
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 0.0, "timepoint": "D14"},
                # Need nonzero totals for other samples
                {"sampleId": "s2", "elementId": "c_other", "abundance": 100.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c_other", "abundance": 100.0, "timepoint": "D14"},
            ]
        )
        result = compute_temporal_metrics(
            df, ["D0", "D7", "D14"], has_subject=False, mode="population", min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(0.0)

    def test_tsi_one_when_all_mass_at_last_timepoint(self):
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 0.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 100.0, "timepoint": "D14"},
                {"sampleId": "s1", "elementId": "c_other", "abundance": 100.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c_other", "abundance": 100.0, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(
            df, ["D0", "D7", "D14"], has_subject=False, mode="population", min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(1.0)

    def test_tsi_intermediate_value(self):
        # Equal abundance at D0 (pos=0) and D14 (pos=2), 3 timepoints
        # TSI = (0*f + 1*0 + 2*f) / (2f * 2) = 2f / 4f = 0.5
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 50.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 50.0, "timepoint": "D14"},
                {"sampleId": "s2", "elementId": "c_other", "abundance": 100.0, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(
            df, ["D0", "D7", "D14"], has_subject=False, mode="population", min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(0.5, abs=0.05)

    def test_log2pd_zero_for_single_timepoint_clone(self):
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "timepoint": "D7"},
                {"sampleId": "s2", "elementId": "c_other", "abundance": 100.0, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["log2PeakDelta"][0] == pytest.approx(0.0)

    def test_log2pd_with_multiple_elements(self):
        # s1: c1=10, c_bg=90 → freq_c1 = 0.1
        # s2: c1=80, c_bg=20 → freq_c1 = 0.8
        # Log2PD = log2(0.8/0.1) = log2(8) = 3.0
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0, "timepoint": "D0"},
                {"sampleId": "s1", "elementId": "c_bg", "abundance": 90.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 80.0, "timepoint": "D7"},
                {"sampleId": "s2", "elementId": "c_bg", "abundance": 20.0, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["log2PeakDelta"][0] == pytest.approx(3.0)

    def test_log2kd_negative_for_contracting_clone(self):
        # s1(D0): c1=80, bg=20 → freq=0.8
        # s2(D7): c1=10, bg=90 → freq=0.1
        # Log2KD = log2(0.1/0.8) = log2(0.125) = -3.0
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 80.0, "timepoint": "D0"},
                {"sampleId": "s1", "elementId": "c_bg", "abundance": 20.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "timepoint": "D7"},
                {"sampleId": "s2", "elementId": "c_bg", "abundance": 90.0, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["log2KineticDelta"][0] == pytest.approx(-3.0)

    def test_fewer_than_two_timepoints_returns_empty(self):
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "timepoint": "D0"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0"], has_subject=False, mode="population", min_subject_count=2)
        assert len(result) == 0

    def test_timepoints_not_in_order_are_excluded(self):
        df = self._build_temporal_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 50.0, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 50.0, "timepoint": "D7"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 50.0, "timepoint": "EXCLUDED"},
            ]
        )
        # Only D0 and D7 in the order — EXCLUDED samples should be dropped
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 6. Consensus & convergence metrics
# ---------------------------------------------------------------------------


class TestSubjectPrevalence:
    def test_counts_distinct_subjects(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0, "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 20.0, "subject": "m2"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 30.0, "subject": "m3"},
                {"sampleId": "s1", "elementId": "c2", "abundance": 5.0, "subject": "m1"},
                {"sampleId": "s2", "elementId": "c2", "abundance": 0.0, "subject": "m2"},
                {"sampleId": "s3", "elementId": "c2", "abundance": 0.0, "subject": "m3"},
            ]
        )
        result = compute_subject_prevalence(df, has_subject=True)
        c1 = result.filter(pl.col("elementId") == "c1")
        c2 = result.filter(pl.col("elementId") == "c2")
        assert c1["subjectPrevalence"][0] == 3
        assert c2["subjectPrevalence"][0] == 1

    def test_prevalence_fraction(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0, "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 20.0, "subject": "m2"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 0.0, "subject": "m3"},
            ]
        )
        result = compute_subject_prevalence(df, has_subject=True)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["subjectPrevalenceFraction"][0] == pytest.approx(2.0 / 3.0)

    def test_without_subject_counts_samples(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0},
                {"sampleId": "s2", "elementId": "c1", "abundance": 20.0},
                {"sampleId": "s3", "elementId": "c1", "abundance": 0.0},
            ]
        )
        result = compute_subject_prevalence(df, has_subject=False)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["subjectPrevalence"][0] == 2  # s1 and s2 (s3 has 0)


class TestConsensusMetrics:
    def _build_consensus_df(self, rows):
        """Build a frequency-ready DataFrame with subject and grouping."""
        df = _simple_df(rows)
        return compute_relative_frequency(df)

    def test_consensus_dominant_is_mode_across_subjects(self):
        # c1 is dominant in lung for m1 and m2, spleen for m3
        df = self._build_consensus_df(
            [
                # m1: c1 dominant in lung
                {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen", "subject": "m1"},
                # m2: c1 dominant in lung
                {"sampleId": "s3", "elementId": "c1", "abundance": 80.0, "grouping": "lung", "subject": "m2"},
                {"sampleId": "s4", "elementId": "c1", "abundance": 20.0, "grouping": "spleen", "subject": "m2"},
                # m3: c1 dominant in spleen
                {"sampleId": "s5", "elementId": "c1", "abundance": 10.0, "grouping": "lung", "subject": "m3"},
                {"sampleId": "s6", "elementId": "c1", "abundance": 90.0, "grouping": "spleen", "subject": "m3"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"

    def test_consensus_dominant_tie_broken_by_mean_frequency(self):
        # c1: m1 dominant=lung, m2 dominant=spleen — tie in count
        # lung mean freq should be higher than spleen
        df = self._build_consensus_df(
            [
                # m1: c1 in lung=95, spleen=5
                {"sampleId": "s1", "elementId": "c1", "abundance": 95.0, "grouping": "lung", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 5.0, "grouping": "spleen", "subject": "m1"},
                # m2: c1 in lung=40, spleen=60
                {"sampleId": "s3", "elementId": "c1", "abundance": 40.0, "grouping": "lung", "subject": "m2"},
                {"sampleId": "s4", "elementId": "c1", "abundance": 60.0, "grouping": "spleen", "subject": "m2"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        # m1 dominant=lung, m2 dominant=spleen → tie 1:1
        # lung mean freq across subjects: mean(95/100, 40/100) = mean(0.95, 0.4) = 0.675
        # spleen mean freq: mean(5/100, 60/100) = mean(0.05, 0.6) = 0.325
        # → lung wins by mean frequency
        assert c1["consensusDominant"][0] == "lung"

    def test_count_dominant_in_per_category(self):
        # Background elements ensure relative frequencies are meaningful per sample
        df = self._build_consensus_df(
            [
                # m1: lung sample — c1=90 of 100 → freq=0.9
                {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung", "subject": "m1"},
                {"sampleId": "s1", "elementId": "bg", "abundance": 10.0, "grouping": "lung", "subject": "m1"},
                # m1: spleen sample — c1=10 of 100 → freq=0.1
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen", "subject": "m1"},
                {"sampleId": "s2", "elementId": "bg", "abundance": 90.0, "grouping": "spleen", "subject": "m1"},
                # m2: lung — c1=80/100=0.8
                {"sampleId": "s3", "elementId": "c1", "abundance": 80.0, "grouping": "lung", "subject": "m2"},
                {"sampleId": "s3", "elementId": "bg", "abundance": 20.0, "grouping": "lung", "subject": "m2"},
                # m2: spleen — c1=20/100=0.2
                {"sampleId": "s4", "elementId": "c1", "abundance": 20.0, "grouping": "spleen", "subject": "m2"},
                {"sampleId": "s4", "elementId": "bg", "abundance": 80.0, "grouping": "spleen", "subject": "m2"},
                # m3: lung — c1=30/100=0.3
                {"sampleId": "s5", "elementId": "c1", "abundance": 30.0, "grouping": "lung", "subject": "m3"},
                {"sampleId": "s5", "elementId": "bg", "abundance": 70.0, "grouping": "lung", "subject": "m3"},
                # m3: spleen — c1=70/100=0.7
                {"sampleId": "s6", "elementId": "c1", "abundance": 70.0, "grouping": "spleen", "subject": "m3"},
                {"sampleId": "s6", "elementId": "bg", "abundance": 30.0, "grouping": "spleen", "subject": "m3"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        # m1: lung(0.9) > spleen(0.1) → dominant=lung
        # m2: lung(0.8) > spleen(0.2) → dominant=lung
        # m3: spleen(0.7) > lung(0.3) → dominant=spleen
        assert c1["countDominantIn_lung"][0] == 2
        assert c1["countDominantIn_spleen"][0] == 1

    def test_mean_ri_nan_below_min_subject_count(self):
        df = self._build_consensus_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen", "subject": "m1"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=3
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert math.isnan(c1["meanRi"][0])

    def test_std_ri_nan_for_single_subject(self):
        df = self._build_consensus_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen", "subject": "m1"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=1
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert math.isnan(c1["stdRi"][0])


class TestTemporalMetricsIntraSubject:
    def _build_temporal_subject_df(self, rows):
        df = _simple_df(rows)
        return compute_relative_frequency(df)

    def test_intra_subject_averages_across_subjects(self):
        df = self._build_temporal_subject_df(
            [
                # m1: c1 at D0=80/100, D7=20/100
                {"sampleId": "s1", "elementId": "c1", "abundance": 80.0, "timepoint": "D0", "subject": "m1"},
                {"sampleId": "s1", "elementId": "bg", "abundance": 20.0, "timepoint": "D0", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 20.0, "timepoint": "D7", "subject": "m1"},
                {"sampleId": "s2", "elementId": "bg", "abundance": 80.0, "timepoint": "D7", "subject": "m1"},
                # m2: c1 at D0=60/100, D7=40/100
                {"sampleId": "s3", "elementId": "c1", "abundance": 60.0, "timepoint": "D0", "subject": "m2"},
                {"sampleId": "s3", "elementId": "bg", "abundance": 40.0, "timepoint": "D0", "subject": "m2"},
                {"sampleId": "s4", "elementId": "c1", "abundance": 40.0, "timepoint": "D7", "subject": "m2"},
                {"sampleId": "s4", "elementId": "bg", "abundance": 60.0, "timepoint": "D7", "subject": "m2"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=True, mode="intra-subject", min_subject_count=2)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert len(c1) == 1
        # Both subjects have peak at D0, so consensus peak = D0
        assert c1["peakTimepoint"][0] == "D0"

    def test_intra_subject_nan_below_min_subject_count(self):
        df = self._build_temporal_subject_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 80.0, "timepoint": "D0", "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 20.0, "timepoint": "D7", "subject": "m1"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=True, mode="intra-subject", min_subject_count=3)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert math.isnan(c1["temporalShiftIndex"][0])


# ---------------------------------------------------------------------------
# 7. Visualization data
# ---------------------------------------------------------------------------


class TestBuildHeatmapData:
    def test_returns_top_n_by_ri(self):
        # Build frequency-ready data
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 1.0, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.0, "grouping": "spleen"},
                {"sampleId": "s1", "elementId": "c2", "frequency": 0.5, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c2", "frequency": 0.5, "grouping": "spleen"},
                {"sampleId": "s1", "elementId": "c3", "frequency": 0.8, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c3", "frequency": 0.2, "grouping": "spleen"},
            ]
        )
        grouping = pl.DataFrame(
            [
                {"elementId": "c1", "ri": 1.0},
                {"elementId": "c2", "ri": 0.0},
                {"elementId": "c3", "ri": 0.5},
            ]
        )
        result = build_heatmap_data(df, grouping, top_n=2)
        # Top 2 by RI: c1 (1.0) and c3 (0.5)
        element_ids = set(result["elementId"].to_list())
        assert element_ids == {"c1", "c3"}

    def test_heatmap_has_correct_columns(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.7, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.3, "grouping": "spleen"},
            ]
        )
        grouping = pl.DataFrame([{"elementId": "c1", "ri": 0.5}])
        result = build_heatmap_data(df, grouping, top_n=10)
        assert "elementId" in result.columns
        assert "groupCategory" in result.columns
        assert "normalizedFrequency" in result.columns


class TestBuildTemporalLineData:
    def test_returns_top_n_by_log2pd(self):
        # c1: big expansion, c2: no expansion, c3: moderate
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.01, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.8, "timepoint": "D7"},
                {"sampleId": "s1", "elementId": "c2", "frequency": 0.5, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c2", "frequency": 0.5, "timepoint": "D7"},
                {"sampleId": "s1", "elementId": "c3", "frequency": 0.1, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c3", "frequency": 0.3, "timepoint": "D7"},
            ]
        )
        result = build_temporal_line_data(df, ["D0", "D7"], top_n=2)
        element_ids = set(result["elementId"].to_list())
        # c1 has highest abs(log2pd), c3 has next highest
        assert "c1" in element_ids
        assert len(element_ids) == 2


class TestBuildPrevalenceHistogram:
    def test_histogram_counts(self):
        prevalence = pl.DataFrame(
            [
                {"elementId": "c1", "subjectPrevalence": 1},
                {"elementId": "c2", "subjectPrevalence": 1},
                {"elementId": "c3", "subjectPrevalence": 3},
                {"elementId": "c4", "subjectPrevalence": 2},
                {"elementId": "c5", "subjectPrevalence": 3},
            ]
        )
        result = build_prevalence_histogram(prevalence)
        hist = dict(zip(result["prevalenceCount"].to_list(), result["cloneCount"].to_list()))
        assert hist[1] == 2
        assert hist[2] == 1
        assert hist[3] == 2


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_subject_produces_pooled_grouping(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.8, "grouping": "lung"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.2, "grouping": "spleen"},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        assert len(result) == 1
        assert "consensusDominant" not in result.columns

    def test_all_subjects_same_dominant(self):
        df = compute_relative_frequency(
            _simple_df(
                [
                    {"sampleId": "s1", "elementId": "c1", "abundance": 90.0, "grouping": "lung", "subject": "m1"},
                    {"sampleId": "s2", "elementId": "c1", "abundance": 10.0, "grouping": "spleen", "subject": "m1"},
                    {"sampleId": "s3", "elementId": "c1", "abundance": 80.0, "grouping": "lung", "subject": "m2"},
                    {"sampleId": "s4", "elementId": "c1", "abundance": 20.0, "grouping": "spleen", "subject": "m2"},
                ]
            )
        )
        result = compute_grouping_metrics(
            df, has_subject=True, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"
        assert c1["countDominantIn_lung"][0] == 2
        assert c1["countDominantIn_spleen"][0] == 0

    def test_clone_only_in_one_subject_prevalence_still_counted(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "abundance": 10.0, "subject": "m1"},
                {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "subject": "m2"},
                {"sampleId": "s3", "elementId": "c1", "abundance": 0.0, "subject": "m3"},
            ]
        )
        result = compute_subject_prevalence(df, has_subject=True)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["subjectPrevalence"][0] == 1

    def test_log2kd_zero_for_single_detected_timepoint(self):
        df = compute_relative_frequency(
            _simple_df(
                [
                    {"sampleId": "s1", "elementId": "c1", "abundance": 100.0, "timepoint": "D0"},
                    {"sampleId": "s2", "elementId": "c1", "abundance": 0.0, "timepoint": "D7"},
                    {"sampleId": "s2", "elementId": "bg", "abundance": 100.0, "timepoint": "D7"},
                ]
            )
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        c1 = result.filter(pl.col("elementId") == "c1")
        assert c1["log2KineticDelta"][0] == pytest.approx(0.0)

    def test_missing_timepoint_values_excluded(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.5, "timepoint": "D0"},
                {"sampleId": "s2", "elementId": "c1", "frequency": 0.5, "timepoint": ""},
                {"sampleId": "s3", "elementId": "c1", "frequency": 0.5, "timepoint": "D7"},
            ]
        )
        result = compute_temporal_metrics(df, ["D0", "D7"], has_subject=False, mode="population", min_subject_count=2)
        assert len(result) > 0

    def test_empty_grouping_produces_empty_result(self):
        df = _simple_df(
            [
                {"sampleId": "s1", "elementId": "c1", "frequency": 0.5, "grouping": ""},
            ]
        )
        result = compute_grouping_metrics(
            df, has_subject=False, mode="population", presence_threshold=0.0, min_subject_count=2
        )
        assert len(result) == 0
