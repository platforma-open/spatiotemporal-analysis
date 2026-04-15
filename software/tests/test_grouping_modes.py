"""Behavioral tests for grouping metrics across population/intra-subject modes (R2, R3, R11-R13, R17b-R21).

Run: uv run pytest tests/test_grouping_modes.py -v
"""

import math

import polars as pl
import pytest

from compartment_analysis import (
    _compute_per_subject_grouping,
    _compute_pooled_grouping,
    _consensus_dominant,
    compute_grouping_metrics,
)


class TestPooledGrouping:
    """Tests for _compute_pooled_grouping — no subject dimension."""

    # R11: RI computed from pooled frequency across all samples
    def test_basic_ri(self):
        df = pl.DataFrame({
            "elementId": ["a", "a"],
            "grouping": ["tissue1", "tissue2"],
            "frequency": [0.8, 0.2],
        })
        result = _compute_pooled_grouping(df, ["tissue1", "tissue2"], presence_threshold=0.0)
        row = result.row(0, named=True)
        assert 0.0 < row["ri"] < 1.0

    # R12: dominant is argmax; tie broken alphabetically
    def test_dominant_is_argmax(self):
        df = pl.DataFrame({
            "elementId": ["a", "a", "a"],
            "grouping": ["lung", "spleen", "lymph"],
            "frequency": [0.1, 0.7, 0.2],
        })
        result = _compute_pooled_grouping(df, ["lung", "lymph", "spleen"], presence_threshold=0.0)
        assert result["dominant"][0] == "spleen"

    # R12: tied frequencies → alphabetically first wins (explicit tie-breaking)
    def test_dominant_tie_alphabetical(self):
        df = pl.DataFrame({
            "elementId": ["a", "a"],
            "grouping": ["zeta", "alpha"],
            "frequency": [0.5, 0.5],
        })
        result = _compute_pooled_grouping(df, ["alpha", "zeta"], presence_threshold=0.0)
        assert result["dominant"][0] == "alpha"

    # R12: tie-breaking is explicit alphabetical — not dependent on categories list order
    def test_dominant_tie_alphabetical_unsorted_categories(self):
        df = pl.DataFrame({
            "elementId": ["a", "a"],
            "grouping": ["zeta", "alpha"],
            "frequency": [0.5, 0.5],
        })
        # Pass categories in reverse alphabetical order — alpha should still win
        result = _compute_pooled_grouping(df, ["zeta", "alpha"], presence_threshold=0.0)
        assert result["dominant"][0] == "alpha"

    # R13: breadth counts groups above presence threshold
    def test_breadth_counts_above_threshold(self):
        df = pl.DataFrame({
            "elementId": ["a", "a", "a"],
            "grouping": ["g1", "g2", "g3"],
            "frequency": [0.5, 0.3, 0.2],
        })
        result = _compute_pooled_grouping(df, ["g1", "g2", "g3"], presence_threshold=0.0)
        assert result["breadth"][0] == 3

    # R13: breadth with nonzero threshold
    def test_breadth_with_nonzero_threshold(self):
        df = pl.DataFrame({
            "elementId": ["a", "a", "a"],
            "grouping": ["g1", "g2", "g3"],
            "frequency": [0.5, 0.005, 0.005],
        })
        result = _compute_pooled_grouping(df, ["g1", "g2", "g3"], presence_threshold=0.01)
        # Only g1 (0.5) is above 0.01
        assert result["breadth"][0] == 1


class TestPerSubjectGrouping:
    """Tests for _compute_per_subject_grouping — per-subject RI/dominant/breadth."""

    # Computes independently per (element, subject) pair
    def test_independent_per_subject(self):
        df = pl.DataFrame({
            "elementId": ["a", "a", "a", "a"],
            "subject": ["sub1", "sub1", "sub2", "sub2"],
            "grouping": ["lung", "spleen", "lung", "spleen"],
            "meanFreq": [0.9, 0.1, 0.5, 0.5],
        })
        result = _compute_per_subject_grouping(df, ["lung", "spleen"], presence_threshold=0.0)
        sub1 = result.filter(pl.col("subject") == "sub1")
        sub2 = result.filter(pl.col("subject") == "sub2")
        # Sub1: highly restricted to lung
        assert sub1["ri"][0] > 0.5
        assert sub1["dominant"][0] == "lung"
        # Sub2: uniform → RI ~ 0
        assert sub2["ri"][0] == pytest.approx(0.0, abs=1e-10)

    # R12: per-subject dominant tie-breaking is alphabetical
    def test_per_subject_dominant_tie_alphabetical(self):
        df = pl.DataFrame({
            "elementId": ["a", "a"],
            "subject": ["sub1", "sub1"],
            "grouping": ["beta", "alpha"],
            "meanFreq": [0.5, 0.5],
        })
        result = _compute_per_subject_grouping(df, ["alpha", "beta"], presence_threshold=0.0)
        assert result["dominant"][0] == "alpha"


class TestConsensusDominant:
    """Tests for _consensus_dominant — R18."""

    # Mode of dominants (no tie)
    def test_mode_is_consensus(self):
        assert _consensus_dominant(["lung", "lung", "spleen"]) == "lung"

    # R18: tie broken by highest mean frequency across subjects
    def test_tie_broken_by_frequency(self):
        # spleen and lung each dominant once → count tie
        # spleen has higher mean frequency → wins
        group_mean_freqs = {"lung": 0.3, "spleen": 0.7}
        assert _consensus_dominant(["spleen", "lung"], group_mean_freqs) == "spleen"

    # R18: ties with equal mean frequency → alphabetical fallback
    def test_tie_broken_alphabetically_when_equal_freqs(self):
        group_mean_freqs = {"lung": 0.5, "spleen": 0.5}
        assert _consensus_dominant(["spleen", "lung"], group_mean_freqs) == "lung"

    # Fallback: no frequencies provided → alphabetical
    def test_tie_broken_alphabetically_without_freqs(self):
        assert _consensus_dominant(["spleen", "lung"]) == "lung"

    # All None → None
    def test_all_none(self):
        assert _consensus_dominant([None, None]) is None

    # Single entry
    def test_single_entry(self):
        assert _consensus_dominant(["lung"]) == "lung"

    # Mixed None and values
    def test_mixed_none(self):
        assert _consensus_dominant([None, "lung", None, "lung"]) == "lung"


class TestComputeGroupingMetrics:
    """Tests for compute_grouping_metrics — full pipeline with subject handling."""

    # No subject: uses pooled path directly
    def test_no_subject_uses_pooled(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.6, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.4, "grouping": "spleen"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=False, mode="population",
                                          presence_threshold=0.0, min_subject_count=2)
        assert len(result) == 1
        assert "ri" in result.columns
        assert "dominant" in result.columns
        assert "breadth" in result.columns

    # With subject: produces consensus columns
    def test_with_subject_has_consensus_columns(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "spleen", "subject": "sub2"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        assert "consensusDominant" in result.columns
        assert "meanRi" in result.columns
        assert "stdRi" in result.columns
        row = result.row(0, named=True)
        assert row["consensusDominant"] == "lung"

    # R20: Mean RI is the arithmetic mean of per-subject RIs (hand-calculated)
    def test_mean_ri_is_arithmetic_mean(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.9, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": "spleen", "subject": "sub2"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Sub1: p=[0.9, 0.1], H = -(0.9*log2(0.9) + 0.1*log2(0.1)), RI = 1 - H/log2(2)
        h_sub1 = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
        ri_sub1 = 1.0 - h_sub1 / math.log2(2)
        # Sub2: p=[0.5, 0.5] → uniform → RI = 0.0
        ri_sub2 = 0.0
        expected_mean = (ri_sub1 + ri_sub2) / 2
        assert row["meanRi"] == pytest.approx(expected_mean, abs=1e-10)

    # R17b: min_subject_count threshold → NaN when too few subjects
    def test_min_subject_count_nans(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=2)
        row = result.row(0, named=True)
        assert math.isnan(row["meanRi"])
        assert math.isnan(row["stdRi"])

    # R21: single subject → StdDev = NaN
    def test_single_subject_std_nan(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        assert math.isnan(row["stdRi"])

    # R18: consensus dominant tie broken by highest mean frequency across subjects
    def test_consensus_dominant_tie_broken_by_frequency(self, make_df):
        # 2 subjects: sub1 dominant=lung, sub2 dominant=spleen → 1-1 count tie
        # lung mean freq across subjects = (0.9 + 0.4) / 2 = 0.65
        # spleen mean freq across subjects = (0.1 + 0.6) / 2 = 0.35
        # lung has higher mean frequency → consensus = lung
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.9, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.4, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.6, "grouping": "spleen", "subject": "sub2"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        assert row["consensusDominant"] == "lung"

    # R19: count dominant per category
    def test_count_dominant_in(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.9, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "spleen", "subject": "sub2"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub3"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub3"},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Sub1: lung dominant, Sub2: spleen dominant, Sub3: lung dominant
        assert row["countDominantIn_lung"] == 2
        assert row["countDominantIn_spleen"] == 1

    # R3: per-subject detail DataFrame returned in intra-subject mode with subject
    def test_per_subject_returned_with_subject(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "spleen", "subject": "sub2"},
        ])
        result, per_subject = compute_grouping_metrics(df, has_subject=True, mode="intra-subject",
                                          presence_threshold=0.0, min_subject_count=1)
        assert per_subject is not None
        assert len(per_subject) == 2  # 1 element x 2 subjects
        assert "elementId" in per_subject.columns
        assert "subject" in per_subject.columns
        assert "ri" in per_subject.columns
        assert "dominant" in per_subject.columns
        assert "breadth" in per_subject.columns
        subs = sorted(per_subject["subject"].to_list())
        assert subs == ["sub1", "sub2"]

    # R3: population mode with subject also returns per-subject detail (intermediate)
    def test_per_subject_returned_in_population_mode_with_subject(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "spleen", "subject": "sub2"},
        ])
        result, per_subject = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        # Even in population mode, per-subject metrics exist as intermediate
        assert per_subject is not None
        assert len(per_subject) == 2

    # No subject → per_subject is None
    def test_per_subject_none_when_no_subject(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.6, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.4, "grouping": "spleen"},
        ])
        result, per_subject = compute_grouping_metrics(df, has_subject=False, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        assert per_subject is None

    # Empty grouping values are excluded
    def test_empty_grouping_excluded(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.5, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": ""},
        ])
        result, _ = compute_grouping_metrics(df, has_subject=False, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Only "lung" category exists → RI = 1.0
        assert row["ri"] == pytest.approx(1.0)
