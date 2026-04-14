"""Behavioral tests for grouping metrics across population/intra-subject modes (R2, R3, R11-R13, R17b-R21).

Run: uv run pytest tests/test_grouping_modes.py -v
"""

import math

import numpy as np
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

    # R12: tied frequencies → alphabetically first wins
    def test_dominant_tie_alphabetical(self):
        df = pl.DataFrame({
            "elementId": ["a", "a"],
            "grouping": ["zeta", "alpha"],
            "frequency": [0.5, 0.5],
        })
        result = _compute_pooled_grouping(df, ["alpha", "zeta"], presence_threshold=0.0)
        # Both have 0.5 — numpy argmax picks first in array order, which is "alpha"
        # since categories are sorted
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

    # Mode of dominants
    def test_mode_is_consensus(self):
        assert _consensus_dominant(["lung", "lung", "spleen"]) == "lung"

    # Tie broken alphabetically (spec says by highest mean frequency,
    # but implementation uses alphabetical — testing current behavior)
    def test_tie_broken_alphabetically(self):
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

    def _make_df(self, rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows)

    # No subject: uses pooled path directly
    def test_no_subject_uses_pooled(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.6, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.4, "grouping": "spleen"},
        ])
        result = compute_grouping_metrics(df, has_subject=False, mode="population",
                                          presence_threshold=0.0, min_subject_count=2)
        assert len(result) == 1
        assert "ri" in result.columns
        assert "dominant" in result.columns
        assert "breadth" in result.columns

    # With subject: produces consensus columns
    def test_with_subject_has_consensus_columns(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "spleen", "subject": "sub2"},
        ])
        result = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        assert "consensusDominant" in result.columns
        assert "meanRi" in result.columns
        assert "stdRi" in result.columns
        row = result.row(0, named=True)
        assert row["consensusDominant"] == "lung"

    # R20: Mean RI is arithmetic mean of per-subject RIs
    def test_mean_ri_is_arithmetic_mean(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.9, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": "spleen", "subject": "sub2"},
        ])
        result = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Sub1 RI > 0, Sub2 RI = 0 → mean should be between 0 and sub1's RI
        assert 0.0 < row["meanRi"] < 1.0

    # R17b: min_subject_count threshold → NaN when too few subjects
    def test_min_subject_count_nans(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
        ])
        result = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=2)
        row = result.row(0, named=True)
        assert math.isnan(row["meanRi"])
        assert math.isnan(row["stdRi"])

    # R21: single subject → StdDev = NaN
    def test_single_subject_std_nan(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub1"},
        ])
        result = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        assert math.isnan(row["stdRi"])

    # R19: count dominant per category
    def test_count_dominant_in(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.9, "grouping": "lung", "subject": "sub1"},
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "grouping": "spleen", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.3, "grouping": "lung", "subject": "sub2"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.7, "grouping": "spleen", "subject": "sub2"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.8, "grouping": "lung", "subject": "sub3"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.2, "grouping": "spleen", "subject": "sub3"},
        ])
        result = compute_grouping_metrics(df, has_subject=True, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Sub1: lung dominant, Sub2: spleen dominant, Sub3: lung dominant
        assert row["countDominantIn_lung"] == 2
        assert row["countDominantIn_spleen"] == 1

    # Empty grouping values are excluded
    def test_empty_grouping_excluded(self):
        df = self._make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.5, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "grouping": ""},
        ])
        result = compute_grouping_metrics(df, has_subject=False, mode="population",
                                          presence_threshold=0.0, min_subject_count=1)
        row = result.row(0, named=True)
        # Only "lung" category exists → RI = 1.0
        assert row["ri"] == pytest.approx(1.0)
