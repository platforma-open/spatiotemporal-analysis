"""Behavioral tests for subject prevalence and histogram (R17, R17a, R30).

Run: uv run pytest tests/test_subject_prevalence.py -v
"""

import polars as pl
import pytest

from compartment_analysis import build_prevalence_histogram, compute_subject_prevalence


class TestSubjectPrevalence:
    """Tests for compute_subject_prevalence — R17, R17a."""

    # R17: counts distinct subjects per element
    def test_prevalence_counts_distinct_subjects(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3", "s4"],
            "elementId": ["a", "a", "a", "b"],
            "abundance": [10.0, 20.0, 30.0, 5.0],
            "subject": ["sub1", "sub2", "sub3", "sub1"],
        })
        result = compute_subject_prevalence(df, has_subject=True)
        a = result.filter(pl.col("elementId") == "a")
        b = result.filter(pl.col("elementId") == "b")
        assert a["subjectPrevalence"][0] == 3
        assert b["subjectPrevalence"][0] == 1

    # R17a: fraction = prevalence / total_subjects
    def test_prevalence_fraction(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "b"],
            "abundance": [10.0, 20.0, 5.0],
            "subject": ["sub1", "sub2", "sub3"],
        })
        result = compute_subject_prevalence(df, has_subject=True)
        a = result.filter(pl.col("elementId") == "a")
        # 2 out of 3 subjects
        assert a["subjectPrevalenceFraction"][0] == pytest.approx(2.0 / 3.0)

    # Without subject column: counts distinct sampleIds
    def test_prevalence_without_subject_counts_samples(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "a"],
            "abundance": [10.0, 20.0, 30.0],
        })
        result = compute_subject_prevalence(df, has_subject=False)
        assert result["subjectPrevalence"][0] == 3

    # Zero abundance excluded from prevalence count
    def test_zero_abundance_not_counted(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "abundance": [10.0, 0.0],
            "subject": ["sub1", "sub2"],
        })
        result = compute_subject_prevalence(df, has_subject=True)
        assert result["subjectPrevalence"][0] == 1

    # Fraction is 1.0 when all subjects have the clone
    def test_fraction_all_subjects(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "abundance": [10.0, 20.0],
            "subject": ["sub1", "sub2"],
        })
        result = compute_subject_prevalence(df, has_subject=True)
        assert result["subjectPrevalenceFraction"][0] == pytest.approx(1.0)


class TestPrevalenceHistogram:
    """Tests for build_prevalence_histogram — R30."""

    # Correct histogram bins
    def test_histogram_shape(self):
        prevalence_df = pl.DataFrame({
            "elementId": ["a", "b", "c", "d", "e"],
            "subjectPrevalence": [1, 1, 2, 3, 3],
        })
        result = build_prevalence_histogram(prevalence_df)
        assert "prevalenceCount" in result.columns
        assert "cloneCount" in result.columns
        # prevalence 1 → 2 clones, 2 → 1 clone, 3 → 2 clones
        r = result.sort("prevalenceCount")
        assert r.filter(pl.col("prevalenceCount") == 1)["cloneCount"][0] == 2
        assert r.filter(pl.col("prevalenceCount") == 2)["cloneCount"][0] == 1
        assert r.filter(pl.col("prevalenceCount") == 3)["cloneCount"][0] == 2

    # Single prevalence value → one bin
    def test_single_bin(self):
        prevalence_df = pl.DataFrame({
            "elementId": ["a", "b"],
            "subjectPrevalence": [1, 1],
        })
        result = build_prevalence_histogram(prevalence_df)
        assert len(result) == 1
        assert result["cloneCount"][0] == 2
