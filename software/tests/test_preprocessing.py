"""Behavioral tests for preprocessing: input reading, filtering, replicate averaging (R7a-R7c).

Run: uv run pytest tests/test_preprocessing.py -v
"""

import polars as pl
import pytest

from compartment_analysis import average_replicates, read_input


class TestReadInput:
    """Tests for read_input — CSV reading, null handling, min abundance filter."""

    # R7c: clone with peak abundance below threshold excluded everywhere
    def test_min_abundance_filter(self, tmp_path):
        csv = tmp_path / "input.csv"
        csv.write_text(
            "sampleId,elementId,abundance\n"
            "s1,a,100\n"
            "s2,a,50\n"
            "s1,b,3\n"
            "s2,b,2\n"
        )
        result = read_input(str(csv), has_grouping=False, has_timepoint=False, min_abundance_threshold=10.0)
        # Clone "b" has max abundance 3 < 10 → excluded
        assert set(result["elementId"].to_list()) == {"a"}

    # R7c: clone above threshold in ANY sample is kept
    def test_min_abundance_keeps_if_above_in_one_sample(self, tmp_path):
        csv = tmp_path / "input.csv"
        csv.write_text(
            "sampleId,elementId,abundance\n"
            "s1,a,5\n"
            "s2,a,15\n"
        )
        result = read_input(str(csv), has_grouping=False, has_timepoint=False, min_abundance_threshold=10.0)
        # Clone "a" has max=15 >= 10 → kept (both rows)
        assert len(result) == 2

    # Null/NaN abundance values are dropped
    def test_null_abundance_dropped(self, tmp_path):
        csv = tmp_path / "input.csv"
        csv.write_text(
            "sampleId,elementId,abundance\n"
            "s1,a,100\n"
            "s1,b,NaN\n"
            "s1,c,\n"
            "s1,d,50\n"
        )
        result = read_input(str(csv), has_grouping=False, has_timepoint=False, min_abundance_threshold=0.0)
        assert set(result["elementId"].to_list()) == {"a", "d"}

    # Categorical columns cast to String
    def test_categorical_columns_cast_to_string(self, tmp_path):
        csv = tmp_path / "input.csv"
        csv.write_text(
            "sampleId,elementId,abundance,grouping,timepoint\n"
            "s1,a,100,1,2\n"
        )
        result = read_input(str(csv), has_grouping=True, has_timepoint=True, min_abundance_threshold=0.0)
        assert result["grouping"].dtype == pl.String
        assert result["timepoint"].dtype == pl.String

    # Zero min_abundance_threshold keeps everything
    def test_zero_threshold_keeps_all(self, tmp_path):
        csv = tmp_path / "input.csv"
        csv.write_text(
            "sampleId,elementId,abundance\n"
            "s1,a,0.001\n"
            "s1,b,1000\n"
        )
        result = read_input(str(csv), has_grouping=False, has_timepoint=False, min_abundance_threshold=0.0)
        assert len(result) == 2


class TestAverageReplicates:
    """Tests for average_replicates — R7a."""

    # R7a: two samples mapping to same condition → abundance averaged
    def test_replicates_averaged(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s1", "s2"],
            "elementId": ["a", "a", "b", "b"],
            "abundance": [10.0, 20.0, 30.0, 40.0],
            "grouping": ["lung", "lung", "lung", "lung"],
        })
        result = average_replicates(df, has_subject=False, has_grouping=True, has_timepoint=False)
        # Two samples map to same (elementId, grouping) → averaged
        a = result.filter(pl.col("elementId") == "a")
        assert a["abundance"][0] == pytest.approx(15.0)
        b = result.filter(pl.col("elementId") == "b")
        assert b["abundance"][0] == pytest.approx(35.0)

    # R7a: no duplicate conditions → DataFrame passes through unchanged
    def test_no_replicates_passes_through(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "abundance": [10.0, 20.0],
            "grouping": ["lung", "spleen"],
        })
        result = average_replicates(df, has_subject=False, has_grouping=True, has_timepoint=False)
        assert len(result) == 2
        assert "sampleId" in result.columns

    # R7a: averaged rows get deterministic synthetic sampleId
    def test_synthetic_sampleid(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "abundance": [10.0, 20.0],
            "grouping": ["lung", "lung"],
            "subject": ["sub1", "sub1"],
        })
        result = average_replicates(df, has_subject=True, has_grouping=True, has_timepoint=False)
        # Synthetic sampleId built from condition columns (subject|grouping)
        assert "sampleId" in result.columns
        assert result["sampleId"][0] == "sub1|lung"

    # Subject + grouping + timepoint all participate in condition combo
    def test_full_condition_combo(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "a"],
            "abundance": [10.0, 20.0, 30.0],
            "subject": ["sub1", "sub1", "sub1"],
            "grouping": ["lung", "lung", "spleen"],
            "timepoint": ["Day0", "Day0", "Day0"],
        })
        result = average_replicates(df, has_subject=True, has_grouping=True, has_timepoint=True)
        # s1 and s2 share (sub1, lung, Day0) → averaged to 15.0
        # s3 is unique (sub1, spleen, Day0) → 30.0
        lung = result.filter(pl.col("grouping") == "lung")
        spleen = result.filter(pl.col("grouping") == "spleen")
        assert lung["abundance"][0] == pytest.approx(15.0)
        assert spleen["abundance"][0] == pytest.approx(30.0)
