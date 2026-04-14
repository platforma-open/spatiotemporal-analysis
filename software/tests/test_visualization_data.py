"""Behavioral tests for visualization data builders (R28-R30).

Run: uv run pytest tests/test_visualization_data.py -v
"""

import polars as pl
import pytest

from compartment_analysis import (
    build_heatmap_data,
    build_temporal_line_data,
)


class TestBuildHeatmapData:
    """Tests for build_heatmap_data — R28: heatmap matrix for top clones by RI."""

    # R28: output has elementId, groupCategory, normalizedFrequency columns
    def test_output_columns(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1"],
            "elementId": ["a", "a"],
            "frequency": [0.6, 0.4],
            "grouping": ["lung", "spleen"],
        })
        result = build_heatmap_data(df, grouping_metrics=None, top_n=50)
        assert "elementId" in result.columns
        assert "groupCategory" in result.columns
        assert "normalizedFrequency" in result.columns

    # R28: without grouping metrics, all elements included (up to top_n)
    def test_no_grouping_metrics_includes_all(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s1", "s1"],
            "elementId": ["a", "a", "b", "b"],
            "frequency": [0.6, 0.4, 0.3, 0.7],
            "grouping": ["lung", "spleen", "lung", "spleen"],
        })
        result = build_heatmap_data(df, grouping_metrics=None, top_n=50)
        assert set(result["elementId"].to_list()) == {"a", "b"}

    # R28: with grouping metrics, top N by RI selected
    def test_top_n_by_ri(self):
        df = pl.DataFrame({
            "sampleId": ["s1"] * 6,
            "elementId": ["a", "a", "b", "b", "c", "c"],
            "frequency": [0.9, 0.1, 0.5, 0.5, 0.7, 0.3],
            "grouping": ["lung", "spleen", "lung", "spleen", "lung", "spleen"],
        })
        grouping_metrics = pl.DataFrame({
            "elementId": ["a", "b", "c"],
            "ri": [0.9, 0.0, 0.5],
        })
        result = build_heatmap_data(df, grouping_metrics, top_n=2)
        # Top 2 by RI: "a" (0.9) and "c" (0.5)
        assert set(result["elementId"].to_list()) == {"a", "c"}

    # R28: empty grouping values excluded
    def test_empty_grouping_excluded(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1"],
            "elementId": ["a", "a"],
            "frequency": [0.6, 0.4],
            "grouping": ["lung", ""],
        })
        result = build_heatmap_data(df, grouping_metrics=None, top_n=50)
        assert "groupCategory" in result.columns
        # Only "lung" remains
        assert result["groupCategory"].to_list() == ["lung"]

    # R28: null grouping values excluded
    def test_null_grouping_excluded(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1"],
            "elementId": ["a", "a"],
            "frequency": [0.6, 0.4],
            "grouping": ["lung", None],
        })
        result = build_heatmap_data(df, grouping_metrics=None, top_n=50)
        assert len(result) == 1

    # Frequencies averaged across samples for same (element, grouping)
    def test_frequencies_averaged(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "frequency": [0.6, 0.4],
            "grouping": ["lung", "lung"],
        })
        result = build_heatmap_data(df, grouping_metrics=None, top_n=50)
        assert result["normalizedFrequency"][0] == pytest.approx(0.5)


class TestBuildTemporalLineData:
    """Tests for build_temporal_line_data — R29: temporal line plot data."""

    # R29: output columns are elementId, timepointValue, frequency
    def test_output_columns(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2"],
            "elementId": ["a", "a"],
            "frequency": [0.3, 0.7],
            "timepoint": ["Day0", "Day7"],
        })
        result = build_temporal_line_data(df, ["Day0", "Day7"], top_n=10)
        assert "elementId" in result.columns
        assert "timepointValue" in result.columns
        assert "frequency" in result.columns

    # R29: top N ranked by Log2 Peak Delta
    def test_top_n_by_log2pd(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s1", "s2", "s1", "s2"],
            "elementId": ["a", "a", "b", "b", "c", "c"],
            "frequency": [0.1, 0.9, 0.5, 0.5, 0.2, 0.8],
            "timepoint": ["Day0", "Day7", "Day0", "Day7", "Day0", "Day7"],
        })
        result = build_temporal_line_data(df, ["Day0", "Day7"], top_n=2)
        elements = set(result["elementId"].to_list())
        # "a" has log2(0.9/0.1) = 3.17, "c" has log2(0.8/0.2) = 2.0, "b" has log2(0.5/0.5) = 0.0
        assert "a" in elements
        assert "c" in elements
        assert "b" not in elements

    # R29: single timepoint → returns all data without ranking
    def test_single_timepoint_no_ranking(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1"],
            "elementId": ["a", "b"],
            "frequency": [0.3, 0.7],
            "timepoint": ["Day0", "Day0"],
        })
        result = build_temporal_line_data(df, ["Day0"], top_n=10)
        assert len(result) == 2

    # R29: empty/null timepoints filtered out
    def test_empty_timepoint_filtered(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "a"],
            "frequency": [0.3, 0.5, 0.2],
            "timepoint": ["Day0", "Day7", ""],
        })
        result = build_temporal_line_data(df, ["Day0", "Day7"], top_n=10)
        assert "" not in result["timepointValue"].to_list()

    # R29: timepoints not in order are filtered out
    def test_out_of_order_timepoints_filtered(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "a"],
            "frequency": [0.3, 0.5, 0.2],
            "timepoint": ["Day0", "Day7", "Day999"],
        })
        result = build_temporal_line_data(df, ["Day0", "Day7"], top_n=10)
        assert "Day999" not in result["timepointValue"].to_list()

    # Multiple samples at same timepoint averaged
    def test_multiple_samples_averaged(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s2", "s3"],
            "elementId": ["a", "a", "a"],
            "frequency": [0.2, 0.4, 0.9],
            "timepoint": ["Day0", "Day0", "Day7"],
        })
        result = build_temporal_line_data(df, ["Day0", "Day7"], top_n=10)
        day0 = result.filter(pl.col("timepointValue") == "Day0")
        assert day0["frequency"][0] == pytest.approx(0.3)
