"""End-to-end integration tests for the full analysis pipeline.

Tests the complete flow: read_input → normalize → compute metrics → write output.

Run: uv run pytest tests/test_end_to_end.py -v
"""

import json
import math

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
)


def _run_pipeline(
    tmp_path,
    csv_text,
    *,
    has_grouping=False,
    has_timepoint=False,
    has_subject=False,
    normalization="relative-frequency",
    timepoint_order=None,
    presence_threshold=0.0,
    min_abundance_threshold=0.0,
    min_subject_count=2,
    top_n=20,
):
    """Run the full analysis pipeline on CSV text, returning all result DataFrames."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(csv_text)

    if timepoint_order is None:
        timepoint_order = []

    df = read_input(str(csv_file), has_grouping, has_timepoint, min_abundance_threshold)
    has_subject = has_subject and "subject" in df.columns

    df = average_replicates(df, has_subject, has_grouping, has_timepoint)

    if normalization == "clr":
        df = compute_clr(df, "population", has_subject)
    else:
        df = compute_relative_frequency(df)

    results = {"normalized": df}

    if has_subject:
        prevalence = compute_subject_prevalence(df, has_subject)
        results["prevalence"] = prevalence
        results["histogram"] = build_prevalence_histogram(prevalence)

    if has_grouping:
        grouping, _ = compute_grouping_metrics(df, has_subject, "population", presence_threshold, min_subject_count)
        results["grouping"] = grouping
        if len(grouping) > 0:
            results["heatmap"] = build_heatmap_data(df, grouping, top_n)

    if has_timepoint and len(timepoint_order) >= 2:
        temporal, _ = compute_temporal_metrics(df, timepoint_order, has_subject, "population", min_subject_count)
        results["temporal"] = temporal
        results["temporal_line"] = build_temporal_line_data(df, timepoint_order, top_n)

    return results


class TestEndToEndGrouping:
    """Full pipeline: grouping metrics computed from raw CSV input."""

    # Grouping-only pipeline produces grouping and heatmap results
    def test_grouping_pipeline_produces_outputs(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,grouping\n"
            "s1,clone1,100,lung\n"
            "s2,clone1,50,spleen\n"
            "s1,clone2,80,lung\n"
            "s2,clone2,20,spleen\n"
        )
        results = _run_pipeline(tmp_path, csv_text, has_grouping=True)
        grouping = results["grouping"]
        assert len(grouping) == 2
        assert "ri" in grouping.columns
        assert "dominant" in grouping.columns
        assert "breadth" in grouping.columns

    # Highly restricted clone → RI close to 1
    def test_restricted_clone_high_ri(self, tmp_path):
        # Need multiple elements per sample so relative frequencies aren't all 1.0
        csv_text = (
            "sampleId,elementId,abundance,grouping\n"
            "s1,clone1,990,lung\n"
            "s1,clone2,10,lung\n"
            "s2,clone1,10,spleen\n"
            "s2,clone2,990,spleen\n"
        )
        results = _run_pipeline(tmp_path, csv_text, has_grouping=True)
        row = results["grouping"].filter(pl.col("elementId") == "clone1").row(0, named=True)
        # clone1: lung freq ~ 0.99, spleen freq ~ 0.01 → highly restricted
        assert row["ri"] > 0.8
        assert row["dominant"] == "lung"


class TestEndToEndTemporal:
    """Full pipeline: temporal metrics from raw CSV input."""

    # Temporal pipeline produces temporal metrics
    def test_temporal_pipeline_produces_outputs(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,timepoint\n"
            "s1,clone1,100,Day0\n"
            "s2,clone1,200,Day7\n"
            "s3,clone1,150,Day14\n"
        )
        results = _run_pipeline(
            tmp_path, csv_text,
            has_timepoint=True,
            timepoint_order=["Day0", "Day7", "Day14"],
        )
        temporal = results["temporal"]
        assert "temporalShiftIndex" in temporal.columns
        assert "log2PeakDelta" in temporal.columns
        assert "log2KineticDelta" in temporal.columns
        assert "peakTimepoint" in temporal.columns

    # Expanding clone: peak at later timepoint, positive fold-changes
    def test_expanding_clone(self, tmp_path):
        # Need multiple elements so relative frequencies reflect abundance ratios
        csv_text = (
            "sampleId,elementId,abundance,timepoint\n"
            "s1,clone1,100,Day0\n"
            "s1,clone2,900,Day0\n"
            "s2,clone1,900,Day7\n"
            "s2,clone2,100,Day7\n"
        )
        results = _run_pipeline(
            tmp_path, csv_text,
            has_timepoint=True,
            timepoint_order=["Day0", "Day7"],
        )
        row = results["temporal"].filter(pl.col("elementId") == "clone1").row(0, named=True)
        # clone1: Day0 freq = 0.1, Day7 freq = 0.9 → peak at Day7
        assert row["peakTimepoint"] == "Day7"
        assert row["log2PeakDelta"] > 0
        assert row["log2KineticDelta"] > 0


class TestEndToEndSubject:
    """Full pipeline with subject dimension: prevalence and cross-subject metrics."""

    # Subject pipeline produces prevalence and histogram
    def test_subject_prevalence_outputs(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,subject\n"
            "s1,clone1,100,sub1\n"
            "s2,clone1,200,sub2\n"
            "s3,clone2,50,sub1\n"
        )
        results = _run_pipeline(tmp_path, csv_text, has_subject=True)
        prevalence = results["prevalence"]
        histogram = results["histogram"]

        c1 = prevalence.filter(pl.col("elementId") == "clone1")
        assert c1["subjectPrevalence"][0] == 2
        assert c1["subjectPrevalenceFraction"][0] == pytest.approx(1.0)

        c2 = prevalence.filter(pl.col("elementId") == "clone2")
        assert c2["subjectPrevalence"][0] == 1

        assert "prevalenceCount" in histogram.columns
        assert "cloneCount" in histogram.columns

    # Full pipeline with all dimensions
    def test_full_pipeline_all_dimensions(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,subject,grouping,timepoint\n"
            "s1,clone1,100,sub1,lung,Day0\n"
            "s2,clone1,200,sub1,spleen,Day7\n"
            "s3,clone1,150,sub2,lung,Day0\n"
            "s4,clone1,300,sub2,spleen,Day7\n"
        )
        results = _run_pipeline(
            tmp_path, csv_text,
            has_subject=True,
            has_grouping=True,
            has_timepoint=True,
            timepoint_order=["Day0", "Day7"],
            min_subject_count=1,
        )
        assert "prevalence" in results
        assert "grouping" in results
        assert "temporal" in results


class TestEndToEndCLR:
    """Full pipeline with CLR normalization."""

    # CLR normalization runs without error and produces valid grouping metrics
    def test_clr_normalization_runs(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,grouping\n"
            "s1,clone1,100,lung\n"
            "s1,clone2,200,lung\n"
            "s2,clone1,150,spleen\n"
            "s2,clone2,50,spleen\n"
        )
        results = _run_pipeline(tmp_path, csv_text, has_grouping=True, normalization="clr")
        grouping = results["grouping"]
        assert len(grouping) == 2
        assert "ri" in grouping.columns


class TestEndToEndMinAbundance:
    """Full pipeline with minimum abundance filter."""

    # R7c: low-abundance clones excluded before any computation
    def test_min_abundance_excludes_low_clones(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,grouping\n"
            "s1,clone1,100,lung\n"
            "s2,clone1,50,spleen\n"
            "s1,clone2,3,lung\n"
            "s2,clone2,2,spleen\n"
        )
        results = _run_pipeline(
            tmp_path, csv_text,
            has_grouping=True,
            min_abundance_threshold=10,
        )
        grouping = results["grouping"]
        assert set(grouping["elementId"].to_list()) == {"clone1"}


class TestEndToEndReplicates:
    """Full pipeline with replicate averaging."""

    # R7a: replicates averaged before grouping metrics
    def test_replicates_averaged(self, tmp_path):
        csv_text = (
            "sampleId,elementId,abundance,grouping\n"
            "rep1,clone1,100,lung\n"
            "rep2,clone1,200,lung\n"
            "rep3,clone1,50,spleen\n"
        )
        results = _run_pipeline(tmp_path, csv_text, has_grouping=True)
        grouping = results["grouping"]
        # rep1 and rep2 share (clone1, lung) → averaged to 150
        # rep3 is unique (clone1, spleen) → 50
        assert len(grouping) == 1
        assert grouping["dominant"][0] == "lung"
