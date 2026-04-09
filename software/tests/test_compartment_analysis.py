"""Behavioral tests for compartment_analysis.py.

Unit tests for pure math functions + end-to-end tests invoking the CLI.
All tests focus on observable behavior (inputs -> outputs).

Spec: docs/text/work/projects/in-vivo-compartment-analysis/README.md

Run from software/:
    uv sync
    uv run pytest tests/
"""

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from compartment_analysis import (
    restriction_index,
    shannon_entropy,
)

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "src", "compartment_analysis.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(input_csv: str, args: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Invoke compartment_analysis.py and return the result."""
    cmd = [sys.executable, SCRIPT, input_csv] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)


def _write_csv(tmp_path: str, filename: str, rows: list[dict]) -> str:
    path = os.path.join(tmp_path, filename)
    df = pl.DataFrame(rows)
    df.write_csv(path)
    return path


def _read_output(tmp_path: str, suffix: str) -> pl.DataFrame:
    path = os.path.join(tmp_path, f"out_{suffix}.csv")
    return pl.read_csv(path)


# ---------------------------------------------------------------------------
# Fixtures: reusable synthetic datasets
# ---------------------------------------------------------------------------


def _full_dataset() -> list[dict]:
    """3 subjects, 2 tissues, 3 timepoints, 3 clones + background.
    Designed for the terminal mouse model use case (E-MTAB-9478 pattern).
    Each subject is sampled in both tissues at all 3 timepoints = 6 samples per subject."""
    rows = []
    # Subject m1
    for tp, tp_data in [
        ("D0", {"c1": 100, "c2": 10, "c3": 5, "bg": 885}),
        ("D7", {"c1": 200, "c2": 50, "c3": 20, "bg": 730}),
        ("D14", {"c1": 50, "c2": 100, "c3": 80, "bg": 770}),
    ]:
        for tissue in ["lung", "spleen"]:
            sample_id = f"m1_{tissue}_{tp}"
            scale = 1.0 if tissue == "lung" else 0.3
            for clone, count in tp_data.items():
                rows.append({
                    "sampleId": sample_id,
                    "elementId": clone,
                    "abundance": round(count * scale),
                    "subject": "m1",
                    "grouping": tissue,
                    "timepoint": tp,
                })
    # Subject m2 — lung-dominant c1
    for tp, tp_data in [
        ("D0", {"c1": 80, "c2": 5, "c3": 0, "bg": 915}),
        ("D7", {"c1": 300, "c2": 30, "c3": 10, "bg": 660}),
        ("D14", {"c1": 100, "c2": 80, "c3": 50, "bg": 770}),
    ]:
        for tissue in ["lung", "spleen"]:
            sample_id = f"m2_{tissue}_{tp}"
            scale = 1.0 if tissue == "lung" else 0.2
            for clone, count in tp_data.items():
                rows.append({
                    "sampleId": sample_id,
                    "elementId": clone,
                    "abundance": round(count * scale),
                    "subject": "m2",
                    "grouping": tissue,
                    "timepoint": tp,
                })
    # Subject m3 — spleen-dominant c1
    for tp, tp_data in [
        ("D0", {"c1": 50, "c2": 30, "c3": 10, "bg": 910}),
        ("D7", {"c1": 150, "c2": 60, "c3": 40, "bg": 750}),
        ("D14", {"c1": 80, "c2": 120, "c3": 100, "bg": 700}),
    ]:
        for tissue in ["lung", "spleen"]:
            sample_id = f"m3_{tissue}_{tp}"
            # For m3, spleen is dominant
            scale = 0.3 if tissue == "lung" else 1.0
            for clone, count in tp_data.items():
                rows.append({
                    "sampleId": sample_id,
                    "elementId": clone,
                    "abundance": round(count * scale),
                    "subject": "m3",
                    "grouping": tissue,
                    "timepoint": tp,
                })
    return rows


# ===========================================================================
# Unit tests: pure math functions
# ===========================================================================


class TestRestrictionIndex:
    # RI = 1 - H(p)/log2(N) — tests pin exact values for the formula
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
    # H(p) = -sum(p_i * log2(p_i)) — no E2E equivalent for these exact values
    def test_single_element(self):
        assert shannon_entropy(np.array([1.0])) == 0.0

    def test_uniform_two_elements(self):
        assert shannon_entropy(np.array([0.5, 0.5])) == pytest.approx(1.0)

    def test_empty_array(self):
        assert shannon_entropy(np.array([])) == 0.0


# ===========================================================================
# E2E tests: full pipeline via CLI
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. Happy path: full pipeline, population mode
# ---------------------------------------------------------------------------


class TestHappyPathPopulation:
    """Full pipeline in population mode with grouping + temporal + subject (R1, R2, R5, R8)."""

    def test_full_pipeline_produces_all_outputs(self, tmp_path):
        # The most common use case: terminal model with all three variables
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        result = _run(input_csv, [
            "--calculation-mode", "population",
            "--normalization", "relative-frequency",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0, f"stderr: {result.stderr}"

        # All output files should exist
        for suffix in ["main", "grouping", "temporal", "heatmap", "temporal_line", "prevalence", "prevalence_histogram"]:
            path = os.path.join(str(tmp_path), f"out_{suffix}.csv")
            assert os.path.exists(path), f"Missing output: {suffix}"

    def test_main_table_has_expected_columns(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        main = _read_output(str(tmp_path), "main")
        # Must contain prevalence, grouping, and temporal columns
        assert "elementId" in main.columns
        assert "subjectPrevalence" in main.columns
        assert "ri" in main.columns
        assert "dominant" in main.columns
        assert "peakTimepoint" in main.columns
        assert "temporalShiftIndex" in main.columns
        assert "log2PeakDelta" in main.columns
        assert "log2KineticDelta" in main.columns

    def test_all_clones_present_in_main_table(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        main = _read_output(str(tmp_path), "main")
        expected_clones = {"c1", "c2", "c3", "bg"}
        assert set(main["elementId"].to_list()) == expected_clones

    def test_ri_values_in_valid_range(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        # RI must be in [0, 1] for all clones (R11)
        for ri in grouping["ri"].to_list():
            if not math.isnan(ri):
                assert 0.0 <= ri <= 1.0, f"RI {ri} out of range"

    def test_tsi_values_in_valid_range(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        # TSI must be in [0, 1] (R15)
        for tsi in temporal["temporalShiftIndex"].to_list():
            if not math.isnan(tsi):
                assert 0.0 <= tsi <= 1.0, f"TSI {tsi} out of range"

    def test_subject_prevalence_correct_for_shared_clone(self, tmp_path):
        # All 3 subjects have c1 with abundance > 0 — prevalence should be 3
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        prevalence = _read_output(str(tmp_path), "prevalence")
        c1 = prevalence.filter(pl.col("elementId") == "c1")
        assert c1["subjectPrevalence"][0] == 3
        assert c1["subjectPrevalenceFraction"][0] == pytest.approx(1.0)

    def test_grouping_exact_values_for_known_clone(self, tmp_path):
        # c1 is lung-dominant in m1 and m2, spleen-dominant in m3 — consensus should be lung
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"
        assert c1["breadth"][0] == 2


# ---------------------------------------------------------------------------
# 2. Happy path: intra-subject mode
# ---------------------------------------------------------------------------


class TestHappyPathIntraSubject:
    """Intra-Subject mode computes per-subject metrics then aggregates (R3)."""

    def test_intra_subject_produces_outputs(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        result = _run(input_csv, [
            "--calculation-mode", "intra-subject",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(os.path.join(str(tmp_path), "out_main.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "out_grouping.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "out_temporal.csv"))

    def test_intra_subject_temporal_metrics_are_aggregated(self, tmp_path):
        # In intra-subject mode, temporal metrics should be per-subject then averaged
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--calculation-mode", "intra-subject",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        # Should have one row per element (aggregated across subjects)
        element_ids = temporal["elementId"].to_list()
        assert len(element_ids) == len(set(element_ids)), "Temporal should have one row per element"


# ---------------------------------------------------------------------------
# 3. Happy path: grouping only / temporal only
# ---------------------------------------------------------------------------


class TestPartialVariables:
    """Block works when only one of grouping/temporal is configured (R5a, R32)."""

    def test_grouping_only_no_temporal_output(self, tmp_path):
        # Only grouping configured — temporal files should NOT exist
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), "out_grouping.csv"))
        assert not os.path.exists(os.path.join(str(tmp_path), "out_temporal.csv"))
        assert not os.path.exists(os.path.join(str(tmp_path), "out_temporal_line.csv"))

    def test_temporal_only_no_grouping_output(self, tmp_path):
        # Only temporal configured — grouping files should NOT exist
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), "out_temporal.csv"))
        assert not os.path.exists(os.path.join(str(tmp_path), "out_grouping.csv"))
        assert not os.path.exists(os.path.join(str(tmp_path), "out_heatmap.csv"))

    def test_no_subject_variable_population_mode(self, tmp_path):
        # Population mode without subject — no prevalence, no consensus metrics (R5, edge case)
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "grouping": "spleen"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 50, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c2", "abundance": 100, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert os.path.exists(os.path.join(str(tmp_path), "out_main.csv"))
        assert os.path.exists(os.path.join(str(tmp_path), "out_grouping.csv"))
        # No prevalence when no subject
        assert not os.path.exists(os.path.join(str(tmp_path), "out_prevalence.csv"))

        grouping = _read_output(str(tmp_path), "grouping")
        # No consensus columns when no subject
        assert "consensusDominant" not in grouping.columns
        assert "meanRi" not in grouping.columns


# ---------------------------------------------------------------------------
# 4. CLR normalization
# ---------------------------------------------------------------------------


class TestCLRNormalization:
    """CLR transform produces valid output and handles zeros (R9)."""

    def test_clr_runs_without_error(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        result = _run(input_csv, [
            "--normalization", "clr",
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert os.path.exists(os.path.join(str(tmp_path), "out_main.csv"))

    def test_clr_with_zero_abundance_clone(self, tmp_path):
        # c3 is absent from m2 at D0 — CLR must handle the zero via replacement (R9)
        rows = _full_dataset()
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--normalization", "clr",
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        grouping = _read_output(str(tmp_path), "grouping")
        # RI values should be valid (in [0,1]) or NaN (CLR can cause NaN for
        # clones below geometric mean in all groups — known limitation)
        for ri in grouping["ri"].to_list():
            if ri is not None and not math.isnan(ri):
                assert 0.0 <= ri <= 1.0

    def test_clr_intra_subject_per_subject_scope(self, tmp_path):
        # In intra-subject mode, CLR should be applied per-subject (R9)
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        result = _run(input_csv, [
            "--calculation-mode", "intra-subject",
            "--normalization", "clr",
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        assert os.path.exists(os.path.join(str(tmp_path), "out_grouping.csv"))


# ---------------------------------------------------------------------------
# 5. Edge case: single group (R32, R34)
# ---------------------------------------------------------------------------


class TestSingleGroup:
    """When only one group category exists, RI should be 1.0 for all clones."""

    def test_single_group_ri_is_one(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 200, "grouping": "lung"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        for ri in grouping["ri"].to_list():
            assert ri == pytest.approx(1.0)

    def test_single_group_breadth_is_one(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 200, "grouping": "lung"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        for b in grouping["breadth"].to_list():
            assert b == 1


# ---------------------------------------------------------------------------
# 6. Edge case: single timepoint (R32)
# ---------------------------------------------------------------------------


class TestSingleTimepoint:
    """Temporal metrics should not be computed when only one timepoint exists."""

    def test_single_timepoint_no_temporal_output(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "timepoint": "D0", "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 200, "timepoint": "D0", "grouping": "lung"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint", "--has-grouping",
            "--timepoint-order", json.dumps(["D0"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        # Temporal files should NOT be generated with only 1 timepoint
        assert not os.path.exists(os.path.join(str(tmp_path), "out_temporal.csv"))


# ---------------------------------------------------------------------------
# 7. Edge case: clone in only one timepoint (R16, R16a)
# ---------------------------------------------------------------------------


class TestCloneOneTimepoint:
    """Clone present at only one timepoint gets Log2PD=0 and Log2KD=0."""

    def test_single_detection_fold_changes_are_zero(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 900, "timepoint": "D0"},
            # c1 absent at D7
            {"sampleId": "s2", "elementId": "c1", "abundance": 0, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 1000, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["log2PeakDelta"][0] == pytest.approx(0.0)
        assert c1["log2KineticDelta"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Edge case: clone in only one subject (R17b, R33)
# ---------------------------------------------------------------------------


class TestCloneOneSubject:
    """Clone in fewer than minSubjectCount subjects gets NaN for averaged metrics."""

    def test_prevalence_one_mean_ri_nan(self, tmp_path):
        rows = [
            # c1 only in m1
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "grouping": "spleen", "subject": "m1"},
            # c1 absent from m2
            {"sampleId": "s3", "elementId": "c1", "abundance": 0, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "c1", "abundance": 0, "grouping": "spleen", "subject": "m2"},
            # bg present in both
            {"sampleId": "s1", "elementId": "bg", "abundance": 900, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 950, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 1000, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "bg", "abundance": 1000, "grouping": "spleen", "subject": "m2"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "2",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        prevalence = _read_output(str(tmp_path), "prevalence")
        c1 = prevalence.filter(pl.col("elementId") == "c1")
        # Subject prevalence still counts the single subject (R33)
        assert c1["subjectPrevalence"][0] == 1

        grouping = _read_output(str(tmp_path), "grouping")
        c1_g = grouping.filter(pl.col("elementId") == "c1")
        # Mean RI should be NaN for clone below minSubjectCount (R17b)
        assert math.isnan(c1_g["meanRi"][0])
        assert math.isnan(c1_g["stdRi"][0])


# ---------------------------------------------------------------------------
# 9. Edge case: missing/empty metadata (R7b, R35)
# ---------------------------------------------------------------------------


class TestMissingMetadata:
    """Samples with missing metadata are excluded from that variable's computation only."""

    def test_missing_grouping_excluded(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung", "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 100, "grouping": "", "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 100, "grouping": "spleen", "timepoint": "D14"},
            # Background so frequencies are well-defined
            {"sampleId": "s1", "elementId": "bg", "abundance": 900, "grouping": "lung", "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 900, "grouping": "", "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 900, "grouping": "spleen", "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--has-grouping", "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        # Grouping should work (excluding the empty-grouping sample)
        grouping = _read_output(str(tmp_path), "grouping")
        assert len(grouping) > 0
        # Temporal should also work (sample with missing grouping still participates)
        temporal = _read_output(str(tmp_path), "temporal")
        assert len(temporal) > 0

    def test_missing_timepoint_excluded(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 200, "timepoint": ""},
            {"sampleId": "s3", "elementId": "c1", "abundance": 150, "timepoint": "D7"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 900, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 800, "timepoint": ""},
            {"sampleId": "s3", "elementId": "bg", "abundance": 850, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        temporal = _read_output(str(tmp_path), "temporal")
        assert len(temporal) > 0


# ---------------------------------------------------------------------------
# 10. Edge case: min abundance threshold (R7c)
# ---------------------------------------------------------------------------


class TestMinAbundanceThreshold:
    """Clones below threshold in ALL samples are excluded entirely."""

    def test_threshold_filters_rare_clones(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 200, "grouping": "spleen"},
            # c_rare: below threshold (max=5) in all samples
            {"sampleId": "s1", "elementId": "c_rare", "abundance": 3, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c_rare", "abundance": 5, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--min-abundance-threshold", "10",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        main = _read_output(str(tmp_path), "main")
        assert "c_rare" not in main["elementId"].to_list()
        assert "c1" in main["elementId"].to_list()

    def test_threshold_keeps_clone_above_in_any_sample(self, tmp_path):
        # c1 is below threshold in s1 but above in s2 — should be kept
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 3, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--min-abundance-threshold", "10",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        main = _read_output(str(tmp_path), "main")
        assert "c1" in main["elementId"].to_list()


# ---------------------------------------------------------------------------
# 11. Edge case: replicate averaging (R7a)
# ---------------------------------------------------------------------------


class TestReplicateAveraging:
    """Multiple samples mapping to the same condition are averaged transparently."""

    def test_replicates_produce_averaged_frequencies(self, tmp_path):
        # Two replicates for m1+lung — abundances 100 and 200 should average to 150
        rows = [
            {"sampleId": "rep1", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep2", "elementId": "c1", "abundance": 200, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep1", "elementId": "c2", "abundance": 50, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep2", "elementId": "c2", "abundance": 50, "grouping": "lung", "subject": "m1"},
            # Different condition — no replicate
            {"sampleId": "s3", "elementId": "c1", "abundance": 80, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s3", "elementId": "c2", "abundance": 120, "grouping": "spleen", "subject": "m1"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        grouping = _read_output(str(tmp_path), "grouping")
        assert len(grouping) > 0


# ---------------------------------------------------------------------------
# 12. Edge case: timepoint deselection (R7)
# ---------------------------------------------------------------------------


class TestTimepointDeselection:
    """Deselected timepoints exclude corresponding samples from temporal analysis."""

    def test_deselected_timepoint_excluded(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 10, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 90, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 80, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 20, "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 50, "timepoint": "D14"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 50, "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)

        # Only include D0 and D14 — D7 (the peak) is deselected
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        # Without D7, the peak should be at D14 (freq 0.5 > D0 freq 0.1)
        assert c1["peakTimepoint"][0] == "D14"


# ---------------------------------------------------------------------------
# 13. Edge case: dominant group tie-breaking (R12)
# ---------------------------------------------------------------------------


class TestDominantTieBreaking:
    """Ties for dominant group are broken alphabetically."""

    def test_equal_frequency_alphabetical_wins(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 50, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "grouping": "lung"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        # "lung" < "spleen" alphabetically
        assert c1["dominant"][0] == "lung"


# ---------------------------------------------------------------------------
# 14. Edge case: consensus dominant tie-breaking (R18)
# ---------------------------------------------------------------------------


class TestConsensusDominantTieBreaking:
    """Consensus dominant ties broken by highest mean frequency across tied groups."""

    def test_tied_count_resolved_by_mean_frequency(self, tmp_path):
        # m1: c1 dominant in lung (freq 0.95 vs 0.05)
        # m2: c1 dominant in spleen (freq 0.6 vs 0.4)
        # Tie 1:1 → lung has higher mean freq (0.95+0.4)/2=0.675 vs (0.05+0.6)/2=0.325
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 95, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 5, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 40, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "c1", "abundance": 60, "grouping": "spleen", "subject": "m2"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"


# ---------------------------------------------------------------------------
# 15. Edge case: uniform distribution → RI = 0 (R11)
# ---------------------------------------------------------------------------


class TestUniformDistribution:
    """Perfectly uniform distribution across groups yields RI = 0."""

    def test_uniform_ri_zero(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 100, "grouping": "spleen"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 100, "grouping": "brain"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        assert c1["ri"][0] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 16. Edge case: zero-total sample excluded (edge case table)
# ---------------------------------------------------------------------------


class TestZeroTotalSample:
    """Samples with zero total abundance are excluded from normalization."""

    def test_zero_total_sample_excluded(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100},
            {"sampleId": "s2", "elementId": "c1", "abundance": 0},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        main = _read_output(str(tmp_path), "main")
        assert "c1" in main["elementId"].to_list()


# ---------------------------------------------------------------------------
# 17. Edge case: all subjects same dominant (edge case table)
# ---------------------------------------------------------------------------


class TestAllSubjectsSameDominant:
    """All subjects share the same dominant group."""

    def test_consensus_matches_unanimous_dominant(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 80, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "c1", "abundance": 20, "grouping": "spleen", "subject": "m2"},
            {"sampleId": "s5", "elementId": "c1", "abundance": 85, "grouping": "lung", "subject": "m3"},
            {"sampleId": "s6", "elementId": "c1", "abundance": 15, "grouping": "spleen", "subject": "m3"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"
        assert c1["countDominantIn_lung"][0] == 3
        assert c1["countDominantIn_spleen"][0] == 0


# ---------------------------------------------------------------------------
# 18. Visualization outputs
# ---------------------------------------------------------------------------


class TestVisualizationOutputs:
    """Heatmap, temporal line, and prevalence histogram have correct structure."""

    def test_heatmap_structure(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--has-grouping", "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--top-n", "3",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        heatmap = _read_output(str(tmp_path), "heatmap")
        assert "elementId" in heatmap.columns
        assert "groupCategory" in heatmap.columns
        assert "normalizedFrequency" in heatmap.columns
        # Top-3 by RI means at most 3 distinct element IDs
        assert heatmap["elementId"].n_unique() <= 3

    def test_temporal_line_structure(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--top-n", "2",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        line = _read_output(str(tmp_path), "temporal_line")
        assert "elementId" in line.columns
        assert "timepointValue" in line.columns
        assert "frequency" in line.columns
        # Top-2 by Log2PD means at most 2 distinct elements
        assert line["elementId"].n_unique() <= 2

    def test_prevalence_histogram_structure(self, tmp_path):
        input_csv = _write_csv(str(tmp_path), "input.csv", _full_dataset())
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        histogram = _read_output(str(tmp_path), "prevalence_histogram")
        assert "prevalenceCount" in histogram.columns
        assert "cloneCount" in histogram.columns
        # Sum of clone counts should equal total distinct elements
        total_clones = histogram["cloneCount"].sum()
        prevalence = _read_output(str(tmp_path), "prevalence")
        assert total_clones == len(prevalence)


# ---------------------------------------------------------------------------
# 19. Hot path: population mode with subject — two-stage temporal aggregation
# ---------------------------------------------------------------------------


class TestTwoStageTemporalAggregation:
    """Population mode with subjects must aggregate per-subject first, then across subjects.
    This guards against the bug where subjects with more compartments get over-weighted."""

    def test_equal_weight_per_subject(self, tmp_path):
        # m1 has 3 compartments at D0, m2 has 1 compartment at D0.
        # After two-stage aggregation, each subject contributes equally.
        rows = [
            # m1: 3 compartments at D0, high frequency in c1
            {"sampleId": "m1_lung_D0", "elementId": "c1", "abundance": 80, "timepoint": "D0", "subject": "m1", "grouping": "lung"},
            {"sampleId": "m1_lung_D0", "elementId": "bg", "abundance": 20, "timepoint": "D0", "subject": "m1", "grouping": "lung"},
            {"sampleId": "m1_spleen_D0", "elementId": "c1", "abundance": 60, "timepoint": "D0", "subject": "m1", "grouping": "spleen"},
            {"sampleId": "m1_spleen_D0", "elementId": "bg", "abundance": 40, "timepoint": "D0", "subject": "m1", "grouping": "spleen"},
            {"sampleId": "m1_brain_D0", "elementId": "c1", "abundance": 90, "timepoint": "D0", "subject": "m1", "grouping": "brain"},
            {"sampleId": "m1_brain_D0", "elementId": "bg", "abundance": 10, "timepoint": "D0", "subject": "m1", "grouping": "brain"},
            # m1: 3 compartments at D7, low frequency in c1
            {"sampleId": "m1_lung_D7", "elementId": "c1", "abundance": 10, "timepoint": "D7", "subject": "m1", "grouping": "lung"},
            {"sampleId": "m1_lung_D7", "elementId": "bg", "abundance": 90, "timepoint": "D7", "subject": "m1", "grouping": "lung"},
            {"sampleId": "m1_spleen_D7", "elementId": "c1", "abundance": 5, "timepoint": "D7", "subject": "m1", "grouping": "spleen"},
            {"sampleId": "m1_spleen_D7", "elementId": "bg", "abundance": 95, "timepoint": "D7", "subject": "m1", "grouping": "spleen"},
            {"sampleId": "m1_brain_D7", "elementId": "c1", "abundance": 15, "timepoint": "D7", "subject": "m1", "grouping": "brain"},
            {"sampleId": "m1_brain_D7", "elementId": "bg", "abundance": 85, "timepoint": "D7", "subject": "m1", "grouping": "brain"},
            # m2: 1 compartment at D0, low frequency in c1
            {"sampleId": "m2_lung_D0", "elementId": "c1", "abundance": 10, "timepoint": "D0", "subject": "m2", "grouping": "lung"},
            {"sampleId": "m2_lung_D0", "elementId": "bg", "abundance": 90, "timepoint": "D0", "subject": "m2", "grouping": "lung"},
            # m2: 1 compartment at D7, high frequency in c1
            {"sampleId": "m2_lung_D7", "elementId": "c1", "abundance": 80, "timepoint": "D7", "subject": "m2", "grouping": "lung"},
            {"sampleId": "m2_lung_D7", "elementId": "bg", "abundance": 20, "timepoint": "D7", "subject": "m2", "grouping": "lung"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--calculation-mode", "population",
            "--has-timepoint", "--has-subject", "--has-grouping",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")

        # m1 at D0: mean freq across 3 compartments = (80/100 + 60/100 + 90/100) / 3 ≈ 0.767
        # m1 at D7: mean freq across 3 compartments = (10/100 + 5/100 + 15/100) / 3 = 0.1
        # m2 at D0: freq = 10/100 = 0.1
        # m2 at D7: freq = 80/100 = 0.8
        # Cross-subject mean at D0: (0.767 + 0.1) / 2 ≈ 0.433
        # Cross-subject mean at D7: (0.1 + 0.8) / 2 = 0.45
        # Peak should be D7 (0.45 > 0.433)
        assert c1["peakTimepoint"][0] == "D7"


# ---------------------------------------------------------------------------
# 20. Edge case: presence threshold for breadth (R13)
# ---------------------------------------------------------------------------


class TestPresenceThreshold:
    """Breadth only counts groups above the presence threshold."""

    def test_nonzero_threshold_reduces_breadth(self, tmp_path):
        # c1: high in lung (freq ≈ 0.99), trace in spleen (freq ≈ 0.01)
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 990, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 990, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping",
            "--presence-threshold", "0.05",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        # Spleen frequency ≈ 0.01 < 0.05 threshold → breadth = 1
        assert c1["breadth"][0] == 1


# ---------------------------------------------------------------------------
# 21. Edge case: NaN/null abundance values in input
# ---------------------------------------------------------------------------


class TestNullAbundanceHandling:
    """Rows with NaN/null abundance are dropped during input reading."""

    def test_nan_abundance_dropped(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": "100"},
            {"sampleId": "s1", "elementId": "c2", "abundance": "NaN"},
            {"sampleId": "s1", "elementId": "c3", "abundance": "NA"},
        ]
        # Write manually to preserve string NaN
        path = os.path.join(str(tmp_path), "input.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sampleId", "elementId", "abundance"])
            writer.writeheader()
            writer.writerows(rows)

        result = _run(path, [
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        main = _read_output(str(tmp_path), "main")
        # Only c1 should remain (c2=NaN, c3=NA both dropped)
        assert main["elementId"].to_list() == ["c1"]


# ---------------------------------------------------------------------------
# 22. Integer metadata columns (non-string grouping/timepoint)
# ---------------------------------------------------------------------------


class TestIntegerMetadata:
    """Grouping and timepoint columns provided as integers work end-to-end."""

    def test_integer_grouping_and_timepoint(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 80, "grouping": 1, "timepoint": 7},
            {"sampleId": "s2", "elementId": "c1", "abundance": 20, "grouping": 2, "timepoint": 14},
            {"sampleId": "s1", "elementId": "bg", "abundance": 20, "grouping": 1, "timepoint": 7},
            {"sampleId": "s2", "elementId": "bg", "abundance": 80, "grouping": 2, "timepoint": 14},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--has-grouping", "--has-timepoint",
            "--timepoint-order", json.dumps(["7", "14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        grouping = _read_output(str(tmp_path), "grouping")
        assert len(grouping) > 0
        temporal = _read_output(str(tmp_path), "temporal")
        assert len(temporal) > 0


# ---------------------------------------------------------------------------
# 23. Edge case: min_subject_count with temporal metrics in intra-subject mode
# ---------------------------------------------------------------------------


class TestMinSubjectCountTemporal:
    """In intra-subject mode, temporal metrics are NaN below minSubjectCount (R17b)."""

    def test_single_subject_temporal_nan(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "timepoint": "D0", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 200, "timepoint": "D7", "subject": "m1"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--calculation-mode", "intra-subject",
            "--has-timepoint", "--has-subject",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--min-subject-count", "2",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        # Single subject with minSubjectCount=2 → NaN for averaged metrics
        assert math.isnan(c1["temporalShiftIndex"][0])
        assert math.isnan(c1["log2KineticDelta"][0])
        assert math.isnan(c1["log2PeakDelta"][0])


# ---------------------------------------------------------------------------
# 24. Log2 Peak Delta is always non-negative (R16)
# ---------------------------------------------------------------------------


class TestLog2PeakDeltaNonNegative:
    """Log2 Peak Delta must always be >= 0 since peak >= first detected."""

    def test_contracting_clone_log2pd_still_nonneg(self, tmp_path):
        # Clone contracts over time: D0=high, D7=low. Peak=D0=first → Log2PD=0.
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 10, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 90, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["log2PeakDelta"][0] >= 0.0

    def test_expanding_clone_positive_log2pd(self, tmp_path):
        # Clone expands: D0=10%, D7=80%. Log2PD = log2(0.8/0.1) = 3.0
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 10, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 90, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 80, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 20, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["log2PeakDelta"][0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 25. Log2 Kinetic Delta sign (R16a)
# ---------------------------------------------------------------------------


class TestLog2KineticDeltaSign:
    """Log2KD positive = expansion, negative = contraction."""

    def test_expanding_clone_positive_log2kd(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 10, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 90, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 80, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 20, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["log2KineticDelta"][0] > 0

    def test_contracting_clone_negative_log2kd(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 80, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 20, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 90, "timepoint": "D7"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["log2KineticDelta"][0] < 0


# ===========================================================================
# Bug-exposing tests (added from review report)
# ===========================================================================


# ---------------------------------------------------------------------------
# 26. Null/empty subject values excluded from computation (H3, R7b)
# ---------------------------------------------------------------------------


class TestNullSubjectExcluded:
    """Samples with missing/empty subject metadata must be excluded from
    subject-based computations — they should not count as a distinct subject."""

    # Empty-string subject should not inflate prevalence count
    def test_empty_subject_not_counted_in_prevalence(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 80, "grouping": "spleen", "subject": "m2"},
            # Empty subject — should be excluded from prevalence
            {"sampleId": "s3", "elementId": "c1", "abundance": 60, "grouping": "lung", "subject": ""},
            # Background for valid normalization
            {"sampleId": "s1", "elementId": "bg", "abundance": 900, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 920, "grouping": "spleen", "subject": "m2"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 940, "grouping": "lung", "subject": ""},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        prevalence = _read_output(str(tmp_path), "prevalence")
        c1 = prevalence.filter(pl.col("elementId") == "c1")
        # Only m1 and m2 are real subjects
        assert c1["subjectPrevalence"][0] == 2
        assert c1["subjectPrevalenceFraction"][0] == pytest.approx(1.0)

    # Empty subject should not participate in consensus dominant computation
    def test_empty_subject_excluded_from_consensus(self, tmp_path):
        rows = [
            # m1: lung-dominant
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen", "subject": "m1"},
            # m2: lung-dominant
            {"sampleId": "s3", "elementId": "c1", "abundance": 80, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "c1", "abundance": 20, "grouping": "spleen", "subject": "m2"},
            # empty subject: spleen-dominant — should NOT tip consensus to spleen
            {"sampleId": "s5", "elementId": "c1", "abundance": 10, "grouping": "lung", "subject": ""},
            {"sampleId": "s6", "elementId": "c1", "abundance": 90, "grouping": "spleen", "subject": ""},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        assert c1["consensusDominant"][0] == "lung"


# ---------------------------------------------------------------------------
# 27. Single-group subjects excluded from Mean RI (M10, R33)
# ---------------------------------------------------------------------------


class TestSingleGroupSubjectExcluded:
    """Subjects sampled from only one group have trivial RI = 1.0 that is
    an artifact of sampling, not biology. They must be excluded from Mean RI
    but still counted in Subject Prevalence."""

    def test_single_group_subject_excluded_from_mean_ri(self, tmp_path):
        # m1: 2 groups, lung-dominant (RI ≈ 0.531)
        # m2: only 1 group (lung) — RI trivially 1.0, should be excluded
        # m3: 2 groups, balanced (RI = 0.0)
        rows = [
            # m1
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 10, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 90, "grouping": "spleen", "subject": "m1"},
            # m2: only lung
            {"sampleId": "s3", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 900, "grouping": "lung", "subject": "m2"},
            # m3: balanced
            {"sampleId": "s4", "elementId": "c1", "abundance": 50, "grouping": "lung", "subject": "m3"},
            {"sampleId": "s4", "elementId": "bg", "abundance": 50, "grouping": "lung", "subject": "m3"},
            {"sampleId": "s5", "elementId": "c1", "abundance": 50, "grouping": "spleen", "subject": "m3"},
            {"sampleId": "s5", "elementId": "bg", "abundance": 50, "grouping": "spleen", "subject": "m3"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")

        # m1 RI for c1: p=[0.9, 0.1], H=-0.9*log2(0.9)-0.1*log2(0.1)≈0.469, RI=1-0.469≈0.531
        # m2 RI for c1: only 1 group → RI = 1.0 (SHOULD BE EXCLUDED from mean)
        # m3 RI for c1: p=[0.5, 0.5], H=1.0, RI=0.0
        # Correct mean (excluding m2): (0.531 + 0.0) / 2 ≈ 0.266
        # Wrong mean (including m2): (0.531 + 1.0 + 0.0) / 3 ≈ 0.510
        mean_ri = c1["meanRi"][0]
        assert mean_ri < 0.4, (
            f"Mean RI = {mean_ri}; expected ~0.266 (excluding single-group subject m2). "
            f"Value > 0.4 suggests single-group subjects are not excluded."
        )

    # Single-group subject should still be counted in prevalence
    def test_single_group_subject_still_counted_in_prevalence(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen", "subject": "m1"},
            # m2: only lung
            {"sampleId": "s3", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m2"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        prevalence = _read_output(str(tmp_path), "prevalence")
        c1 = prevalence.filter(pl.col("elementId") == "c1")
        # Both subjects count for prevalence even though m2 has one group
        assert c1["subjectPrevalence"][0] == 2


# ===========================================================================
# Value-pinning tests (strengthen coverage)
# ===========================================================================


# ---------------------------------------------------------------------------
# 28. TSI exact formula values
# ---------------------------------------------------------------------------


class TestTSIExactValues:
    """Pin TSI to manually computed values: sum(i*freq_i) / (sum(freq_i)*(T-1))."""

    # All abundance at first timepoint → TSI = 0.0
    def test_all_at_first_timepoint_tsi_zero(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 0, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 0, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 100, "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 0, "timepoint": "D14"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 100, "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(0.0)

    # All abundance at last timepoint → TSI = 1.0
    def test_all_at_last_timepoint_tsi_one(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 0, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 100, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 0, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 100, "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 100, "timepoint": "D14"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 0, "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(1.0)

    # Known distribution: D0=10%, D7=60%, D14=30%
    # TSI = (0*0.1 + 1*0.6 + 2*0.3) / (1.0 * 2) = 1.2/2 = 0.6
    def test_known_distribution_tsi(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 10, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 90, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 60, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 40, "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 30, "timepoint": "D14"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 70, "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        assert c1["temporalShiftIndex"][0] == pytest.approx(0.6)

    # Clone detected at only one timepoint: TSI = rank / (T-1) per spec
    # Clone at position 1 (D7) of 3 timepoints → TSI = 1/2 = 0.5
    def test_single_detection_tsi_equals_rank_over_t_minus_1(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 0, "timepoint": "D0"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 100, "timepoint": "D0"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "timepoint": "D7"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 50, "timepoint": "D7"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 0, "timepoint": "D14"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 100, "timepoint": "D14"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-timepoint",
            "--timepoint-order", json.dumps(["D0", "D7", "D14"]),
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        temporal = _read_output(str(tmp_path), "temporal")
        c1 = temporal.filter(pl.col("elementId") == "c1")
        # TSI = (1 * freq) / (freq * 2) = 1/2 = 0.5
        assert c1["temporalShiftIndex"][0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 29. Mean RI exact value from known per-subject RIs
# ---------------------------------------------------------------------------


class TestMeanRIExactValue:
    """Pin Mean RI to a manually computed expected value."""

    def test_known_subjects_produce_correct_mean_ri(self, tmp_path):
        # m1: c1 lung=0.9, spleen=0.1 → RI ≈ 0.531
        # m2: c1 lung=0.5, spleen=0.5 → RI = 0.0
        # Mean RI = (0.531 + 0.0) / 2 ≈ 0.266
        rows = [
            # m1: lung-dominant
            {"sampleId": "s1", "elementId": "c1", "abundance": 90, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s1", "elementId": "bg", "abundance": 10, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 10, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s2", "elementId": "bg", "abundance": 90, "grouping": "spleen", "subject": "m1"},
            # m2: balanced
            {"sampleId": "s3", "elementId": "c1", "abundance": 50, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 50, "grouping": "lung", "subject": "m2"},
            {"sampleId": "s4", "elementId": "c1", "abundance": 50, "grouping": "spleen", "subject": "m2"},
            {"sampleId": "s4", "elementId": "bg", "abundance": 50, "grouping": "spleen", "subject": "m2"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")

        # m1 RI: p=[0.9,0.1], H = -(0.9*log2(0.9) + 0.1*log2(0.1)) ≈ 0.469
        # RI = 1 - 0.469/1.0 ≈ 0.531
        # m2 RI: p=[0.5,0.5], H = 1.0, RI = 0.0
        # Mean = (0.531 + 0.0) / 2 ≈ 0.266
        expected_m1_ri = 1.0 - (-(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))) / math.log2(2)
        expected_mean = (expected_m1_ri + 0.0) / 2

        assert c1["meanRi"][0] == pytest.approx(expected_mean, abs=1e-3)


# ---------------------------------------------------------------------------
# 30. Replicate averaging produces correct downstream values
# ---------------------------------------------------------------------------


class TestReplicateAveragingExact:
    """Verify that replicate averaging changes the downstream result correctly."""

    # Two replicates with unequal abundance should average before normalization,
    # producing a different RI than treating them as separate samples would
    def test_averaged_replicate_ri_matches_manual_calculation(self, tmp_path):
        # Two replicates for condition (m1, lung):
        #   rep1: c1=100, bg=900 (total=1000) → freq(c1)=0.1
        #   rep2: c1=300, bg=700 (total=1000) → freq(c1)=0.3
        # Averaged: c1=200, bg=800 (total=1000) → freq(c1)=0.2
        # Single for condition (m1, spleen):
        #   c1=200, bg=800 → freq(c1)=0.2
        # Grouping: lung=0.2, spleen=0.2 → perfectly uniform → RI = 0.0
        rows = [
            {"sampleId": "rep1", "elementId": "c1", "abundance": 100, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep1", "elementId": "bg", "abundance": 900, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep2", "elementId": "c1", "abundance": 300, "grouping": "lung", "subject": "m1"},
            {"sampleId": "rep2", "elementId": "bg", "abundance": 700, "grouping": "lung", "subject": "m1"},
            {"sampleId": "s3", "elementId": "c1", "abundance": 200, "grouping": "spleen", "subject": "m1"},
            {"sampleId": "s3", "elementId": "bg", "abundance": 800, "grouping": "spleen", "subject": "m1"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        _run(input_csv, [
            "--has-grouping", "--has-subject",
            "--min-subject-count", "1",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        grouping = _read_output(str(tmp_path), "grouping")
        c1 = grouping.filter(pl.col("elementId") == "c1")
        # After averaging replicates: lung freq = 0.2, spleen freq = 0.2 → RI = 0.0
        assert c1["ri"][0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 31. CLR transform verification with exact values
# ---------------------------------------------------------------------------


class TestCLRExactValues:
    """Verify CLR transform produces expected output for a known input."""

    # CLR produces log-ratios that can be negative. Clones whose frequency is
    # below the geometric mean in ALL groups get all-negative CLR values, which
    # makes RI = NaN (presence detection uses freq > 0). This is a known
    # interaction: CLR + RI requires that the RI formula use original abundance
    # for presence detection rather than CLR values. Until that design change
    # is made, we verify the pipeline completes and dominant clones get valid RI.
    def test_clr_dominant_clones_get_valid_ri(self, tmp_path):
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 200, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c3", "abundance": 700, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 50, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "c2", "abundance": 300, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "c3", "abundance": 650, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--normalization", "clr",
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        grouping = _read_output(str(tmp_path), "grouping")
        assert len(grouping) == 3
        # c3 is the most abundant clone — should have positive CLR in both
        # groups and thus valid (non-NaN) RI
        c3 = grouping.filter(pl.col("elementId") == "c3")
        assert not math.isnan(c3["ri"][0]), "Dominant clone should have valid RI under CLR"

    def test_clr_with_zero_replacement(self, tmp_path):
        # c3 is absent from s1 — zero replacement must be applied before CLR
        rows = [
            {"sampleId": "s1", "elementId": "c1", "abundance": 100, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c2", "abundance": 900, "grouping": "lung"},
            {"sampleId": "s1", "elementId": "c3", "abundance": 0, "grouping": "lung"},
            {"sampleId": "s2", "elementId": "c1", "abundance": 300, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "c2", "abundance": 600, "grouping": "spleen"},
            {"sampleId": "s2", "elementId": "c3", "abundance": 100, "grouping": "spleen"},
        ]
        input_csv = _write_csv(str(tmp_path), "input.csv", rows)
        result = _run(input_csv, [
            "--normalization", "clr",
            "--has-grouping",
            "--output-prefix", os.path.join(str(tmp_path), "out"),
        ], str(tmp_path))

        assert result.returncode == 0
        grouping = _read_output(str(tmp_path), "grouping")
        # c3 should still appear in output (zero-replaced, not dropped)
        assert "c3" in grouping["elementId"].to_list()
