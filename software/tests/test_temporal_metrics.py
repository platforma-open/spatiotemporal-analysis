"""Behavioral tests for temporal metrics (R14-R16a) and temporal modes (R2, R3).

Run: uv run pytest tests/test_temporal_metrics.py -v
"""

import math

import polars as pl
import pytest

from compartment_analysis import _compute_temporal_for_element, compute_temporal_metrics


class TestComputeTemporalForElement:
    """Tests for _compute_temporal_for_element — the core per-element temporal formula."""

    TIMEPOINTS = ["Day0", "Day7", "Day14", "Day28"]

    # R14: peak timepoint is the timepoint with maximum frequency
    def test_peak_is_max_frequency(self):
        tp_freq = {"Day0": 0.1, "Day7": 0.5, "Day14": 0.3, "Day28": 0.1}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["peakTimepoint"] == "Day7"

    # R14: tied max frequency → earliest timepoint wins (argmax first-occurrence)
    def test_peak_tie_broken_by_earliest(self):
        tp_freq = {"Day0": 0.5, "Day7": 0.5, "Day14": 0.0, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["peakTimepoint"] == "Day0"

    # R15: all abundance at first timepoint → TSI = 0.0
    def test_tsi_all_at_first_equals_zero(self):
        tp_freq = {"Day0": 1.0, "Day7": 0.0, "Day14": 0.0, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["temporalShiftIndex"] == pytest.approx(0.0)

    # R15: all abundance at last timepoint → TSI = 1.0
    def test_tsi_all_at_last_equals_one(self):
        tp_freq = {"Day0": 0.0, "Day7": 0.0, "Day14": 0.0, "Day28": 1.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["temporalShiftIndex"] == pytest.approx(1.0)

    # R15: equal abundance everywhere → TSI = 0.5 for even spacing
    def test_tsi_uniform_distribution(self):
        tp_freq = {"Day0": 0.25, "Day7": 0.25, "Day14": 0.25, "Day28": 0.25}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # TSI = sum(i * 0.25) / (1.0 * 3) = (0 + 0.25 + 0.5 + 0.75) / 3 = 0.5
        assert result["temporalShiftIndex"] == pytest.approx(0.5)

    # R15: hand-calculated TSI with 4 timepoints
    def test_tsi_hand_calculated(self):
        tp_freq = {"Day0": 0.1, "Day7": 0.3, "Day14": 0.4, "Day28": 0.2}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # TSI = (0*0.1 + 1*0.3 + 2*0.4 + 3*0.2) / (1.0 * 3) = (0 + 0.3 + 0.8 + 0.6) / 3 = 1.7/3
        expected = (0 * 0.1 + 1 * 0.3 + 2 * 0.4 + 3 * 0.2) / (1.0 * 3)
        assert result["temporalShiftIndex"] == pytest.approx(expected)

    # R15 edge: clone at only one timepoint → TSI = position / (T-1)
    def test_tsi_single_timepoint_detected(self):
        tp_freq = {"Day0": 0.0, "Day7": 0.0, "Day14": 0.5, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # Only detected at position 2, TSI = (2 * 0.5) / (0.5 * 3) = 2/3
        assert result["temporalShiftIndex"] == pytest.approx(2.0 / 3.0)

    # R16: log2(peak/first) when peak > first
    def test_log2pd_basic_expansion(self):
        tp_freq = {"Day0": 0.1, "Day7": 0.4, "Day14": 0.3, "Day28": 0.2}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # Peak = Day7 (0.4), first detected = Day0 (0.1)
        expected = math.log2(0.4 / 0.1)
        assert result["log2PeakDelta"] == pytest.approx(expected)

    # R16: peak == first detected → Log2PD = 0.0
    def test_log2pd_first_is_peak(self):
        tp_freq = {"Day0": 0.5, "Day7": 0.3, "Day14": 0.1, "Day28": 0.1}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # Peak is at Day0 which is also first → log2(0.5/0.5) = 0
        assert result["log2PeakDelta"] == pytest.approx(0.0)

    # R16 edge: single timepoint → Log2PD = 0.0
    def test_log2pd_single_timepoint(self):
        tp_freq = {"Day0": 0.0, "Day7": 0.3, "Day14": 0.0, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        # Only one detected timepoint: peak == first → log2(0.3/0.3) = 0
        assert result["log2PeakDelta"] == pytest.approx(0.0)

    # R16: Log2PD always >= 0 (peak >= first by definition) — parametrized so
    # each scenario appears as its own test ID for clearer failure localization
    @pytest.mark.parametrize("tp_freq", [
        {"Day0": 0.1, "Day7": 0.5, "Day14": 0.2, "Day28": 0.2},  # peak mid-series
        {"Day0": 0.5, "Day7": 0.3, "Day14": 0.1, "Day28": 0.1},  # peak at first
        {"Day0": 0.0, "Day7": 0.0, "Day14": 0.0, "Day28": 0.5},  # peak at last
    ])
    def test_log2pd_always_non_negative(self, tp_freq):
        result = _compute_temporal_for_element("x", tp_freq, self.TIMEPOINTS, 4)
        assert result["log2PeakDelta"] >= 0.0

    # R16a: log2(last/first) when expanding
    def test_log2kd_expansion(self):
        tp_freq = {"Day0": 0.1, "Day7": 0.2, "Day14": 0.3, "Day28": 0.4}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        expected = math.log2(0.4 / 0.1)
        assert result["log2KineticDelta"] == pytest.approx(expected)

    # R16a: log2(last/first) when contracting → negative
    def test_log2kd_contraction(self):
        tp_freq = {"Day0": 0.4, "Day7": 0.3, "Day14": 0.2, "Day28": 0.1}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        expected = math.log2(0.1 / 0.4)
        assert result["log2KineticDelta"] == pytest.approx(expected)
        assert result["log2KineticDelta"] < 0

    # R16a edge: single timepoint → Log2KD = 0.0
    def test_log2kd_single_timepoint(self):
        tp_freq = {"Day0": 0.0, "Day7": 0.5, "Day14": 0.0, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["log2KineticDelta"] == pytest.approx(0.0)

    # Edge: clone not detected at any timepoint
    def test_log2kd_no_detection(self):
        tp_freq = {"Day0": 0.0, "Day7": 0.0, "Day14": 0.0, "Day28": 0.0}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["log2KineticDelta"] == pytest.approx(0.0)
        assert result["log2PeakDelta"] == pytest.approx(0.0)

    # Clone detected at non-adjacent timepoints only
    def test_sparse_detection(self):
        tp_freq = {"Day0": 0.2, "Day7": 0.0, "Day14": 0.0, "Day28": 0.6}
        result = _compute_temporal_for_element("clone1", tp_freq, self.TIMEPOINTS, 4)
        assert result["peakTimepoint"] == "Day28"
        assert result["log2PeakDelta"] == pytest.approx(math.log2(0.6 / 0.2))
        assert result["log2KineticDelta"] == pytest.approx(math.log2(0.6 / 0.2))


class TestComputeTemporalMetrics:
    """Tests for compute_temporal_metrics — population vs intra-subject modes."""

    # R7: deselected timepoints (not in timepoint_order) excluded from computation
    def test_deselected_timepoints_excluded(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "timepoint": "Day0"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "timepoint": "Day7"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.8, "timepoint": "Day14"},
        ])
        # Only Day0 and Day14 selected (Day7 deselected)
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day14"], has_subject=False, mode="population", min_subject_count=1
        )
        row = result.row(0, named=True)
        # Day7 data must not participate in computation
        # Peak is Day14 (0.8 > 0.2), Log2KD = log2(0.8/0.2)
        assert row["peakTimepoint"] == "Day14"
        assert row["log2KineticDelta"] == pytest.approx(math.log2(0.8 / 0.2))
        assert row["log2PeakDelta"] == pytest.approx(math.log2(0.8 / 0.2))

    # R14: tied max frequency across timepoints → earliest timepoint wins
    # Guards the reverse-iteration when/then chain in the vectorized path
    def test_peak_tie_broken_by_earliest_timepoint(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.5, "timepoint": "Day0"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.5, "timepoint": "Day7"},
        ])
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=False, mode="population", min_subject_count=1
        )
        assert result["peakTimepoint"][0] == "Day0"

    # R15 edge: T=1 → returns empty DataFrame (metrics not computed)
    def test_single_timepoint_returns_empty(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.5, "timepoint": "Day0"},
        ])
        result, _ = compute_temporal_metrics(df, ["Day0"], has_subject=False, mode="population", min_subject_count=2)
        assert result.is_empty()

    # Population mode: averages frequency across subjects per timepoint
    def test_population_mode_averages_across_subjects(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "timepoint": "Day0", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.4, "timepoint": "Day0", "subject": "sub2"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.6, "timepoint": "Day7", "subject": "sub1"},
            {"sampleId": "s4", "elementId": "a", "frequency": 0.8, "timepoint": "Day7", "subject": "sub2"},
        ])
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=True, mode="population", min_subject_count=1
        )
        assert len(result) == 1
        # Population: mean freq at Day0 = (0.2+0.4)/2=0.3, Day7 = (0.6+0.8)/2=0.7
        # Peak = Day7, Log2PD = log2(0.7/0.3)
        row = result.row(0, named=True)
        assert row["peakTimepoint"] == "Day7"
        assert row["log2PeakDelta"] == pytest.approx(math.log2(0.7 / 0.3))

    # Intra-subject mode: per-subject metrics then averaged
    def test_intra_subject_computes_per_subject_then_averages(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "timepoint": "Day0", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.9, "timepoint": "Day7", "subject": "sub1"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.3, "timepoint": "Day0", "subject": "sub2"},
            {"sampleId": "s4", "elementId": "a", "frequency": 0.7, "timepoint": "Day7", "subject": "sub2"},
        ])
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=True, mode="intra-subject", min_subject_count=1
        )
        assert len(result) == 1
        row = result.row(0, named=True)
        # Sub1: Log2PD = log2(0.9/0.1), Sub2: Log2PD = log2(0.7/0.3)
        # Average
        sub1_l2pd = math.log2(0.9 / 0.1)
        sub2_l2pd = math.log2(0.7 / 0.3)
        expected_avg = (sub1_l2pd + sub2_l2pd) / 2
        assert row["log2PeakDelta"] == pytest.approx(expected_avg, abs=1e-6)

    # R17b: intra-subject temporal metrics NaN below minSubjectCount
    def test_intra_subject_min_subject_count(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "timepoint": "Day0", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.8, "timepoint": "Day7", "subject": "sub1"},
        ])
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=True, mode="intra-subject", min_subject_count=2
        )
        row = result.row(0, named=True)
        # Only 1 subject, min_subject_count=2 → NaN
        assert math.isnan(row["temporalShiftIndex"])
        assert math.isnan(row["log2PeakDelta"])
        assert math.isnan(row["log2KineticDelta"])

    # R3: per-subject detail DataFrame returned in intra-subject mode
    def test_per_subject_returned_in_intra_subject_mode(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "timepoint": "Day0", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.9, "timepoint": "Day7", "subject": "sub1"},
            {"sampleId": "s3", "elementId": "a", "frequency": 0.3, "timepoint": "Day0", "subject": "sub2"},
            {"sampleId": "s4", "elementId": "a", "frequency": 0.7, "timepoint": "Day7", "subject": "sub2"},
        ])
        result, per_subject = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=True, mode="intra-subject", min_subject_count=1
        )
        assert per_subject is not None
        assert len(per_subject) == 2  # 1 element x 2 subjects
        assert "elementId" in per_subject.columns
        assert "subject" in per_subject.columns
        assert "peakTimepoint" in per_subject.columns
        assert "temporalShiftIndex" in per_subject.columns
        assert "log2PeakDelta" in per_subject.columns
        assert "log2KineticDelta" in per_subject.columns
        subs = sorted(per_subject["subject"].to_list())
        assert subs == ["sub1", "sub2"]

    # Population mode: per_subject is None (not computed)
    def test_per_subject_none_in_population_mode(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "timepoint": "Day0", "subject": "sub1"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.8, "timepoint": "Day7", "subject": "sub1"},
        ])
        result, per_subject = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=True, mode="population", min_subject_count=1
        )
        assert per_subject is None

    # No subject → per_subject is None
    def test_per_subject_none_when_no_subject(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.2, "timepoint": "Day0"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.8, "timepoint": "Day7"},
        ])
        result, per_subject = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=False, mode="population", min_subject_count=1
        )
        assert per_subject is None

    # Multiple elements: each computed independently
    def test_multiple_elements(self, make_df):
        df = make_df([
            {"sampleId": "s1", "elementId": "a", "frequency": 0.1, "timepoint": "Day0"},
            {"sampleId": "s1", "elementId": "b", "frequency": 0.9, "timepoint": "Day0"},
            {"sampleId": "s2", "elementId": "a", "frequency": 0.9, "timepoint": "Day7"},
            {"sampleId": "s2", "elementId": "b", "frequency": 0.1, "timepoint": "Day7"},
        ])
        result, _ = compute_temporal_metrics(
            df, ["Day0", "Day7"], has_subject=False, mode="population", min_subject_count=1
        )
        assert len(result) == 2
        a = result.filter(pl.col("elementId") == "a").row(0, named=True)
        b = result.filter(pl.col("elementId") == "b").row(0, named=True)
        # Clone "a" expands, "b" contracts
        assert a["log2KineticDelta"] > 0
        assert b["log2KineticDelta"] < 0
