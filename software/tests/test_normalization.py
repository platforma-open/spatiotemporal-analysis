"""Behavioral tests for normalization functions (R8, R9).

Run: uv run pytest tests/test_normalization.py -v
"""

import math

import polars as pl
import pytest

from compartment_analysis import compute_clr, compute_relative_frequency


class TestRelativeFrequency:
    """Tests for compute_relative_frequency — R8."""

    # Two samples with known abundances; frequency = abundance / sample_total
    def test_basic_two_samples(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s2", "s2"],
            "elementId": ["a", "b", "a", "b"],
            "abundance": [30.0, 70.0, 50.0, 50.0],
        })
        result = compute_relative_frequency(df)
        s1 = result.filter(pl.col("sampleId") == "s1").sort("elementId")
        assert s1["frequency"][0] == pytest.approx(0.3)  # a in s1: 30/100
        assert s1["frequency"][1] == pytest.approx(0.7)  # b in s1: 70/100

        s2 = result.filter(pl.col("sampleId") == "s2").sort("elementId")
        assert s2["frequency"][0] == pytest.approx(0.5)
        assert s2["frequency"][1] == pytest.approx(0.5)

    # Per-sample frequencies must sum to 1.0
    def test_frequencies_sum_to_one(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s1"],
            "elementId": ["a", "b", "c"],
            "abundance": [10.0, 20.0, 70.0],
        })
        result = compute_relative_frequency(df)
        total = result["frequency"].sum()
        assert total == pytest.approx(1.0)

    # Sample with zero total abundance should be excluded, not cause division by zero
    def test_zero_total_sample_excluded(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s2"],
            "elementId": ["a", "b", "a"],
            "abundance": [50.0, 50.0, 0.0],
        })
        result = compute_relative_frequency(df)
        # s2 had total=0, but clone "a" in s2 has abundance 0 which means
        # s2 total is 0 → row dropped by filter(sampleTotal > 0)
        assert result.filter(pl.col("sampleId") == "s2").is_empty()

    # Single element per sample: frequency should be 1.0
    def test_single_element_per_sample(self):
        df = pl.DataFrame({
            "sampleId": ["s1"],
            "elementId": ["a"],
            "abundance": [42.0],
        })
        result = compute_relative_frequency(df)
        assert result["frequency"][0] == pytest.approx(1.0)


class TestCLR:
    """Tests for compute_clr — R9."""

    def _make_df(self, sample_data: dict[str, list]) -> pl.DataFrame:
        """Helper to build a DataFrame with sampleId, elementId, abundance."""
        return pl.DataFrame(sample_data)

    # CLR values within each sample should sum to approximately 0 (centered)
    def test_clr_values_sum_to_zero_per_sample(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s1"],
            "elementId": ["a", "b", "c"],
            "abundance": [10.0, 20.0, 70.0],
        })
        result = compute_clr(df, mode="population", has_subject=False)
        clr_sum = result["frequency"].sum()
        assert clr_sum == pytest.approx(0.0, abs=1e-10)

    # No zeros: CLR = log(freq_i / geometric_mean(freq))
    def test_clr_basic_no_zeros(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1"],
            "elementId": ["a", "b"],
            "abundance": [25.0, 75.0],
        })
        result = compute_clr(df, mode="population", has_subject=False)
        freqs = result.sort("elementId")["frequency"].to_list()
        # With 2 components, no zeros: standard CLR
        # freq = [0.25, 0.75], geo_mean = exp(mean(log([0.25, 0.75])))
        # clr_a = log(0.25) - mean(log([0.25, 0.75]))
        # clr_b = log(0.75) - mean(log([0.25, 0.75]))
        geo_mean = math.exp((math.log(0.25) + math.log(0.75)) / 2)
        expected_a = math.log(0.25 / geo_mean)
        expected_b = math.log(0.75 / geo_mean)
        assert freqs[0] == pytest.approx(expected_a, abs=1e-6)
        assert freqs[1] == pytest.approx(expected_b, abs=1e-6)

    # Zero frequency replaced with delta = 0.65 * min(nonzero) / D, then CLR applied
    def test_clr_zero_replacement(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s1"],
            "elementId": ["a", "b", "c"],
            "abundance": [0.0, 50.0, 50.0],
        })
        result = compute_clr(df, mode="population", has_subject=False)
        # Element "a" had zero abundance → should get a small replacement value
        # The CLR value for "a" should be much smaller than for "b" and "c"
        r = result.sort("elementId")
        assert r["frequency"][0] < r["frequency"][1]
        assert r["frequency"][0] < r["frequency"][2]
        # CLR still sums to ~0
        assert r["frequency"].sum() == pytest.approx(0.0, abs=1e-10)

    # Population mode: CLR applied globally across all samples
    def test_clr_population_mode_is_global(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s2", "s2"],
            "elementId": ["a", "b", "a", "b"],
            "abundance": [30.0, 70.0, 60.0, 40.0],
            "subject": ["sub1", "sub1", "sub2", "sub2"],
        })
        result = compute_clr(df, mode="population", has_subject=True)
        # In population mode, CLR is global — all samples use the same min_nonzero
        # Each sample's CLR values should sum to ~0 independently
        for sid in ["s1", "s2"]:
            sample = result.filter(pl.col("sampleId") == sid)
            assert sample["frequency"].sum() == pytest.approx(0.0, abs=1e-10)

    # Intra-subject CLR uses per-subject min(nonzero) for zero replacement;
    # population mode uses global min(nonzero). When a subject's own minimum
    # differs from the global minimum, the delta applied to zero frequencies
    # differs — so CLR output must differ between modes for that subject.
    def test_clr_intra_subject_differs_from_population(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s1", "s2", "s2", "s2"],
            "elementId": ["a", "b", "c", "a", "b", "c"],
            # sub1: freqs = [0.1, 0.9, 0.0]   → min_nz = 0.1
            # sub2: freqs = [0.0, 0.5, 0.5]   → min_nz = 0.5
            # Global min_nz = 0.1 → delta differs for sub2's zero element "a"
            "abundance": [10.0, 90.0, 0.0, 0.0, 50.0, 50.0],
            "subject": ["sub1", "sub1", "sub1", "sub2", "sub2", "sub2"],
        })
        result_intra = compute_clr(df, mode="intra-subject", has_subject=True)
        result_pop = compute_clr(df, mode="population", has_subject=True)

        def freq_of(result, sid, eid):
            return result.filter(
                (pl.col("sampleId") == sid) & (pl.col("elementId") == eid)
            )["frequency"][0]

        # sub2 element "a" was zero — its CLR value depends on delta, which
        # differs between modes (per-subject min=0.5 vs global min=0.1)
        assert freq_of(result_intra, "s2", "a") != pytest.approx(
            freq_of(result_pop, "s2", "a"), abs=1e-6
        )
        # Both modes still produce CLR-centered output (per-sample sum ~ 0)
        for result in (result_intra, result_pop):
            for sid in ["s1", "s2"]:
                sample = result.filter(pl.col("sampleId") == sid)
                assert sample["frequency"].sum() == pytest.approx(0.0, abs=1e-10)

    # Multiple samples per subject in intra-subject mode
    def test_clr_intra_subject_multiple_samples(self):
        df = pl.DataFrame({
            "sampleId": ["s1", "s1", "s2", "s2"],
            "elementId": ["a", "b", "a", "b"],
            "abundance": [20.0, 80.0, 40.0, 60.0],
            "subject": ["sub1", "sub1", "sub1", "sub1"],
        })
        result = compute_clr(df, mode="intra-subject", has_subject=True)
        # Each sample's CLR should sum to ~0
        for sid in ["s1", "s2"]:
            sample = result.filter(pl.col("sampleId") == sid)
            assert sample["frequency"].sum() == pytest.approx(0.0, abs=1e-10)
