"""Behavioral tests for restriction index and grouping metrics (R11-R13).

Run: uv run pytest tests/test_restriction_index.py -v
"""

import math

import numpy as np
import pytest

from compartment_analysis import restriction_index, shannon_entropy


class TestShannonEntropy:
    """Tests for shannon_entropy — foundation of Restriction Index (R11)."""

    # Uniform distribution across N groups → H = log2(N)
    def test_uniform_distribution(self):
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert shannon_entropy(p) == pytest.approx(math.log2(4))

    # Single nonzero value → H = 0 (no uncertainty)
    def test_single_nonzero(self):
        p = np.array([0.0, 0.0, 1.0])
        assert shannon_entropy(p) == pytest.approx(0.0)

    # Empty / all-zero array → H = 0
    def test_all_zeros(self):
        p = np.array([0.0, 0.0])
        assert shannon_entropy(p) == pytest.approx(0.0)

    # Empty array → H = 0
    def test_empty_array(self):
        p = np.array([])
        assert shannon_entropy(p) == pytest.approx(0.0)

    # Two equal values → H = log2(2) = 1.0
    def test_two_equal(self):
        p = np.array([0.5, 0.5])
        assert shannon_entropy(p) == pytest.approx(1.0)

    # Entropy auto-normalizes: [10, 10] should equal [0.5, 0.5]
    def test_auto_normalizes(self):
        p_raw = np.array([10.0, 10.0])
        p_norm = np.array([0.5, 0.5])
        assert shannon_entropy(p_raw) == pytest.approx(shannon_entropy(p_norm))

    # Skewed distribution has lower entropy than uniform
    def test_skewed_less_than_uniform(self):
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        skewed = np.array([0.9, 0.05, 0.03, 0.02])
        assert shannon_entropy(skewed) < shannon_entropy(uniform)


class TestRestrictionIndex:
    """Tests for restriction_index — R11 formula: RI = 1 - H(p) / log2(N)."""

    # Equal frequency across all groups → RI = 0.0 (no restriction)
    def test_uniform_equals_zero(self):
        freq = np.array([0.25, 0.25, 0.25, 0.25])
        assert restriction_index(freq) == pytest.approx(0.0, abs=1e-10)

    # Clone in exactly one group → RI = 1.0 (R34: maximally restricted)
    def test_single_group_equals_one(self):
        freq = np.array([0.0, 0.0, 1.0])
        assert restriction_index(freq) == pytest.approx(1.0)

    # Clone absent from all groups → RI = NaN (guard case)
    def test_all_zero_returns_nan(self):
        freq = np.array([0.0, 0.0, 0.0])
        assert math.isnan(restriction_index(freq))

    # Hand-calculated: 2 groups, p = [0.75, 0.25]
    # H = -(0.75*log2(0.75) + 0.25*log2(0.25)) = 0.8113
    # RI = 1 - 0.8113/log2(2) = 1 - 0.8113 = 0.1887
    def test_two_groups_hand_calculated(self):
        freq = np.array([0.75, 0.25])
        expected_h = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
        expected_ri = 1.0 - expected_h / math.log2(2)
        assert restriction_index(freq) == pytest.approx(expected_ri, abs=1e-6)

    # Highly skewed → RI close to 1.0
    def test_three_groups_skewed(self):
        freq = np.array([0.98, 0.01, 0.01])
        ri = restriction_index(freq)
        assert ri > 0.8
        assert ri <= 1.0

    # RI is in [0, 1] for any valid input
    def test_ri_range(self):
        for _ in range(20):
            freq = np.random.dirichlet(np.ones(5))
            ri = restriction_index(freq)
            assert 0.0 <= ri <= 1.0 or math.isnan(ri)

    # Two groups with equal frequency → RI = 0.0
    def test_two_equal_groups(self):
        freq = np.array([0.5, 0.5])
        assert restriction_index(freq) == pytest.approx(0.0, abs=1e-10)

    # Only zeros among some groups but one nonzero: N=1 nonzero → RI = 1.0
    def test_one_nonzero_among_many(self):
        freq = np.array([0.0, 0.0, 0.0, 5.0, 0.0])
        assert restriction_index(freq) == pytest.approx(1.0)
