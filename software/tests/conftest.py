"""Shared pytest fixtures for compartment_analysis tests."""

import polars as pl
import pytest


@pytest.fixture
def make_df():
    """Factory fixture for constructing small DataFrames from a list of row dicts."""
    def _make_df(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows)
    return _make_df
