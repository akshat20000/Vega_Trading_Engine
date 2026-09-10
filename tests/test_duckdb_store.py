"""
Tests for vega/data/duckdb_store.py

The standard pattern in every test:
  1. Use ParquetStore to save synthetic bars to tmp_path (the source of truth).
  2. Use DuckDBStore to query those Parquet files.
  3. Assert that the results are correct.

This mirrors the real system design:
    ParquetStore.save -> .parquet file -> DuckDBStore.query

No CSV files, no real market data, no network access.

Test coverage:
  - count_bars() returns correct count
  - query_bars() with no filter returns all bars
  - query_bars() returns bars sorted chronologically
  - query_bars() with start filter only
  - query_bars() with end filter only
  - query_bars() with both start and end filters
  - query_bars() returns empty list for date range with no matches
  - query_bars() start == end returns exactly matching bars
  - min_low() returns the correct minimum
  - max_high() returns the correct maximum
  - avg_volume() returns the correct average (within floating-point tolerance)
  - FileNotFoundError raised for nonexistent files (all methods)
  - DuckDBStore and ParquetStore return consistent data for the same file
  - run_sql() executes arbitrary SQL and returns correct results
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import make_bar
from vega.data.duckdb_store import DuckDBStore
from vega.data.models import Bar
from vega.data.parquet_store import ParquetStore


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def parquet_store() -> ParquetStore:
    return ParquetStore()


@pytest.fixture
def db_store() -> DuckDBStore:
    return DuckDBStore()


@pytest.fixture
def five_dated_bars() -> list[Bar]:
    """
    Five bars on specific dates with distinct, hand-verifiable OHLCV values.

    Date       open    high    low     close   volume
    2023-01-02  100     110      90     105     1000
    2023-01-03  110     120     100     115     1100
    2023-01-04  120     130     110     125     1200
    2023-01-05  130     140     120     135     1300
    2023-01-06  140     150     130     145     1400

    Known aggregates:
        min_low    = 90.0
        max_high   = 150.0
        avg_volume = (1000+1100+1200+1300+1400)/5 = 1200.0
    """
    base = datetime(2023, 1, 2)
    return [
        make_bar(
            timestamp = base + timedelta(days=i),
            open_     = 100.0 + i * 10,
            high      = 110.0 + i * 10,
            low       = 90.0  + i * 10,
            close     = 105.0 + i * 10,
            volume    = 1000.0 + i * 100,
        )
        for i in range(5)
    ]


@pytest.fixture
def saved_parquet(
    parquet_store: ParquetStore,
    five_dated_bars: list[Bar],
    tmp_path: Path,
) -> Path:
    """
    Convenience fixture: saves five_dated_bars to a Parquet file and
    returns the path. Any test that needs a pre-populated Parquet file
    can request this fixture instead of repeating the save() call.
    """
    filepath = tmp_path / "NIFTY50.parquet"
    parquet_store.save(five_dated_bars, filepath)
    return filepath


# ─────────────────────────────────────────────────────────────────────────────
# 1. count_bars()
# ─────────────────────────────────────────────────────────────────────────────

class TestCountBars:

    def test_count_returns_correct_number(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert db_store.count_bars(saved_parquet) == 5

    def test_count_returns_int(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        result = db_store.count_bars(saved_parquet)
        assert isinstance(result, int)

    def test_count_nonexistent_file_raises_file_not_found(
        self, db_store: DuckDBStore, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            db_store.count_bars(tmp_path / "missing.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# 2. query_bars() — no filter
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryBarsNoFilter:

    def test_no_filter_returns_all_bars(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        bars = db_store.query_bars(saved_parquet)
        assert len(bars) == 5

    def test_no_filter_returns_bar_objects(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        bars = db_store.query_bars(saved_parquet)
        assert all(isinstance(b, Bar) for b in bars)

    def test_no_filter_returns_chronological_order(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """Bars must always be returned oldest-first, regardless of file order."""
        bars = db_store.query_bars(saved_parquet)
        for i in range(1, len(bars)):
            assert bars[i].timestamp > bars[i - 1].timestamp

    def test_no_filter_nonexistent_file_raises_file_not_found(
        self, db_store: DuckDBStore, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            db_store.query_bars(tmp_path / "missing.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# 3. query_bars() — date range filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestQueryBarsDateFilter:
    """
    Reference data from five_dated_bars fixture:
        Day 0: 2023-01-02
        Day 1: 2023-01-03
        Day 2: 2023-01-04
        Day 3: 2023-01-05
        Day 4: 2023-01-06
    """

    def test_start_filter_excludes_earlier_bars(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """start=2023-01-04 should return 3 bars (Jan 4, 5, 6)."""
        bars = db_store.query_bars(saved_parquet, start=datetime(2023, 1, 4))
        assert len(bars) == 3
        assert bars[0].timestamp == datetime(2023, 1, 4)

    def test_end_filter_excludes_later_bars(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """end=2023-01-04 should return 3 bars (Jan 2, 3, 4)."""
        bars = db_store.query_bars(saved_parquet, end=datetime(2023, 1, 4))
        assert len(bars) == 3
        assert bars[-1].timestamp == datetime(2023, 1, 4)

    def test_start_and_end_filter_returns_middle_bars(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """start=2023-01-03, end=2023-01-05 should return 3 bars."""
        bars = db_store.query_bars(
            saved_parquet,
            start=datetime(2023, 1, 3),
            end=datetime(2023, 1, 5),
        )
        assert len(bars) == 3
        assert bars[0].timestamp == datetime(2023, 1, 3)
        assert bars[-1].timestamp == datetime(2023, 1, 5)

    def test_start_filter_is_inclusive(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """A bar whose timestamp exactly equals start must be included."""
        bars = db_store.query_bars(saved_parquet, start=datetime(2023, 1, 2))
        assert bars[0].timestamp == datetime(2023, 1, 2)

    def test_end_filter_is_inclusive(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """A bar whose timestamp exactly equals end must be included."""
        bars = db_store.query_bars(saved_parquet, end=datetime(2023, 1, 6))
        assert bars[-1].timestamp == datetime(2023, 1, 6)

    def test_single_day_filter_returns_one_bar(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """start == end returns exactly the one bar on that date."""
        bars = db_store.query_bars(
            saved_parquet,
            start=datetime(2023, 1, 4),
            end=datetime(2023, 1, 4),
        )
        assert len(bars) == 1
        assert bars[0].timestamp == datetime(2023, 1, 4)

    def test_date_range_with_no_matches_returns_empty_list(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """
        A date range that falls between bars or entirely outside the data
        must return an empty list, not raise an error.
        """
        bars = db_store.query_bars(
            saved_parquet,
            start=datetime(2024, 1, 1),   # far in the future
            end=datetime(2024, 12, 31),
        )
        assert bars == []

    def test_filtered_result_preserves_ohlcv_values(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """
        The bar for 2023-01-04 (i=2 in five_dated_bars) should have:
            open=120, high=130, low=110, close=125, volume=1200
        """
        bars = db_store.query_bars(
            saved_parquet,
            start=datetime(2023, 1, 4),
            end=datetime(2023, 1, 4),
        )
        bar = bars[0]
        assert bar.open   == 120.0
        assert bar.high   == 130.0
        assert bar.low    == 110.0
        assert bar.close  == 125.0
        assert bar.volume == 1200.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aggregate queries
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateQueries:
    """
    Reference aggregates from five_dated_bars:
        min_low    = 90.0   (day 0: low=90)
        max_high   = 150.0  (day 4: high=150)
        avg_volume = 1200.0 (mean of 1000, 1100, 1200, 1300, 1400)
    """

    def test_min_low_returns_correct_value(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert db_store.min_low(saved_parquet) == 90.0

    def test_min_low_returns_float(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert isinstance(db_store.min_low(saved_parquet), float)

    def test_max_high_returns_correct_value(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert db_store.max_high(saved_parquet) == 150.0

    def test_max_high_returns_float(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert isinstance(db_store.max_high(saved_parquet), float)

    def test_avg_volume_returns_correct_value(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """
        avg_volume is tested with pytest.approx because DuckDB's internal
        floating-point accumulation order may differ slightly from Python's
        sum()/len(). The result should be 1200.0 exactly for these values,
        but approx() makes the test robust to any rounding differences.
        """
        assert db_store.avg_volume(saved_parquet) == pytest.approx(1200.0, rel=1e-9)

    def test_avg_volume_returns_float(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        assert isinstance(db_store.avg_volume(saved_parquet), float)

    def test_min_low_nonexistent_file_raises_file_not_found(
        self, db_store: DuckDBStore, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            db_store.min_low(tmp_path / "missing.parquet")

    def test_max_high_nonexistent_file_raises_file_not_found(
        self, db_store: DuckDBStore, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            db_store.max_high(tmp_path / "missing.parquet")

    def test_avg_volume_nonexistent_file_raises_file_not_found(
        self, db_store: DuckDBStore, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            db_store.avg_volume(tmp_path / "missing.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Consistency between DuckDBStore and ParquetStore
# ─────────────────────────────────────────────────────────────────────────────

class TestConsistencyWithParquetStore:
    """
    DuckDBStore and ParquetStore read from the same Parquet file.
    Their outputs must agree on every field.
    """

    def test_duckdb_bars_match_parquet_store_bars(
        self,
        db_store: DuckDBStore,
        parquet_store: ParquetStore,
        five_dated_bars: list[Bar],
        tmp_path: Path,
    ) -> None:
        filepath = tmp_path / "consistency.parquet"
        parquet_store.save(five_dated_bars, filepath)

        ps_bars = parquet_store.load(filepath)
        db_bars = db_store.query_bars(filepath)

        assert len(ps_bars) == len(db_bars)

        for ps_bar, db_bar in zip(ps_bars, db_bars):
            assert db_bar.timestamp == ps_bar.timestamp
            assert db_bar.open      == ps_bar.open
            assert db_bar.high      == ps_bar.high
            assert db_bar.low       == ps_bar.low
            assert db_bar.close     == ps_bar.close
            assert db_bar.volume    == ps_bar.volume


# ─────────────────────────────────────────────────────────────────────────────
# 6. run_sql() — ad-hoc query method
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSQL:

    def test_run_sql_count(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """run_sql should be able to execute the same COUNT query manually."""
        rows = db_store.run_sql(
            f"SELECT COUNT(*) FROM read_parquet('{saved_parquet.as_posix()}')"
        )
        assert rows[0][0] == 5

    def test_run_sql_returns_list_of_tuples(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        rows = db_store.run_sql(
            f"SELECT open, close FROM read_parquet('{saved_parquet.as_posix()}')"
            " ORDER BY timestamp LIMIT 1"
        )
        assert isinstance(rows, list)
        assert isinstance(rows[0], tuple)
        assert len(rows[0]) == 2

    def test_run_sql_with_params(
        self, db_store: DuckDBStore, saved_parquet: Path
    ) -> None:
        """run_sql must support parameterised queries."""
        rows = db_store.run_sql(
            "SELECT COUNT(*) FROM read_parquet(?)",
            params=[str(saved_parquet)]
        )
        assert rows[0][0] == 5
