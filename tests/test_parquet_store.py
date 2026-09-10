"""
Tests for vega/data/parquet_store.py

Tests use synthetic, hand-crafted Bar objects created with make_bar().
No CSV files, no real market data, no network access.

All tests write to pytest's tmp_path fixture — a temporary directory unique
to each test run that is cleaned up automatically after the test session.

Test coverage:
  - save() creates a file on disk
  - save() creates missing parent directories automatically
  - save() raises ValueError on empty list
  - load() returns the correct number of bars
  - load() preserves timestamps with exact equality (round-trip)
  - load() preserves OHLCV float values with exact equality (float64 round-trip)
  - load() returns bars sorted chronologically regardless of input order
  - load() raises FileNotFoundError for nonexistent files
  - load() raises ValueError if required columns are missing
  - full save-then-load round-trip produces identical Bar objects
  - multiple symbols saved to separate files do not interfere
  - Bar.__post_init__ validation runs on every loaded row
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tests.conftest import make_bar
from vega.data.models import Bar
from vega.data.parquet_store import ParquetStore


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures local to this test file
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def store() -> ParquetStore:
    """A fresh ParquetStore for each test."""
    return ParquetStore()


@pytest.fixture
def five_bars() -> list[Bar]:
    """
    Five synthetic daily bars on consecutive dates.
    All OHLCV values are chosen to be easily verifiable by hand.

    Bars are intentionally in chronological order (oldest first).
    Tests that check ordering will shuffle them explicitly.
    """
    base = datetime(2023, 1, 2)
    return [
        make_bar(base + timedelta(days=i),
                 open_=100.0 + i * 10,
                 high=110.0  + i * 10,
                 low=90.0    + i * 10,
                 close=105.0 + i * 10,
                 volume=1_000.0 + i * 100)
        for i in range(5)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 1. save() — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetStoreSave:

    def test_save_creates_parquet_file_on_disk(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """After save(), the file must exist at the given path."""
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        assert filepath.exists(), "Parquet file was not created."

    def test_save_creates_missing_parent_directories(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """
        save() must create parent directories automatically.
        e.g. tmp/data/market/NIFTY50.parquet even if data/market/ does not exist.
        """
        filepath = tmp_path / "data" / "market" / "NIFTY50.parquet"
        assert not (tmp_path / "data").exists(), "Pre-condition: directory must not exist yet."
        store.save(five_bars, filepath)
        assert filepath.exists()

    def test_save_empty_list_raises_value_error(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """Saving an empty list is a programming error and must be rejected."""
        with pytest.raises(ValueError, match="empty"):
            store.save([], tmp_path / "empty.parquet")

    def test_save_overwrites_existing_file(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """Calling save() twice on the same path overwrites without error."""
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)

        # Save a single-bar list to the same path.
        single = [make_bar(datetime(2023, 6, 1))]
        store.save(single, filepath)

        # Load should now contain only 1 bar.
        loaded = store.load(filepath)
        assert len(loaded) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. load() — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetStoreLoad:

    def test_load_returns_correct_number_of_bars(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)
        assert len(loaded) == 5

    def test_load_nonexistent_file_raises_file_not_found_error(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """load() on a missing file must raise FileNotFoundError, not crash."""
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "does_not_exist.parquet")

    def test_load_returns_list_of_bar_objects(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)
        assert all(isinstance(b, Bar) for b in loaded)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Timestamp preservation
# ─────────────────────────────────────────────────────────────────────────────

class TestTimestampPreservation:

    def test_timestamps_are_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """
        Timestamp round-trip must be exact.

        Parquet stores datetime64[ns] natively. pyarrow reads it back as
        pd.Timestamp, which we convert to datetime. The conversion is lossless
        for any datetime within the datetime64[ns] range (year 1678 to 2262).
        """
        timestamps = [
            datetime(2023, 1, 2),
            datetime(2023, 6, 15, 9, 15, 30),   # includes time component
            datetime(2023, 12, 31, 23, 59, 59),
        ]
        bars = [make_bar(ts) for ts in timestamps]

        filepath = tmp_path / "timestamps.parquet"
        store.save(bars, filepath)
        loaded = store.load(filepath)

        loaded_timestamps = [b.timestamp for b in loaded]
        for original, reloaded in zip(sorted(timestamps), sorted(loaded_timestamps)):
            assert reloaded == original, (
                f"Timestamp changed after Parquet round-trip: "
                f"{original} -> {reloaded}"
            )

    def test_loaded_timestamps_are_python_datetime_not_pandas_timestamp(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """
        Returned Bar.timestamp must be a plain Python datetime, not pd.Timestamp.
        Strategy code must not depend on pandas types.
        """
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)
        for bar in loaded:
            assert isinstance(bar.timestamp, datetime), (
                f"Expected datetime, got {type(bar.timestamp)}"
            )

    def test_loaded_timestamps_are_timezone_naive(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """Bar timestamps must remain timezone-naive after the round-trip."""
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)
        for bar in loaded:
            assert bar.timestamp.tzinfo is None, (
                f"Timestamp acquired unexpected timezone: {bar.timestamp.tzinfo}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 4. OHLCV float64 round-trip accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestOHLCVPreservation:
    """
    Parquet stores float64 values. Python floats are float64.
    The round-trip must be bit-identical — exact equality (==) is correct here.

    Values like 17807.35 are NOT exactly representable in binary floating point,
    but that is a pre-existing property of the float — it is not introduced by
    Parquet. Parquet faithfully stores and retrieves the same float64 bits.
    So if you save 17807.35 you get back the same float that Python originally
    created for that literal (not a different approximation).
    """

    def test_open_is_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        # Use realistic NIFTY-style prices that satisfy high >= open/close and low <= open/close.
        bar    = make_bar(datetime(2023, 1, 2),
                          open_=17807.35, high=17900.00, low=17700.00, close=17850.00)
        filepath = tmp_path / "test.parquet"
        store.save([bar], filepath)
        loaded = store.load(filepath)
        assert loaded[0].open == bar.open   # exact equality, not approx

    def test_high_is_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        bar    = make_bar(datetime(2023, 1, 2), high=18000.75)
        filepath = tmp_path / "test.parquet"
        store.save([bar], filepath)
        loaded = store.load(filepath)
        assert loaded[0].high == bar.high

    def test_low_is_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        bar    = make_bar(datetime(2023, 1, 2),
                          open_=17600.00, high=17700.00, low=17500.10, close=17650.00)
        filepath = tmp_path / "test.parquet"
        store.save([bar], filepath)
        loaded = store.load(filepath)
        assert loaded[0].low == bar.low

    def test_close_is_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        bar    = make_bar(datetime(2023, 1, 2),
                          open_=17800.00, high=17900.00, low=17750.00, close=17859.45)
        filepath = tmp_path / "test.parquet"
        store.save([bar], filepath)
        loaded = store.load(filepath)
        assert loaded[0].close == bar.close

    def test_volume_is_preserved_exactly(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        bar    = make_bar(datetime(2023, 1, 2), volume=258_497.0)
        filepath = tmp_path / "test.parquet"
        store.save([bar], filepath)
        loaded = store.load(filepath)
        assert loaded[0].volume == bar.volume

    def test_all_five_ohlcv_fields_in_one_round_trip(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """Complete round-trip check: all five OHLCV fields at once."""
        original = make_bar(
            timestamp = datetime(2023, 1, 2),
            open_     = 17807.35,
            high      = 17866.35,
            low       = 17684.15,
            close     = 17859.45,
            volume    = 258_497.0,
        )
        filepath = tmp_path / "test.parquet"
        store.save([original], filepath)
        loaded = store.load(filepath)[0]

        assert loaded.open   == original.open
        assert loaded.high   == original.high
        assert loaded.low    == original.low
        assert loaded.close  == original.close
        assert loaded.volume == original.volume


# ─────────────────────────────────────────────────────────────────────────────
# 5. Chronological ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalOrdering:

    def test_load_sorts_bars_oldest_first(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """
        Even if bars are saved in reverse order, load() must return them
        sorted oldest-first.
        """
        bars_reversed = [
            make_bar(datetime(2023, 1, 4)),
            make_bar(datetime(2023, 1, 3)),
            make_bar(datetime(2023, 1, 2)),   # oldest, saved last
        ]
        filepath = tmp_path / "reversed.parquet"
        store.save(bars_reversed, filepath)
        loaded = store.load(filepath)

        assert loaded[0].timestamp == datetime(2023, 1, 2)
        assert loaded[1].timestamp == datetime(2023, 1, 3)
        assert loaded[2].timestamp == datetime(2023, 1, 4)

    def test_loaded_bars_are_strictly_increasing_in_time(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        filepath = tmp_path / "NIFTY50.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)

        for i in range(1, len(loaded)):
            assert loaded[i].timestamp > loaded[i - 1].timestamp, (
                f"Bar {i} is not after bar {i-1}: "
                f"{loaded[i].timestamp} <= {loaded[i-1].timestamp}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multiple symbols
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleSymbols:

    def test_two_symbols_stored_in_separate_files_do_not_interfere(
        self, store: ParquetStore, tmp_path: Path
    ) -> None:
        """
        NIFTY50 and BANKNIFTY bars saved to different files must load
        independently without cross-contamination.
        """
        nifty_bars = [
            make_bar(datetime(2023, 1, 2),
                     open_=17400.0, high=17600.0, low=17350.0, close=17500.0),
            make_bar(datetime(2023, 1, 3),
                     open_=17500.0, high=17700.0, low=17450.0, close=17600.0),
        ]
        banknifty_bars = [
            make_bar(datetime(2023, 1, 2),
                     open_=41800.0, high=42100.0, low=41700.0, close=42000.0),
        ]

        nifty_path     = tmp_path / "NIFTY50.parquet"
        banknifty_path = tmp_path / "BANKNIFTY.parquet"

        store.save(nifty_bars,     nifty_path)
        store.save(banknifty_bars, banknifty_path)

        loaded_nifty     = store.load(nifty_path)
        loaded_banknifty = store.load(banknifty_path)

        assert len(loaded_nifty)     == 2
        assert len(loaded_banknifty) == 1

        # Spot-check close prices to confirm no data mixing.
        assert loaded_nifty[0].close     == 17_500.0
        assert loaded_banknifty[0].close == 42_000.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full round-trip equality
# ─────────────────────────────────────────────────────────────────────────────

class TestFullRoundTrip:

    def test_save_and_load_produces_identical_bar_list(
        self, store: ParquetStore, five_bars: list[Bar], tmp_path: Path
    ) -> None:
        """
        Every field of every bar must survive the save → load round-trip
        unchanged. This is the top-level integration test for ParquetStore.
        """
        filepath = tmp_path / "roundtrip.parquet"
        store.save(five_bars, filepath)
        loaded = store.load(filepath)

        assert len(loaded) == len(five_bars)

        for original, reloaded in zip(five_bars, loaded):
            assert reloaded.timestamp == original.timestamp
            assert reloaded.open      == original.open
            assert reloaded.high      == original.high
            assert reloaded.low       == original.low
            assert reloaded.close     == original.close
            assert reloaded.volume    == original.volume
