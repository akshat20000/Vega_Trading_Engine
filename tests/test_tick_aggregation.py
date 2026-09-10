"""
Tests for vega/data/tick_aggregator.py

Each test uses hand-crafted ticks with known prices and timestamps so that
the expected OHLCV output can be calculated manually and verified exactly.

No CSV files are read in any test — all data is created inline.

Test coverage:
  - OHLCV correctness for a single bucket
  - Volume sum correctness
  - Multiple buckets in one call
  - Single tick per bucket (open == high == low == close)
  - Empty bucket gap between two occupied buckets (gap is dropped)
  - 1-minute aggregation (intraday)
  - 1-hour aggregation (intraday)
  - Daily aggregation spanning multiple days
  - Output is sorted chronologically
  - Unsorted input ticks are handled correctly
  - Empty tick list raises ValueError
  - Unknown frequency string raises ValueError
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.conftest import make_tick
from vega.data.models import Bar, Tick
from vega.data.tick_aggregator import aggregate_ticks


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def dt(year: int, month: int, day: int,
       hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    """Compact datetime constructor used throughout this file."""
    return datetime(year, month, day, hour, minute, second)


# ─────────────────────────────────────────────────────────────────────────────
# 1. OHLCV correctness — single 1-minute bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestOHLCVCorrectness:
    """Verify that open/high/low/close/volume are computed correctly."""

    def test_open_is_first_tick_price(self) -> None:
        """open must equal the price of the first tick in the bucket."""
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 1
        assert bars[0].open == pytest.approx(100.0)

    def test_close_is_last_tick_price(self) -> None:
        """close must equal the price of the last tick in the bucket."""
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert bars[0].close == pytest.approx(98.0)

    def test_high_is_max_price_in_bucket(self) -> None:
        """high must equal the maximum price across all ticks in the bucket."""
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),  # max
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert bars[0].high == pytest.approx(105.0)

    def test_low_is_min_price_in_bucket(self) -> None:
        """low must equal the minimum price across all ticks in the bucket."""
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),  # min
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert bars[0].low == pytest.approx(98.0)

    def test_volume_is_sum_of_all_tick_volumes(self) -> None:
        """volume must equal the sum of all tick volumes in the bucket."""
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert bars[0].volume == pytest.approx(45.0)   # 10 + 20 + 15

    def test_full_ohlcv_in_one_assertion(self) -> None:
        """
        Complete OHLCV check for a three-tick, one-minute bucket.
        This is the canonical 'sanity check' test.

        Ticks:  (09:15:00, 100), (09:15:30, 105), (09:15:59, 98)
        Expected bar:
            open   = 100  (first)
            high   = 105  (max)
            low    = 98   (min)
            close  = 98   (last)
            volume = 45   (sum)
        """
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")

        assert len(bars) == 1
        bar = bars[0]
        assert bar.open   == pytest.approx(100.0)
        assert bar.high   == pytest.approx(105.0)
        assert bar.low    == pytest.approx(98.0)
        assert bar.close  == pytest.approx(98.0)
        assert bar.volume == pytest.approx(45.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Single tick per bucket
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleTickBucket:
    """When a bucket contains exactly one tick, open == high == low == close."""

    def test_single_tick_all_prices_equal(self) -> None:
        ticks = [make_tick(dt(2023, 1, 2, 9, 15, 0), price=200.0, volume=50)]
        bars  = aggregate_ticks(ticks, frequency="1min")

        assert len(bars) == 1
        bar = bars[0]
        assert bar.open   == pytest.approx(200.0)
        assert bar.high   == pytest.approx(200.0)
        assert bar.low    == pytest.approx(200.0)
        assert bar.close  == pytest.approx(200.0)
        assert bar.volume == pytest.approx(50.0)

    def test_single_tick_bar_validates_correctly(self) -> None:
        """A bar created from a single tick should not fail Bar.__post_init__."""
        ticks = [make_tick(dt(2023, 1, 2, 9, 15, 0), price=17_500.0, volume=100)]
        bars  = aggregate_ticks(ticks, frequency="1min")
        # If Bar.__post_init__ raises, aggregate_ticks would raise too.
        assert isinstance(bars[0], Bar)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multiple buckets
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleBuckets:
    """Verify that ticks in different time buckets produce separate bars."""

    def test_two_minute_buckets_produce_two_bars(self) -> None:
        ticks = [
            # Bucket 1: 09:15
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=102.0, volume=20),
            # Bucket 2: 09:16
            make_tick(dt(2023, 1, 2, 9, 16, 0),  price=98.0,  volume=15),
            make_tick(dt(2023, 1, 2, 9, 16, 45), price=101.0, volume=25),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 2

    def test_two_buckets_ohlcv_values(self) -> None:
        """Verify OHLCV for each bucket independently."""
        ticks = [
            # Bucket 09:15: open=100, high=102, low=100, close=102, vol=30
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=102.0, volume=20),
            # Bucket 09:16: open=98, high=101, low=98, close=101, vol=40
            make_tick(dt(2023, 1, 2, 9, 16, 0),  price=98.0,  volume=15),
            make_tick(dt(2023, 1, 2, 9, 16, 45), price=101.0, volume=25),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")

        b1, b2 = bars[0], bars[1]

        assert b1.open   == pytest.approx(100.0)
        assert b1.high   == pytest.approx(102.0)
        assert b1.low    == pytest.approx(100.0)
        assert b1.close  == pytest.approx(102.0)
        assert b1.volume == pytest.approx(30.0)

        assert b2.open   == pytest.approx(98.0)
        assert b2.high   == pytest.approx(101.0)
        assert b2.low    == pytest.approx(98.0)
        assert b2.close  == pytest.approx(101.0)
        assert b2.volume == pytest.approx(40.0)

    def test_empty_bucket_between_two_occupied_buckets_is_dropped(self) -> None:
        """
        If minute 09:15 and 09:17 have ticks but 09:16 does not,
        the output should have only 2 bars — not 3.
        """
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0), price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 17, 0), price=105.0, volume=20),
            # 09:16 intentionally empty
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. Daily aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestDailyAggregation:
    """Tests for "1D" frequency — each day produces one bar."""

    def test_three_days_produce_three_bars(self) -> None:
        ticks = [
            # Day 1
            make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=500),
            make_tick(dt(2023, 1, 2, 15, 25), price=103.0, volume=700),
            # Day 2
            make_tick(dt(2023, 1, 3, 9, 15), price=104.0, volume=300),
            make_tick(dt(2023, 1, 3, 15, 25), price=101.0, volume=400),
            # Day 3
            make_tick(dt(2023, 1, 4, 9, 15), price=102.0, volume=600),
        ]
        bars = aggregate_ticks(ticks, frequency="1D")
        assert len(bars) == 3

    def test_daily_bar_ohlcv_day_one(self) -> None:
        """
        Day 1 ticks: 100.0 then 103.0
        Expected: open=100, high=103, low=100, close=103, volume=1200
        """
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15),  price=100.0, volume=500),
            make_tick(dt(2023, 1, 2, 15, 25), price=103.0, volume=700),
        ]
        bars = aggregate_ticks(ticks, frequency="1D")
        assert len(bars) == 1
        bar = bars[0]
        assert bar.open   == pytest.approx(100.0)
        assert bar.high   == pytest.approx(103.0)
        assert bar.low    == pytest.approx(100.0)
        assert bar.close  == pytest.approx(103.0)
        assert bar.volume == pytest.approx(1200.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 1-hour aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestHourlyAggregation:

    def test_two_hours_produce_two_bars(self) -> None:
        ticks = [
            make_tick(dt(2023, 1, 2, 9,  15), price=100.0, volume=100),
            make_tick(dt(2023, 1, 2, 9,  45), price=102.0, volume=150),
            make_tick(dt(2023, 1, 2, 10, 15), price=99.0,  volume=200),
            make_tick(dt(2023, 1, 2, 10, 45), price=104.0, volume=250),
        ]
        bars = aggregate_ticks(ticks, frequency="1H")
        assert len(bars) == 2

    def test_hourly_high_is_max_across_all_ticks_in_hour(self) -> None:
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=100),
            make_tick(dt(2023, 1, 2, 9, 30), price=120.0, volume=150),   # max
            make_tick(dt(2023, 1, 2, 9, 45), price=108.0, volume=200),
        ]
        bars = aggregate_ticks(ticks, frequency="1H")
        assert len(bars) == 1
        assert bars[0].high == pytest.approx(120.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Chronological ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestChronologicalOrder:

    def test_output_bars_are_sorted_oldest_first(self) -> None:
        """Output bars must always be oldest-first regardless of input order."""
        ticks = [
            make_tick(dt(2023, 1, 4, 9, 15), price=103.0, volume=50),   # day 3
            make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=50),   # day 1
            make_tick(dt(2023, 1, 3, 9, 15), price=101.0, volume=50),   # day 2
        ]
        bars = aggregate_ticks(ticks, frequency="1D")
        assert len(bars) == 3
        assert bars[0].timestamp < bars[1].timestamp < bars[2].timestamp

    def test_unsorted_input_ticks_produce_correct_ohlcv(self) -> None:
        """
        When ticks are not in time order, aggregation must still produce
        the correct open (first in time) and close (last in time).
        """
        # Deliberate reverse order
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 59), price=98.0,  volume=15),  # last in time
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=105.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),  # first in time
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 1
        assert bars[0].open  == pytest.approx(100.0)  # 09:15:00 is first in time
        assert bars[0].close == pytest.approx(98.0)   # 09:15:59 is last in time


# ─────────────────────────────────────────────────────────────────────────────
# 7. Volume accumulation across many ticks
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeAccumulation:

    def test_volume_is_exact_sum_of_all_ticks_in_bucket(self) -> None:
        volumes = [10, 20, 30, 40, 50]
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, i * 10), price=100.0 + i, volume=float(v))
            for i, v in enumerate(volumes)
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 1
        assert bars[0].volume == pytest.approx(sum(volumes))   # 150.0

    def test_volume_per_bucket_when_ticks_span_multiple_buckets(self) -> None:
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0), price=100.0, volume=100),
            make_tick(dt(2023, 1, 2, 9, 15, 30), price=101.0, volume=200),
            make_tick(dt(2023, 1, 2, 9, 16, 0), price=99.0,  volume=50),
            make_tick(dt(2023, 1, 2, 9, 16, 30), price=102.0, volume=75),
        ]
        bars = aggregate_ticks(ticks, frequency="1min")
        assert len(bars) == 2
        assert bars[0].volume == pytest.approx(300.0)   # 100 + 200
        assert bars[1].volume == pytest.approx(125.0)   # 50 + 75


# ─────────────────────────────────────────────────────────────────────────────
# 8. 5-minute aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestFiveMinuteAggregation:

    def test_ticks_within_same_five_minute_window_become_one_bar(self) -> None:
        """
        09:15:00, 09:17:30, 09:19:59 are all within the 09:15–09:19 bucket.
        09:20:00 starts a new bucket.
        """
        ticks = [
            make_tick(dt(2023, 1, 2, 9, 15, 0),  price=100.0, volume=10),
            make_tick(dt(2023, 1, 2, 9, 17, 30), price=103.0, volume=20),
            make_tick(dt(2023, 1, 2, 9, 19, 59), price=101.0, volume=15),
            make_tick(dt(2023, 1, 2, 9, 20, 0),  price=105.0, volume=25),  # new bucket
        ]
        bars = aggregate_ticks(ticks, frequency="5min")
        assert len(bars) == 2
        # First bar covers 09:15–09:19
        assert bars[0].open  == pytest.approx(100.0)
        assert bars[0].close == pytest.approx(101.0)
        # Second bar is just the 09:20 tick
        assert bars[1].open  == pytest.approx(105.0)
        assert bars[1].close == pytest.approx(105.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Error cases
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorCases:

    def test_empty_tick_list_raises_value_error(self) -> None:
        """aggregate_ticks([]) must raise ValueError, not return an empty list."""
        with pytest.raises(ValueError, match="empty"):
            aggregate_ticks([], frequency="1D")

    def test_unknown_frequency_raises_value_error(self) -> None:
        """A frequency string not in FREQUENCY_MAP must raise ValueError."""
        ticks = [make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=10)]
        with pytest.raises(ValueError, match="Unrecognised frequency"):
            aggregate_ticks(ticks, frequency="2W")

    def test_unknown_frequency_error_lists_supported_values(self) -> None:
        """The error message should name at least one supported frequency."""
        ticks = [make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=10)]
        with pytest.raises(ValueError, match="1min"):
            aggregate_ticks(ticks, frequency="INVALID")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Return type
# ─────────────────────────────────────────────────────────────────────────────

class TestReturnType:

    def test_returns_list_of_bar_objects(self) -> None:
        ticks = [make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=50)]
        bars  = aggregate_ticks(ticks, frequency="1min")
        assert isinstance(bars, list)
        assert all(isinstance(b, Bar) for b in bars)

    def test_bar_timestamp_is_datetime(self) -> None:
        from datetime import datetime as dt_type
        ticks = [make_tick(dt(2023, 1, 2, 9, 15), price=100.0, volume=50)]
        bars  = aggregate_ticks(ticks, frequency="1min")
        assert isinstance(bars[0].timestamp, dt_type)
