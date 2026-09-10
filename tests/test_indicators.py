"""
Unit tests for Technical Analysis Indicators with hand-calculated expected values.

All tests use synthetic, hand-calculated data to verify correctness against
first principles — NOT against another third-party library.
"""

from __future__ import annotations

from datetime import datetime
import math

import pytest

from vega.data.models import Bar
from vega.indicators.atr import atr, true_range
from vega.indicators.ema import ema
from vega.indicators.obv import obv
from vega.indicators.rsi import rsi


# ─────────────────────────────────────────────────────────────────────────────
# Helper to create valid Bar objects for tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_bar(
    t: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
) -> Bar:
    return Bar(
        timestamp=datetime(2023, 1, 1, 9, t),
        open=float(open_),
        high=float(high),
        low=float(low),
        close=float(close),
        volume=float(volume),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. EMA Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEMA:
    def test_ema_known_hand_calculated_values(self) -> None:
        """
        Hand calculation for period=3:
          values = [10.0, 11.0, 12.0, 13.0, 14.0]
          alpha = 2 / (3 + 1) = 0.5
          index 0: nan
          index 1: nan
          index 2: SMA(10, 11, 12) = 11.0
          index 3: 0.5 * 13.0 + 0.5 * 11.0 = 12.0
          index 4: 0.5 * 14.0 + 0.5 * 12.0 = 13.0
        """
        prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        result = ema(prices, period=3)

        assert len(result) == 5
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(11.0)
        assert result[3] == pytest.approx(12.0)
        assert result[4] == pytest.approx(13.0)

    def test_ema_non_linear_hand_calculated(self) -> None:
        """
        Hand calculation for period=2:
          values = [2.0, 4.0, 6.0, 8.0, 12.0]
          alpha = 2 / (2 + 1) = 2/3
          index 0: nan
          index 1: SMA(2, 4) = 3.0
          index 2: (2/3)*6 + (1/3)*3 = 4 + 1 = 5.0
          index 3: (2/3)*8 + (1/3)*5 = 16/3 + 5/3 = 21/3 = 7.0
          index 4: (2/3)*12 + (1/3)*7 = 24/3 + 7/3 = 31/3 = 10.333333333333334
        """
        prices = [2.0, 4.0, 6.0, 8.0, 12.0]
        result = ema(prices, period=2)

        assert math.isnan(result[0])
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(5.0)
        assert result[3] == pytest.approx(7.0)
        assert result[4] == pytest.approx(31.0 / 3.0)

    def test_ema_period_one(self) -> None:
        """When period=1, alpha=1.0 and EMA exactly equals the input values."""
        prices = [15.5, 17.2, 16.0, 22.4]
        result = ema(prices, period=1)
        assert result == [15.5, 17.2, 16.0, 22.4]

    def test_ema_constant_prices(self) -> None:
        """Constant prices must produce constant EMA equal to the price."""
        prices = [42.0] * 8
        result = ema(prices, period=3)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        for val in result[2:]:
            assert val == pytest.approx(42.0)

    def test_ema_deterministic_output(self) -> None:
        """EMA must produce identical results when called repeatedly."""
        prices = [10.0, 12.0, 14.0, 11.0, 13.0, 16.0]
        run1 = ema(prices, period=3)
        run2 = ema(prices, period=3)
        assert run1[2:] == run2[2:]

    def test_ema_insufficient_data_raises_value_error(self) -> None:
        """Passing fewer elements than period must raise ValueError."""
        with pytest.raises(ValueError, match="Insufficient data"):
            ema([10.0, 20.0], period=3)

    def test_ema_invalid_period_raises_value_error(self) -> None:
        """Non-positive period must raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            ema([10.0, 20.0], period=0)
        with pytest.raises(ValueError, match="positive integer"):
            ema([10.0, 20.0], period=-2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RSI Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRSI:
    def test_rsi_wilder_1978_hand_calculated_example(self) -> None:
        """
        J. Welles Wilder Jr. (1978) 14-period RSI textbook example.
        Prices: 16 values covering 15 changes.
        First 14 changes (t=1..14) yield initial avg_gain=0.23857, avg_loss=0.10.
        RS = 2.385714 -> RSI_14 = 70.46413502.
        At t=15, price changes by -0.28 (loss 0.28), yielding RSI_15 = 66.24961855.
        """
        wilder_prices = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
        ]
        result = rsi(wilder_prices, period=14)

        assert len(result) == 16
        # Warmup indices 0..13 must be NaN
        for i in range(14):
            assert math.isnan(result[i])

        # Hand-calculated Wilder textbook values
        assert result[14] == pytest.approx(70.464135, abs=1e-4)
        assert result[15] == pytest.approx(66.249619, abs=1e-4)

    def test_rsi_simple_hand_calculated_period_two(self) -> None:
        """
        Hand calculation for period=2:
          prices = [10.0, 12.0, 11.0, 13.0]
          changes:
            t=1: +2.0 (gain 2.0, loss 0.0)
            t=2: -1.0 (gain 0.0, loss 1.0)
          Seed at index 2:
            avg_gain = (2 + 0)/2 = 1.0
            avg_loss = (0 + 1)/2 = 0.5
            RS = 1.0 / 0.5 = 2.0
            RSI = 100 - (100 / 3) = 66.66666667
          At index 3:
            change: +2.0 (gain 2.0, loss 0.0)
            avg_gain = (1.0 * 1 + 2.0) / 2 = 1.5
            avg_loss = (0.5 * 1 + 0.0) / 2 = 0.25
            RS = 1.5 / 0.25 = 6.0
            RSI = 100 - (100 / 7) = 85.71428571
        """
        prices = [10.0, 12.0, 11.0, 13.0]
        result = rsi(prices, period=2)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(66.6666667, abs=1e-5)
        assert result[3] == pytest.approx(85.7142857, abs=1e-5)

    def test_rsi_strictly_rising_prices_zero_loss(self) -> None:
        """Monotonically rising prices produce 0 losses, hence RSI = 100.0."""
        prices = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        result = rsi(prices, period=3)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert math.isnan(result[2])
        assert result[3] == 100.0
        assert result[4] == 100.0
        assert result[5] == 100.0

    def test_rsi_strictly_falling_prices_zero_gain(self) -> None:
        """Monotonically falling prices produce 0 gains, hence RSI = 0.0."""
        prices = [50.0, 48.0, 46.0, 44.0, 42.0]
        result = rsi(prices, period=2)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0

    def test_rsi_flat_prices(self) -> None:
        """Flat prices produce 0 gain and 0 loss, documented as neutral RSI = 50.0."""
        prices = [100.0] * 6
        result = rsi(prices, period=3)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert math.isnan(result[2])
        assert result[3] == 50.0
        assert result[4] == 50.0
        assert result[5] == 50.0

    def test_rsi_insufficient_data_raises_value_error(self) -> None:
        """Need at least period + 1 prices; fewer must raise ValueError."""
        with pytest.raises(ValueError, match="Insufficient data"):
            rsi([10.0, 11.0], period=2)  # 2 values has only 1 change, needs 2 changes

    def test_rsi_invalid_period_raises_value_error(self) -> None:
        """Non-positive period must raise ValueError."""
        with pytest.raises(ValueError, match="positive integer"):
            rsi([10.0, 11.0, 12.0], period=0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ATR Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestATR:
    def test_true_range_hand_calculated(self) -> None:
        """
        Hand calculation for True Range:
          Bar 0: H=105, L=95, C=100  -> TR = 105 - 95 = 10.0
          Bar 1: H=110, L=102, C=108 -> TR = max(8, |110-100|, |102-100|) = max(8, 10, 2) = 10.0
          Bar 2: H=106, L=90, C=92   -> TR = max(16, |106-108|, |90-108|) = max(16, 2, 18) = 18.0
          Bar 3: H=98,  L=91, C=95   -> TR = max(7, |98-92|, |91-92|) = max(7, 6, 1) = 7.0
        """
        bars = [
            _make_bar(0, 100, 105, 95, 100),
            _make_bar(1, 103, 110, 102, 108),
            _make_bar(2, 105, 106, 90, 92),
            _make_bar(3, 93, 98, 91, 95),
        ]
        tr_vals = true_range(bars)
        assert tr_vals == [10.0, 10.0, 18.0, 7.0]

    def test_atr_hand_calculated_values(self) -> None:
        """
        Using the TR values above: [10.0, 10.0, 18.0, 7.0]
        Period = 3:
          index 0: nan
          index 1: nan
          index 2: SMA(10, 10, 18) = 38 / 3 = 12.66666667
          index 3: Wilder smoothing = (12.66666667 * 2 + 7.0) / 3 = 32.33333333 / 3 = 10.77777778
        """
        bars = [
            _make_bar(0, 100, 105, 95, 100),
            _make_bar(1, 103, 110, 102, 108),
            _make_bar(2, 105, 106, 90, 92),
            _make_bar(3, 93, 98, 91, 95),
        ]
        result = atr(bars, period=3)

        assert len(result) == 4
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(38.0 / 3.0)
        assert result[3] == pytest.approx(10.77777778, abs=1e-6)

    def test_atr_first_bar_handling(self) -> None:
        """Bar 0 has no previous close, so TR_0 must strictly equal High_0 - Low_0."""
        bars = [_make_bar(0, 100, 112.5, 98.0, 105.0)]
        assert true_range(bars) == [14.5]

    def test_atr_constant_bars(self) -> None:
        """Bars with High == Low == Close have 0 volatility; ATR must be 0.0."""
        bars = [_make_bar(t, 100, 100, 100, 100) for t in range(5)]
        result = atr(bars, period=3)

        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert result[4] == 0.0

    def test_atr_insufficient_data_raises_value_error(self) -> None:
        """Fewer bars than period must raise ValueError."""
        bars = [_make_bar(t, 100, 105, 95, 100) for t in range(2)]
        with pytest.raises(ValueError, match="Insufficient data"):
            atr(bars, period=3)

    def test_atr_invalid_period_raises_value_error(self) -> None:
        """Non-positive period must raise ValueError."""
        bars = [_make_bar(t, 100, 105, 95, 100) for t in range(5)]
        with pytest.raises(ValueError, match="positive integer"):
            atr(bars, period=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. OBV Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestOBV:
    def test_obv_hand_calculated_mixed_sequence(self) -> None:
        """
        Hand calculation:
          Bar 0: Close 100, Vol 1000 -> Initial OBV = 1000.0
          Bar 1: Close 105, Vol 500  -> Close > Prev -> OBV = 1000 + 500 = 1500.0
          Bar 2: Close 102, Vol 300  -> Close < Prev -> OBV = 1500 - 300 = 1200.0
          Bar 3: Close 102, Vol 800  -> Close == Prev -> OBV = 1200.0 (unchanged)
          Bar 4: Close 108, Vol 400  -> Close > Prev -> OBV = 1200 + 400 = 1600.0
        """
        bars = [
            _make_bar(0, 100, 102, 98, 100, volume=1000),
            _make_bar(1, 101, 106, 101, 105, volume=500),
            _make_bar(2, 105, 105, 101, 102, volume=300),
            _make_bar(3, 102, 103, 101, 102, volume=800),
            _make_bar(4, 103, 109, 102, 108, volume=400),
        ]
        result = obv(bars)
        assert result == [1000.0, 1500.0, 1200.0, 1200.0, 1600.0]

    def test_obv_strictly_increasing_prices(self) -> None:
        """Volume should accumulate on every bar."""
        bars = [
            _make_bar(0, 10, 11, 9, 10, volume=100),
            _make_bar(1, 11, 13, 11, 12, volume=200),
            _make_bar(2, 13, 15, 13, 14, volume=300),
        ]
        assert obv(bars) == [100.0, 300.0, 600.0]

    def test_obv_strictly_decreasing_prices(self) -> None:
        """Volume should decrement on every bar after bar 0."""
        bars = [
            _make_bar(0, 30, 31, 29, 30, volume=500),
            _make_bar(1, 28, 29, 27, 28, volume=200),
            _make_bar(2, 25, 26, 24, 25, volume=150),
        ]
        assert obv(bars) == [500.0, 300.0, 150.0]

    def test_obv_unchanged_prices(self) -> None:
        """Volume is ignored when close equals previous close."""
        bars = [
            _make_bar(0, 50, 51, 49, 50, volume=100),
            _make_bar(1, 50, 52, 48, 50, volume=250),
            _make_bar(2, 50, 53, 47, 50, volume=350),
        ]
        assert obv(bars) == [100.0, 100.0, 100.0]

    def test_obv_zero_volume(self) -> None:
        """Zero volume bars must not alter OBV."""
        bars = [
            _make_bar(0, 50, 51, 49, 50, volume=100),
            _make_bar(1, 52, 55, 51, 54, volume=0),
            _make_bar(2, 53, 54, 49, 51, volume=0),
        ]
        assert obv(bars) == [100.0, 100.0, 100.0]

    def test_obv_insufficient_data_raises_value_error(self) -> None:
        """Empty bar list must raise ValueError."""
        with pytest.raises(ValueError, match="empty list"):
            obv([])

    def test_obv_custom_initial_value(self) -> None:
        """Explicit initial value can be supplied (e.g. starting at 0.0)."""
        bars = [
            _make_bar(0, 10, 11, 9, 10, volume=100),
            _make_bar(1, 11, 13, 11, 12, volume=200),
        ]
        assert obv(bars, initial=0.0) == [0.0, 200.0]
