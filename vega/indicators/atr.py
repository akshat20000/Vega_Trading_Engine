"""
Average True Range (ATR) implementation from first principles.

What ATR measures:
    The Average True Range (developed by J. Welles Wilder Jr. in 1978) is a
    volatility indicator that measures the degree of price volatility.
    Unlike standard deviation, True Range incorporates overnight price gaps
    between the previous close and the current high or low.

Mathematical Definition:
    1. True Range (TR):
       For the first bar (t = 0), there is no previous close. As explicitly
       defined by Wilder, the True Range of the first bar is simply:
           TR_0 = High_0 - Low_0

       For all subsequent bars (t >= 1):
           TR_t = max(
               High_t - Low_t,
               abs(High_t - Close_(t-1)),
               abs(Low_t - Close_(t-1))
           )

    2. Average True Range (ATR):
       - Initial ATR at index (period - 1) is the simple average (SMA) of the
         first `period` True Ranges:
             ATR_(period-1) = sum(TR_0 .. TR_(period-1)) / period
       - Indices 0 to period - 2 are float('nan') (warmup period).
       - Subsequent bars (t >= period) use Wilder's recursive smoothing:
             ATR_t = (ATR_(t-1) * (period - 1) + TR_t) / period
"""

from __future__ import annotations

import math
from typing import Sequence

from vega.data.models import Bar


def true_range(bars: Sequence[Bar]) -> list[float]:
    """
    Calculate True Range (TR) for each bar in a sequence of Bar objects.

    For bar 0, where no previous close exists, TR is High_0 - Low_0.
    For bar t >= 1, TR is max(High_t - Low_t, |High_t - Close_(t-1)|, |Low_t - Close_(t-1)|).

    Args:
        bars: Sequence of Bar objects.

    Returns:
        A list of float True Range values, of identical length to `bars`.

    Raises:
        ValueError: If `bars` is empty.
    """
    if not bars:
        raise ValueError("Cannot calculate True Range on an empty list of bars.")

    n = len(bars)
    tr_values: list[float] = [0.0] * n

    # First bar: no previous close available, TR is high - low
    tr_values[0] = float(bars[0].high - bars[0].low)

    # Subsequent bars: incorporate previous close
    for t in range(1, n):
        prev_close = float(bars[t - 1].close)
        curr_high = float(bars[t].high)
        curr_low = float(bars[t].low)

        h_minus_l = curr_high - curr_low
        h_minus_pc = abs(curr_high - prev_close)
        l_minus_pc = abs(curr_low - prev_close)

        tr_values[t] = max(h_minus_l, h_minus_pc, l_minus_pc)

    return tr_values


def atr(bars: Sequence[Bar], period: int = 14) -> list[float]:
    """
    Calculate Average True Range (ATR) for a sequence of Bar objects.

    Args:
        bars: Sequence of Bar objects.
        period: Lookback period for ATR smoothing (default: 14, must be >= 1).

    Returns:
        A list of float values of identical length to `bars`.
        Indices 0 to period - 2 are float('nan') (warmup period).
        Index period - 1 is the initial SMA of the first `period` True Ranges.
        Subsequent indices are smoothed using Wilder's technique:
            ATR_t = (ATR_(t-1) * (period - 1) + TR_t) / period

    Raises:
        ValueError: If `period < 1`.
        ValueError: If `len(bars) < period` (insufficient data).

    Example:
        >>> atr_values = atr(bars, period=14)
    """
    if period < 1:
        raise ValueError(f"period must be a positive integer, got {period}")

    n = len(bars)
    if n < period:
        raise ValueError(
            f"Insufficient data: period={period} requires at least {period} bars, "
            f"but got {n}."
        )

    # Calculate True Range for all bars
    tr_values = true_range(bars)

    result: list[float] = [math.nan] * n

    # Step 1: Initial seed at index (period - 1) using SMA of first `period` True Ranges
    initial_atr = sum(tr_values[:period]) / float(period)
    result[period - 1] = float(initial_atr)

    # Step 2: Wilder smoothing for subsequent bars
    current_atr = float(initial_atr)
    for t in range(period, n):
        current_atr = (current_atr * (period - 1.0) + tr_values[t]) / float(period)
        result[t] = current_atr

    return result
