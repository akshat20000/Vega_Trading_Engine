"""
Exponential Moving Average (EMA) implementation from first principles.

What EMA measures:
    The Exponential Moving Average is a trend-following momentum indicator that
    places a greater weight and significance on the most recent data points.
    Unlike a Simple Moving Average (SMA) where all past prices in the window have
    equal weight, EMA reacts faster to recent price changes.

Mathematical Definition:
    For a given period N >= 1:
        alpha = 2.0 / (period + 1)

    Recursive formula:
        EMA_t = alpha * Price_t + (1 - alpha) * EMA_(t-1)

Initialization Method:
    To start the recursive calculation, an initial seed value is needed.
    The industry-standard convention (Murphy, Technical Analysis of the Financial
    Markets; Achelis; TA-Lib):
      - The first valid EMA value is placed at index (period - 1), initialized
        as the Simple Moving Average (SMA) of the first `period` prices:
            EMA_(period-1) = (Price_0 + Price_1 + ... + Price_(period-1)) / period
      - Prior indices [0 .. period-2] are set to float('nan') because insufficient
        data exists to form a full period window.
      - For period == 1, alpha = 2 / (1 + 1) = 1.0, so EMA_t = Price_t for all t.

Why not call pandas.ewm()?
    Calling pandas.ewm() conceals the underlying recurrence relation and can
    introduce subtle discrepancies depending on `adjust=True` vs `adjust=False`
    and `min_periods` defaults. Writing from first principles ensures absolute
    transparency, predictable memory and performance, and zero dependency on
    pandas-specific behaviors.
"""

from __future__ import annotations

import math
from typing import Sequence


def ema(values: Sequence[float], period: int) -> list[float]:
    """
    Calculate Exponential Moving Average (EMA) for a sequence of values.

    Args:
        values: Sequence of numeric price values (e.g. closing prices).
        period: Number of periods for the moving average (must be >= 1).

    Returns:
        A list of float values of identical length to `values`.
        Indices 0 to period - 2 are float('nan') (warmup period).
        Index period - 1 is initialized as the SMA of the first `period` items.
        Subsequent indices are recursively computed using alpha = 2 / (period + 1).

    Raises:
        ValueError: If `period < 1`.
        ValueError: If `len(values) < period` (insufficient data).

    Example:
        >>> prices = [10.0, 11.0, 12.0, 13.0, 14.0]
        >>> ema(prices, period=3)
        [nan, nan, 11.0, 12.0, 13.0]
    """
    if period < 1:
        raise ValueError(f"period must be a positive integer, got {period}")

    n = len(values)
    if n < period:
        raise ValueError(
            f"Insufficient data: period={period} requires at least {period} data points, "
            f"but got {n}."
        )

    # Special case: period = 1 -> alpha = 1.0, EMA_t = Price_t
    if period == 1:
        return [float(v) for v in values]

    result: list[float] = [math.nan] * n

    # Step 1: Initialize seed value at index (period - 1) using SMA of first `period` bars
    initial_sma = sum(values[:period]) / float(period)
    result[period - 1] = float(initial_sma)

    # Step 2: Recursive calculation from index `period` onwards
    alpha = 2.0 / (period + 1.0)
    one_minus_alpha = 1.0 - alpha

    current_ema = float(initial_sma)
    for t in range(period, n):
        current_ema = (alpha * float(values[t])) + (one_minus_alpha * current_ema)
        result[t] = current_ema

    return result
