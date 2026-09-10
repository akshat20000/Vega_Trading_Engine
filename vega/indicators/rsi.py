"""
Relative Strength Index (RSI) implementation from first principles.

What RSI measures:
    The Relative Strength Index (developed by J. Welles Wilder Jr. in 1978) is a
    momentum oscillator that measures the speed and magnitude of recent price
    changes on a scale of 0 to 100. It is traditionally used to identify
    overbought (>70) and oversold (<30) conditions.

Mathematical Steps:
    1. Price Changes:
       For each consecutive pair of prices:
           change_t = Price_t - Price_(t-1)

    2. Gains and Losses:
           gain_t = max(change_t, 0.0)
           loss_t = max(-change_t, 0.0)
       Both gain_t and loss_t are non-negative numbers (>= 0.0).

    3. Initial Average Gain and Loss (Wilder Seed):
       Over the first `period` changes (from t = 1 to t = period):
           avg_gain_period = sum(gain_1 .. gain_period) / period
           avg_loss_period = sum(loss_1 .. loss_period) / period

    4. Wilder Smoothing for subsequent bars (t > period):
           avg_gain_t = (avg_gain_(t-1) * (period - 1) + gain_t) / period
           avg_loss_t = (avg_loss_(t-1) * (period - 1) + loss_t) / period

    5. Relative Strength (RS) and RSI:
           RS = avg_gain / avg_loss
           RSI = 100.0 - (100.0 / (1.0 + RS))
               = 100.0 * (avg_gain / (avg_gain + avg_loss))

Edge Cases Handled Explicitly:
    - Insufficient Data:
      Calculating `period` price changes requires at least `period + 1` prices.
      If len(values) < period + 1, a ValueError is raised.
      Indices 0 to period - 1 are set to float('nan') because the first `period`
      changes have not yet accumulated.
    - Zero Losses (avg_loss == 0.0 and avg_gain > 0.0):
      Prices only went up. RS is effectively infinite. RSI = 100.0.
    - Zero Gains (avg_gain == 0.0 and avg_loss > 0.0):
      Prices only went down. RS is 0.0. RSI = 0.0.
    - Flat Prices (avg_gain == 0.0 and avg_loss == 0.0):
      Prices did not move at all. Neither bulls nor bears have momentum.
      RSI is set to 50.0 (neutral midpoint).
"""

from __future__ import annotations

import math
from typing import Sequence


def _calculate_rsi_value(avg_gain: float, avg_loss: float) -> float:
    """
    Calculate RSI from average gain and average loss with explicit edge cases.

    - Flat prices (gain == 0, loss == 0): returns 50.0 (neutral).
    - Monotonically increasing (loss == 0, gain > 0): returns 100.0 (max bullish).
    - Monotonically decreasing (gain == 0, loss > 0): returns 0.0 (max bearish).
    - Standard case: returns 100.0 - (100.0 / (1.0 + RS)).
    """
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            return 50.0  # Flat prices: perfectly neutral
        return 100.0    # All gains, zero loss
    if avg_gain == 0.0:
        return 0.0      # All losses, zero gain

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi(values: Sequence[float], period: int = 14) -> list[float]:
    """
    Calculate Relative Strength Index (RSI) for a sequence of values.

    Args:
        values: Sequence of numeric price values (e.g. closing prices).
        period: Lookback period for RSI changes (default: 14, must be >= 1).

    Returns:
        A list of float values of identical length to `values`.
        Indices 0 to period - 1 are float('nan') (warmup period).
        Index period is the first computed RSI value (from the first `period` changes).
        Subsequent indices are smoothed using Wilder's technique.

    Raises:
        ValueError: If `period < 1`.
        ValueError: If `len(values) < period + 1` (insufficient data).

    Example:
        >>> prices = [10.0, 11.0, 12.0, 11.0, 13.0]
        >>> rsi(prices, period=3)
        [nan, nan, nan, 66.66666666666667, 80.0]
    """
    if period < 1:
        raise ValueError(f"period must be a positive integer, got {period}")

    n = len(values)
    if n < period + 1:
        raise ValueError(
            f"Insufficient data: period={period} requires at least {period + 1} values "
            f"to form {period} price changes, but got {n}."
        )

    result: list[float] = [math.nan] * n

    # Step 1: Calculate initial average gain and loss over the first `period` changes
    sum_gain = 0.0
    sum_loss = 0.0
    for i in range(1, period + 1):
        diff = float(values[i]) - float(values[i - 1])
        if diff > 0.0:
            sum_gain += diff
        elif diff < 0.0:
            sum_loss += -diff

    avg_gain = sum_gain / float(period)
    avg_loss = sum_loss / float(period)

    # First RSI value is at index `period`
    result[period] = _calculate_rsi_value(avg_gain, avg_loss)

    # Step 2: Wilder smoothing for subsequent bars from index (period + 1) to n - 1
    for t in range(period + 1, n):
        diff = float(values[t]) - float(values[t - 1])
        gain = diff if diff > 0.0 else 0.0
        loss = -diff if diff < 0.0 else 0.0

        # Wilder smoothing: (prev_avg * (period - 1) + current) / period
        avg_gain = (avg_gain * (period - 1.0) + gain) / float(period)
        avg_loss = (avg_loss * (period - 1.0) + loss) / float(period)

        result[t] = _calculate_rsi_value(avg_gain, avg_loss)

    return result
