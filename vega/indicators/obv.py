"""
On-Balance Volume (OBV) implementation from first principles.

What OBV measures:
    On-Balance Volume (developed by Joseph Granville in 1963) is a cumulative
    momentum indicator that relates volume to price change. The theory behind
    OBV is that volume precedes price movement: when volume increases sharply
    without a significant price change, price will eventually catch up.

Mathematical Rules:
    For a sequence of bars:
      - At bar 0 (first bar): No previous close exists.
        Initial OBV value convention: OBV_0 = float(bars[0].volume).
        (This matches the standard convention of Granville 1963, TA-Lib, and pandas-ta).

      - For each subsequent bar t >= 1:
        If Close_t > Close_(t-1):
            OBV_t = OBV_(t-1) + Volume_t
        Else if Close_t < Close_(t-1):
            OBV_t = OBV_(t-1) - Volume_t
        Else (Close_t == Close_(t-1)):
            OBV_t = OBV_(t-1)   # Volume is ignored when price is unchanged
"""

from __future__ import annotations

from typing import Sequence

from vega.data.models import Bar


def obv(bars: Sequence[Bar], initial: float | None = None) -> list[float]:
    """
    Calculate On-Balance Volume (OBV) for a sequence of Bar objects.

    Args:
        bars: Sequence of Bar objects.
        initial: Initial OBV value for bar 0. If None, defaults to `float(bars[0].volume)`,
                 which is the standard convention of Granville, TA-Lib, and pandas-ta.

    Returns:
        A list of float values representing cumulative OBV, identical in length to `bars`.

    Raises:
        ValueError: If `bars` is empty.

    Example:
        >>> obv_values = obv(bars)
    """
    if not bars:
        raise ValueError("Cannot calculate OBV on an empty list of bars.")

    n = len(bars)
    result: list[float] = [0.0] * n

    # Step 1: Initialize first bar
    start_val = float(bars[0].volume) if initial is None else float(initial)
    result[0] = start_val

    # Step 2: Accumulate based on close price comparison
    current_obv = start_val
    for t in range(1, n):
        curr_close = float(bars[t].close)
        prev_close = float(bars[t - 1].close)
        volume = float(bars[t].volume)

        if curr_close > prev_close:
            current_obv += volume
        elif curr_close < prev_close:
            current_obv -= volume
        # else curr_close == prev_close: current_obv remains unchanged

        result[t] = current_obv

    return result
