"""
Tick-to-Bar aggregator.

Converts a list of Tick objects into OHLCV Bar objects by grouping
ticks into fixed time buckets and computing open/high/low/close/volume
for each bucket.

How aggregation works for each time bucket:
    open   = price of the FIRST tick in the bucket
    high   = HIGHEST price seen across all ticks in the bucket
    low    = LOWEST  price seen across all ticks in the bucket
    close  = price of the LAST  tick in the bucket
    volume = SUM of all volumes in the bucket

Empty buckets (no ticks in that period) are dropped automatically.
This is standard behaviour in financial data processing.

Supported frequency strings:
    "1min"  — 1-minute bars
    "5min"  — 5-minute bars
    "15min" — 15-minute bars
    "1H"    — 1-hour bars
    "1D"    — daily bars

Typical usage:
    from vega.data.tick_aggregator import aggregate_ticks
    bars = aggregate_ticks(ticks, frequency="1min")
"""

from __future__ import annotations

import pandas as pd

from vega.data.models import Bar, Tick


# Maps our friendly frequency strings to pandas resample aliases.
# pandas 2.x uses "min" for minutes and "h" for hours.
FREQUENCY_MAP: dict[str, str] = {
    "1min":  "1min",
    "5min":  "5min",
    "15min": "15min",
    "1H":    "1h",
    "1D":    "1D",
}


def aggregate_ticks(ticks: list[Tick], frequency: str = "1D") -> list[Bar]:
    """
    Aggregate a list of Tick objects into OHLCV Bar objects.

    The resulting bars use the bucket's START time as the bar timestamp
    (e.g. the 09:15 bar covers 09:15:00 – 09:15:59 for "1min" frequency).

    Args:
        ticks:     A non-empty list of Tick objects.
        frequency: How to bucket the ticks. Must be one of:
                   "1min", "5min", "15min", "1H", "1D".

    Returns:
        A list of Bar objects sorted chronologically (oldest first).
        Empty time buckets are not included in the output.

    Raises:
        ValueError: If ticks is empty or frequency is not recognised.

    Example:
        ticks = [
            Tick(datetime(2023,1,2,9,15,0),  price=100.0, volume=10),
            Tick(datetime(2023,1,2,9,15,30), price=105.0, volume=20),
            Tick(datetime(2023,1,2,9,16,0),  price=98.0,  volume=15),
        ]
        bars = aggregate_ticks(ticks, "1min")
        # bars[0]: open=100, high=105, low=100, close=105, volume=30
        # bars[1]: open=98,  high=98,  low=98,  close=98,  volume=15
    """
    if not ticks:
        raise ValueError(
            "aggregate_ticks() received an empty list. "
            "Provide at least one Tick to aggregate."
        )

    if frequency not in FREQUENCY_MAP:
        raise ValueError(
            f"Unrecognised frequency '{frequency}'. "
            f"Supported values: {list(FREQUENCY_MAP.keys())}"
        )

    pandas_freq = FREQUENCY_MAP[frequency]

    # ── Step 1: Build a DataFrame from the ticks ──────────────────────────────
    df = pd.DataFrame(
        {
            "timestamp": [t.timestamp for t in ticks],
            "price":     [t.price     for t in ticks],
            "volume":    [t.volume    for t in ticks],
        }
    )

    # Sort by time (handles unsorted input gracefully).
    df = df.sort_values("timestamp").set_index("timestamp")

    # ── Step 2: Resample into OHLCV ───────────────────────────────────────────
    # pandas ohlc() on a price series produces: open, high, low, close
    price_resampled  = df["price"].resample(pandas_freq)
    volume_resampled = df["volume"].resample(pandas_freq)

    ohlc = price_resampled.ohlc()      # DataFrame: open, high, low, close
    vol  = volume_resampled.sum()      # Series: total volume per bucket

    # Drop rows where no ticks occurred (ohlc produces NaN for empty buckets).
    ohlc = ohlc.dropna()

    # ── Step 3: Convert back to list[Bar] ─────────────────────────────────────
    bars: list[Bar] = []
    for ts, row in ohlc.iterrows():
        bar = Bar(
            timestamp = ts.to_pydatetime(),
            open      = float(row["open"]),
            high      = float(row["high"]),
            low       = float(row["low"]),
            close     = float(row["close"]),
            volume    = float(vol[ts]),
        )
        bars.append(bar)

    # Output is already sorted because we sorted the DataFrame by timestamp
    # and pandas preserves that order through resample.
    return bars
