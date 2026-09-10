"""
Parquet storage for OHLCV Bar data.

Apache Parquet is a columnar binary format built for efficient storage and
retrieval of time-series data. It handles timestamps natively and compresses
dramatically better than CSV.

Why Parquet (not CSV for long-term storage)?
    - Columnar layout: reading only 'close' prices does not read other columns.
    - Native timestamp storage: no string parsing on every load.
    - ~5-10x smaller than equivalent CSV for OHLCV data.
    - Reads via pyarrow are substantially faster than csv.reader.

Parquet files are the persisted dataset.
DuckDB (see duckdb_store.py) queries them directly without duplicating data.

Float64 round-trip accuracy:
    Python floats are 64-bit IEEE 754. Parquet stores them as float64.
    Reading back produces the bit-identical float64 value — exact equality
    (==) holds for any value representable in float64, which covers all
    typical OHLCV prices. No precision is lost in the round-trip.

Typical usage:
    from vega.data.parquet_store import ParquetStore
    store = ParquetStore()
    store.save(bars, "data/market/NIFTY50.parquet")
    loaded_bars = store.load("data/market/NIFTY50.parquet")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from vega.data.models import Bar


# ─────────────────────────────────────────────────────────────────────────────
# Public helper — also imported by duckdb_store.py
# ─────────────────────────────────────────────────────────────────────────────

def to_python_datetime(value: object) -> datetime:
    """
    Convert a timestamp value to a timezone-naive Python datetime.

    pyarrow / DuckDB may return pd.Timestamp or datetime objects
    when reading Parquet files. This helper normalises both into a
    plain Python datetime so Bar objects remain free of pandas types.

    Timezone info is stripped if present — all Bar timestamps in this
    project are timezone-naive (IST is applied at the application level
    when needed, not stored in the timestamp field).

    Args:
        value: A pd.Timestamp, datetime, or any value pd.Timestamp accepts.

    Returns:
        A timezone-naive Python datetime object.
    """
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        # Fallback: let pandas parse it (handles numpy datetime64, etc.)
        dt = pd.Timestamp(value).to_pydatetime()

    # Strip timezone — Bar timestamps are always timezone-naive.
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return dt


# ─────────────────────────────────────────────────────────────────────────────
# ParquetStore
# ─────────────────────────────────────────────────────────────────────────────

class ParquetStore:
    """
    Saves and loads list[Bar] as Apache Parquet files.

    Each symbol gets its own file (e.g. "data/market/NIFTY50.parquet").
    The interface is intentionally simple: a list[Bar] in, a list[Bar] out.

    This class is responsible ONLY for storage and retrieval of market data.
    It contains no strategy logic, indicators, risk rules, or broker code.

    Example:
        store = ParquetStore()

        # Save
        store.save(bars, "data/market/NIFTY50.parquet")

        # Load (always returned chronologically sorted)
        bars = store.load("data/market/NIFTY50.parquet")
    """

    def save(self, bars: list[Bar], filepath: str | Path) -> None:
        """
        Write a list of Bar objects to a Parquet file.

        - The parent directory is created automatically if it does not exist.
        - Bars are written in the order provided (not sorted here).
        - Existing files are overwritten without warning.
        - Explicit float64 dtype is enforced for all OHLCV columns so that
          the round-trip equality guarantee holds regardless of Python version.

        Args:
            bars:     A non-empty list of Bar objects to persist.
            filepath: Destination path, e.g. "data/market/NIFTY50.parquet".
                      The .parquet extension is not enforced but is recommended.

        Raises:
            ValueError: If bars is empty.

        Example:
            store.save(bars, "data/market/NIFTY50.parquet")
        """
        if not bars:
            raise ValueError(
                "Cannot save an empty list of bars to Parquet. "
                "Provide at least one Bar object."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build DataFrame with explicit dtypes.
        # Using pd.array(..., dtype="float64") prevents silent upcasting to
        # object dtype if any value happens to be None or mixed-type.
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([b.timestamp for b in bars]),
            "open":      pd.array([b.open   for b in bars], dtype="float64"),
            "high":      pd.array([b.high   for b in bars], dtype="float64"),
            "low":       pd.array([b.low    for b in bars], dtype="float64"),
            "close":     pd.array([b.close  for b in bars], dtype="float64"),
            "volume":    pd.array([b.volume for b in bars], dtype="float64"),
        })

        # index=False: the row index (0, 1, 2, ...) is not written to the file.
        df.to_parquet(filepath, engine="pyarrow", index=False)

    def load(self, filepath: str | Path) -> list[Bar]:
        """
        Read Bar objects from a Parquet file.

        - Returns bars sorted chronologically (oldest first).
        - Bar.__post_init__ runs for every row, so any Parquet file that
          contains inconsistent OHLCV values (high < low, etc.) is rejected
          immediately with a clear ValueError.

        Args:
            filepath: Path to the .parquet file to read.

        Returns:
            A list of Bar objects sorted by timestamp (oldest first).

        Raises:
            FileNotFoundError: If the Parquet file does not exist.
            ValueError:        If required columns are missing in the file.

        Example:
            bars = store.load("data/market/NIFTY50.parquet")
            print(bars[0].close)
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {filepath}\n"
                "Call ParquetStore.save() first to create it."
            )

        df = pd.read_parquet(filepath, engine="pyarrow")

        # Validate that all expected columns are present.
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"Parquet file is missing required columns: {missing}\n"
                f"Found columns: {list(df.columns)}\n"
                f"File: {filepath}"
            )

        bars: list[Bar] = []
        for _, row in df.iterrows():
            bars.append(
                Bar(
                    timestamp = to_python_datetime(row["timestamp"]),
                    open      = float(row["open"]),
                    high      = float(row["high"]),
                    low       = float(row["low"]),
                    close     = float(row["close"]),
                    volume    = float(row["volume"]),
                )
            )

        # Always return oldest-first regardless of storage order.
        bars.sort(key=lambda b: b.timestamp)
        return bars
