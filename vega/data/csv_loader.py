"""
CSV loader for OHLCV bar data.

Reads a CSV file with columns: timestamp, open, high, low, close, volume.
Column order does not matter — the header row is used for lookup.
Returns a list of Bar objects sorted chronologically (oldest first).

Typical usage:
    from vega.data.csv_loader import load_bars
    bars = load_bars("data/market/NIFTY50_daily.csv")
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from vega.data.models import Bar


# Every CSV must have these columns (checked case-insensitively).
REQUIRED_COLUMNS: set[str] = {"timestamp", "open", "high", "low", "close", "volume"}

# Timestamp formats tried in order. First match wins.
TIMESTAMP_FORMATS: list[str] = [
    "%Y-%m-%d %H:%M:%S",   # 2023-01-02 09:15:00
    "%Y-%m-%d",             # 2023-01-02
    "%d-%m-%Y",             # 02-01-2023
    "%d/%m/%Y",             # 02/01/2023
]


def _parse_timestamp(value: str) -> datetime:
    """
    Try each known timestamp format until one parses successfully.

    Args:
        value: Raw timestamp string from the CSV.

    Returns:
        A datetime object.

    Raises:
        ValueError: If no format matches.
    """
    stripped = value.strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(stripped, fmt)
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse timestamp '{value}'. "
        f"Supported formats: {TIMESTAMP_FORMATS}"
    )


def load_bars(filepath: str | Path) -> list[Bar]:
    """
    Load OHLCV data from a CSV file and return a sorted list of Bar objects.

    The CSV must have a header row with at minimum these columns
    (column names are case-insensitive, extra columns are ignored):
        timestamp, open, high, low, close, volume

    Bars are automatically sorted chronologically.
    Duplicate timestamps are rejected with a ValueError.

    Args:
        filepath: Path to the CSV file (relative or absolute).

    Returns:
        A list of Bar objects sorted from oldest to newest.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing, data cannot be parsed,
                    or duplicate timestamps are found.

    Example:
        bars = load_bars("data/market/NIFTY50_daily.csv")
        print(bars[0].close)  # 17859.45
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    bars: list[Bar] = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {filepath}")

        # Normalise column names to lowercase for case-insensitive matching.
        actual_columns = {col.strip().lower() for col in reader.fieldnames}
        missing = REQUIRED_COLUMNS - actual_columns
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {missing}\n"
                f"Found columns: {actual_columns}\n"
                f"File: {filepath}"
            )

        for line_num, row in enumerate(reader, start=2):
            # Normalise all keys to lowercase and strip whitespace.
            clean_row = {k.strip().lower(): v.strip() for k, v in row.items()}

            try:
                bar = Bar(
                    timestamp = _parse_timestamp(clean_row["timestamp"]),
                    open      = float(clean_row["open"]),
                    high      = float(clean_row["high"]),
                    low       = float(clean_row["low"]),
                    close     = float(clean_row["close"]),
                    volume    = float(clean_row["volume"]),
                )
                bars.append(bar)
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Error reading {filepath} on line {line_num}: {exc}"
                ) from exc

    if not bars:
        raise ValueError(f"CSV file contains no data rows: {filepath}")

    # Sort chronologically. For already-sorted files this is a no-op (O(n)).
    bars.sort(key=lambda b: b.timestamp)

    # Reject duplicate timestamps — each bar must be unique in time.
    for i in range(1, len(bars)):
        if bars[i].timestamp == bars[i - 1].timestamp:
            raise ValueError(
                f"Duplicate timestamp found in {filepath}: {bars[i].timestamp}. "
                "Each bar must have a unique timestamp."
            )

    return bars
