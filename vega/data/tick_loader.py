"""
CSV loader for raw tick data.

Reads a CSV file with columns: timestamp, price, volume.
Returns a list of Tick objects sorted chronologically (oldest first).

Typical usage:
    from vega.data.tick_loader import load_ticks
    ticks = load_ticks("data/market/NIFTY50_ticks.csv")
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from vega.data.models import Tick


REQUIRED_COLUMNS: set[str] = {"timestamp", "price", "volume"}

# Tick timestamps always include time (milliseconds optional).
TIMESTAMP_FORMATS: list[str] = [
    "%Y-%m-%d %H:%M:%S.%f",   # 2023-01-02 09:15:00.123456
    "%Y-%m-%d %H:%M:%S",      # 2023-01-02 09:15:00
    "%Y-%m-%d %H:%M",         # 2023-01-02 09:15
]


def _parse_timestamp(value: str) -> datetime:
    """
    Try each tick timestamp format until one parses successfully.

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
        f"Cannot parse tick timestamp '{value}'. "
        f"Supported formats: {TIMESTAMP_FORMATS}"
    )


def load_ticks(filepath: str | Path) -> list[Tick]:
    """
    Load raw tick data from a CSV file.

    The CSV must have a header row with these columns
    (case-insensitive, extra columns are ignored):
        timestamp, price, volume

    Ticks are automatically sorted chronologically.

    Args:
        filepath: Path to the tick CSV file.

    Returns:
        A list of Tick objects sorted from oldest to newest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data cannot be parsed.

    Example:
        ticks = load_ticks("data/market/NIFTY50_ticks.csv")
        print(ticks[0].price)  # 17807.00
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Tick CSV file not found: {filepath}")

    ticks: list[Tick] = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"Tick CSV file appears to be empty: {filepath}")

        actual_columns = {col.strip().lower() for col in reader.fieldnames}
        missing = REQUIRED_COLUMNS - actual_columns
        if missing:
            raise ValueError(
                f"Tick CSV is missing required columns: {missing}\n"
                f"Found columns: {actual_columns}\n"
                f"File: {filepath}"
            )

        for line_num, row in enumerate(reader, start=2):
            clean_row = {k.strip().lower(): v.strip() for k, v in row.items()}

            try:
                tick = Tick(
                    timestamp = _parse_timestamp(clean_row["timestamp"]),
                    price     = float(clean_row["price"]),
                    volume    = float(clean_row["volume"]),
                )
                ticks.append(tick)
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Error reading {filepath} on line {line_num}: {exc}"
                ) from exc

    if not ticks:
        raise ValueError(f"Tick CSV file contains no data rows: {filepath}")

    ticks.sort(key=lambda t: t.timestamp)
    return ticks
