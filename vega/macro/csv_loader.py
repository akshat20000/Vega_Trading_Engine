"""
CSV loader for macro proxy data.

Reads a CSV file containing macro indicator rows:
    date, nifty_trend, india_vix, usdinr

Column order does not matter — header lookup is case-insensitive.
Returns a list of MacroSnapshot objects sorted chronologically (oldest first).

Typical usage:
    from vega.macro.csv_loader import load_macro_snapshots
    snapshots = load_macro_snapshots("data/macro/macro_data.csv")
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from vega.macro.models import MacroSnapshot


# Required columns in the macro CSV (checked case-insensitively)
REQUIRED_COLUMNS: set[str] = {"date", "nifty_trend", "india_vix", "usdinr"}

# Date formats attempted sequentially
DATE_FORMATS: list[str] = [
    "%Y-%m-%d",            # 2023-01-02
    "%Y-%m-%d %H:%M:%S",  # 2023-01-02 00:00:00
    "%d-%m-%Y",            # 02-01-2023
    "%d/%m/%Y",            # 02/01/2023
]


def _parse_date(value: str) -> date:
    """
    Parse a date string using known formats.

    Args:
        value: Raw date string from CSV.

    Returns:
        A datetime.date object.

    Raises:
        ValueError: If no format matches.
    """
    stripped = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot parse date '{value}'. "
        f"Supported formats: {DATE_FORMATS}"
    )


def load_macro_snapshots(filepath: str | Path) -> list[MacroSnapshot]:
    """
    Load macro indicator snapshots from a CSV file.

    The CSV must have a header row with at minimum these columns:
        date, nifty_trend, india_vix, usdinr

    Rows are validated and returned sorted chronologically (oldest first).
    Duplicate dates are rejected with a ValueError.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A list of MacroSnapshot objects sorted from oldest to newest.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, missing required columns, contains
                    malformed/impossible rows, or has duplicate dates.

    Example:
        >>> snapshots = load_macro_snapshots("data/macro/macro_data.csv")
        >>> print(snapshots[0].india_vix)
        14.51
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Macro CSV file not found: {filepath}")

    snapshots: list[MacroSnapshot] = []

    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file appears to be empty: {filepath}")

        # Normalise columns to lowercase
        actual_columns = {col.strip().lower() for col in reader.fieldnames}
        missing = REQUIRED_COLUMNS - actual_columns
        if missing:
            raise ValueError(
                f"Macro CSV is missing required columns: {missing}\n"
                f"Found columns: {actual_columns}\n"
                f"File: {filepath}"
            )

        for line_num, row in enumerate(reader, start=2):
            clean_row = {k.strip().lower(): v.strip() for k, v in row.items()}

            try:
                parsed_date = _parse_date(clean_row["date"])
                nifty_trend = float(clean_row["nifty_trend"])
                india_vix   = float(clean_row["india_vix"])
                usdinr      = float(clean_row["usdinr"])

                snapshot = MacroSnapshot(
                    date=parsed_date,
                    nifty_trend=nifty_trend,
                    india_vix=india_vix,
                    usdinr=usdinr,
                )
                snapshots.append(snapshot)
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Error reading {filepath} on line {line_num}: {exc}"
                ) from exc

    if not snapshots:
        raise ValueError(f"Macro CSV contains no data rows: {filepath}")

    # Sort chronologically
    snapshots.sort(key=lambda s: s.date)

    # Reject duplicate dates
    for i in range(1, len(snapshots)):
        if snapshots[i].date == snapshots[i - 1].date:
            raise ValueError(
                f"Duplicate date found in {filepath}: {snapshots[i].date}. "
                "Each macro snapshot must have a unique date."
            )

    return snapshots
