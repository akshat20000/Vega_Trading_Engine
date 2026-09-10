"""
DuckDB query layer for Parquet-stored market data.

DuckDB is an in-process analytical SQL engine that can query Parquet files
directly. It is a Python library — no server, no installation beyond pip.

Why DuckDB?
    - Reads Parquet files with predicate pushdown: only matching rows are
      read from disk, not the whole file. This matters for date-range queries
      over large historical datasets.
    - SQL is the natural language for aggregation: COUNT, MIN, MAX, AVG are
      single-line expressions.
    - Zero infrastructure: duckdb.connect() creates an in-memory engine with
      no files written and no background processes started.

Why not PostgreSQL?
    - PostgreSQL requires running a separate database server process.
    - Our data is append-only historical data, not live transactional data.
    - Parquet + DuckDB is sufficient and requires zero operational overhead.

Design rule — DuckDB does NOT store or duplicate data:
    DuckDB reads the .parquet files that ParquetStore wrote.
    The .parquet files are always the single source of truth.
    If you delete a .parquet file, DuckDB has nothing to query.

Data flow:
    CSV / Ticks
        ↓  (csv_loader / tick_aggregator)
    list[Bar]
        ↓  (ParquetStore.save)
    .parquet file    <-- single source of truth
        ↓  (DuckDBStore reads_parquet directly)
    SQL queries --> list[Bar] or aggregate values

Typical usage:
    from vega.data.duckdb_store import DuckDBStore
    store = DuckDBStore()
    bars  = store.query_bars("data/market/NIFTY50.parquet",
                              start=datetime(2023, 1, 1),
                              end=datetime(2023, 3, 31))
    count = store.count_bars("data/market/NIFTY50.parquet")
    low   = store.min_low("data/market/NIFTY50.parquet")
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb

from vega.data.models import Bar
from vega.data.parquet_store import to_python_datetime


class DuckDBStore:
    """
    Provides SQL analytical queries over Parquet files using DuckDB.

    The connection is in-memory only — DuckDB writes nothing to disk.
    One DuckDBStore instance per use-site is the intended pattern.

    This class contains no strategy logic, indicators, risk rules,
    or broker code. It is responsible only for querying market data.

    Example:
        store = DuckDBStore()

        # Load a date range
        bars = store.query_bars(
            "data/market/NIFTY50.parquet",
            start=datetime(2023, 1, 1),
            end=datetime(2023, 3, 31),
        )

        # Aggregate statistics
        print(store.count_bars("data/market/NIFTY50.parquet"))  # 62
        print(store.max_high("data/market/NIFTY50.parquet"))    # e.g. 18500.0
    """

    def __init__(self) -> None:
        # In-memory DuckDB connection — nothing is written to disk by DuckDB.
        self._conn = duckdb.connect()

    def close(self) -> None:
        """Close the DuckDB in-memory connection. Safe to call multiple times."""
        self._conn.close()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sql_path(self, filepath: str | Path) -> str:
        """
        Return the absolute path as a forward-slash string for use in SQL.

        DuckDB's read_parquet() requires forward slashes on all platforms.
        Backslashes on Windows would need escaping inside SQL string literals,
        so we resolve to an absolute path and convert to POSIX form.
        """
        return Path(filepath).resolve().as_posix()

    def _require_file(self, filepath: str | Path) -> None:
        """
        Raise FileNotFoundError with a helpful message if the file is absent.

        Called at the top of every public method so the error is clear
        before DuckDB gets a chance to produce a less readable internal error.
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"Parquet file not found: {filepath}\n"
                "Create it first with ParquetStore.save()."
            )

    def _rows_to_bars(self, rows: list[tuple]) -> list[Bar]:
        """
        Convert raw DuckDB result rows into Bar objects.

        Expected column order (as selected in every query method):
            0: timestamp, 1: open, 2: high, 3: low, 4: close, 5: volume

        Bar.__post_init__ runs for each row, so any OHLCV inconsistency
        in the Parquet data is caught and raised as a ValueError.
        """
        bars: list[Bar] = []
        for row in rows:
            ts = to_python_datetime(row[0])
            bars.append(
                Bar(
                    timestamp = ts,
                    open      = float(row[1]),
                    high      = float(row[2]),
                    low       = float(row[3]),
                    close     = float(row[4]),
                    volume    = float(row[5]),
                )
            )
        return bars

    # ── Query methods ─────────────────────────────────────────────────────────

    def query_bars(
        self,
        filepath: str | Path,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[Bar]:
        """
        Load Bar objects from a Parquet file, optionally filtered by date.

        The date filter is applied inside DuckDB (predicate pushdown), so only
        matching rows are read from the Parquet file — the entire file is not
        loaded into memory first.

        Both start and end are inclusive. Passing None removes that bound.

        Args:
            filepath: Path to the .parquet file to query.
            start:    Only return bars where timestamp >= start.
                      Pass None for no lower bound (return from beginning).
            end:      Only return bars where timestamp <= end.
                      Pass None for no upper bound (return to the end).

        Returns:
            A list of Bar objects sorted chronologically (oldest first).
            An empty list if no bars match the filter — not an error.

        Raises:
            FileNotFoundError: If the Parquet file does not exist.

        Example:
            # All of 2023 Q1
            bars = store.query_bars(
                "NIFTY50.parquet",
                start=datetime(2023, 1, 1),
                end=datetime(2023, 3, 31),
            )
        """
        self._require_file(filepath)
        path = self._sql_path(filepath)

        # Build the WHERE clause from whichever bounds are provided.
        # Parameterised (?) to let DuckDB handle type conversion correctly.
        conditions: list[str] = []
        params: list[object] = []

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        sql = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM   read_parquet('{path}')
            {where}
            ORDER  BY timestamp
        """

        rows = self._conn.execute(sql, params).fetchall()
        return self._rows_to_bars(rows)

    def count_bars(self, filepath: str | Path) -> int:
        """
        Return the total number of bars in a Parquet file.

        SQL equivalent: SELECT COUNT(*) FROM file.parquet

        Args:
            filepath: Path to the .parquet file.

        Returns:
            Row count as a Python int.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._require_file(filepath)
        path = self._sql_path(filepath)
        result = self._conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{path}')"
        ).fetchone()
        return int(result[0])

    def min_low(self, filepath: str | Path) -> float:
        """
        Return the lowest 'low' price across all bars in the file.

        SQL equivalent: SELECT MIN(low) FROM file.parquet

        Args:
            filepath: Path to the .parquet file.

        Returns:
            The minimum low as a Python float.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._require_file(filepath)
        path = self._sql_path(filepath)
        result = self._conn.execute(
            f"SELECT MIN(low) FROM read_parquet('{path}')"
        ).fetchone()
        return float(result[0])

    def max_high(self, filepath: str | Path) -> float:
        """
        Return the highest 'high' price across all bars in the file.

        SQL equivalent: SELECT MAX(high) FROM file.parquet

        Args:
            filepath: Path to the .parquet file.

        Returns:
            The maximum high as a Python float.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._require_file(filepath)
        path = self._sql_path(filepath)
        result = self._conn.execute(
            f"SELECT MAX(high) FROM read_parquet('{path}')"
        ).fetchone()
        return float(result[0])

    def avg_volume(self, filepath: str | Path) -> float:
        """
        Return the average volume across all bars in the file.

        SQL equivalent: SELECT AVG(volume) FROM file.parquet

        Note on floating-point precision:
            DuckDB's internal accumulation order for AVG may differ from a
            simple Python sum(volumes)/len(volumes). The results will be very
            close but may not be bit-identical. Use pytest.approx() when
            testing this value.

        Args:
            filepath: Path to the .parquet file.

        Returns:
            The average volume as a Python float.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self._require_file(filepath)
        path = self._sql_path(filepath)
        result = self._conn.execute(
            f"SELECT AVG(volume) FROM read_parquet('{path}')"
        ).fetchone()
        return float(result[0])

    def run_sql(self, sql: str, params: list | None = None) -> list[tuple]:
        """
        Execute arbitrary SQL against any Parquet file referenced in the query.

        Use read_parquet('path/to/file.parquet') inside your SQL to name files.
        Results are returned as a list of plain Python tuples.

        This method is intended for ad-hoc exploration and is not called by
        the trading engine internally.

        Args:
            sql:    A DuckDB SQL query string.
            params: Optional list of positional ? parameters.

        Returns:
            A list of tuples, one per result row.

        Example:
            rows = store.run_sql(
                "SELECT date_trunc('month', timestamp) AS month, "
                "       AVG(close) AS avg_close "
                "FROM   read_parquet(?) "
                "GROUP  BY 1 "
                "ORDER  BY 1",
                params=["data/market/NIFTY50.parquet"]
            )
        """
        return self._conn.execute(sql, params or []).fetchall()
