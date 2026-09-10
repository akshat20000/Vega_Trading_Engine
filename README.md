# Vega Quant Trading Engine

A modular Python trading engine with lightweight supporting services for API,
dashboard, and state caching. Built as a technical assessment for a Quant
Developer role at Vega Developers.

The core trading and backtest engine is a single-process Python application.
Redis, FastAPI, and Streamlit run as lightweight companion services via Docker Compose.

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| Phase 1 | Complete | Data foundation: models, CSV/tick loaders, tick aggregation, contracts |
| Phase 2 | Complete | Storage: Parquet + DuckDB |
| Phase 3–9 | Planned | Indicators, macro regime, strategies, risk, broker, backtest, API |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic sample data
python scripts/generate_sample_data.py

# Run all tests
pytest -q
```

---

## Package Structure

```
vega/
  data/          Market data loading and storage
  indicators/    Technical analysis (EMA, RSI, ATR, OBV) — Phase 3
  macro/         Macro regime engine — Phase 4
  strategy/      ATR Grid and Stop-and-Reverse strategies — Phase 6
  risk/          Pre-order risk management — Phase 5
  orders/        Order and fill models — Phase 5
  broker/        PaperBroker + KiteBroker skeleton — Phase 5
  portfolio/     Position and P&L tracking — Phase 6
  engine/        Backtest and walk-forward — Phase 7
  reporting/     Trade blotter and backtest reports — Phase 7
  cache/         Redis latest-state cache — Phase 9
  api/           FastAPI REST endpoints — Phase 9
dashboard/       Streamlit dashboard — Phase 9
tests/           pytest test suite
validation/      pandas-ta / vectorbt cross-validation (optional)
data/market/     Synthetic sample OHLCV CSV files
data/macro/      Synthetic sample macro CSV files
```

---

## Storage Design

### Why Apache Parquet?

Parquet is a **columnar binary format** built for time-series data.

| Property | CSV | Parquet |
|---|---|---|
| Size | ~1x baseline | ~5–10x smaller |
| Column access | Reads entire row | Reads only requested columns |
| Timestamp handling | String parsing on every load | Native datetime64 storage |
| Schema enforcement | None | Typed schema per column |

When a strategy needs only the `close` column for 5 years of daily data,
Parquet reads only that column's bytes from disk. CSV reads every character
of every row.

**Float64 round-trip accuracy**: Python floats are 64-bit IEEE 754. Parquet
stores them as float64. Reading back produces the bit-identical float64 —
exact equality (`==`) holds. No precision is lost in the save/load cycle.

### Why DuckDB?

DuckDB is an **in-process analytical SQL engine**. It reads Parquet files
directly — no server, no background process, no configuration file.

```python
import duckdb
conn = duckdb.connect()   # in-memory, nothing written to disk

# Query Parquet directly with SQL
rows = conn.execute("""
    SELECT AVG(close), MIN(low), MAX(high)
    FROM   read_parquet('data/market/NIFTY50.parquet')
    WHERE  timestamp >= '2023-01-01'
      AND  timestamp <= '2023-03-31'
""").fetchall()
```

DuckDB applies **predicate pushdown** into the Parquet reader: only rows
matching the `WHERE` clause are read from disk, not the whole file.

### Why NOT PostgreSQL?

PostgreSQL is the right tool when you need:
- A running database server that multiple applications write to simultaneously
- Complex relational joins across many tables
- Live transactional writes with ACID guarantees

This project needs none of those things. Our data is append-only historical
OHLCV data, read by a single process. Parquet + DuckDB is sufficient and
requires zero operational overhead.

PostgreSQL would be the right upgrade if the engine were deployed in a
multi-user environment with live data streaming from multiple sources.

### Data Flow

```
CSV files / Tick CSV
        |
        v  (csv_loader / tick_loader / tick_aggregator)
  list[Bar]
        |
        v  (ParquetStore.save)
  .parquet file   <-- single source of truth on disk
        |
        +---> ParquetStore.load()    --> list[Bar]  (full file)
        |
        +---> DuckDBStore.query_bars()  --> list[Bar]  (date-range filtered)
        +---> DuckDBStore.count_bars()  --> int
        +---> DuckDBStore.min_low()     --> float
        +---> DuckDBStore.max_high()    --> float
        +---> DuckDBStore.avg_volume()  --> float
        +---> DuckDBStore.run_sql()     --> list[tuple]  (arbitrary SQL)
```

**DuckDB does not store a separate copy of the data.** Deleting a `.parquet`
file removes the data from both `ParquetStore` and `DuckDBStore`.

---

## Configuration

All tunable parameters are in [`config.yaml`](config.yaml).
Copy [`config.example.yaml`](config.example.yaml) as a starting reference.

To override locally without committing changes, create `config.local.yaml`
(already in `.gitignore`).

> **Security**: `config.yaml` contains only numeric parameters.
> Broker credentials (when added in a future phase) must go in
> `config.local.yaml` — never in `config.yaml`.

---

## Sample Data

Synthetic sample data is generated by `scripts/generate_sample_data.py`
(seeded with `random.seed(42)` for reproducibility):

| File | Rows | Description |
|---|---|---|
| `data/market/NIFTY50_daily.csv` | 120 | Daily OHLCV bars, Jan–Jun 2023 |
| `data/market/NIFTY50_ticks.csv` | 60 | Intraday ticks for 2023-01-02 |
| `data/macro/macro_data.csv` | 120 | NIFTY trend, India VIX, USDINR |

All unit tests use **synthetic, hand-crafted data** created in `tests/conftest.py`.
No test reads from these CSV files.

---

## Testing

```bash
# Run all tests
pytest -q

# Run a specific test file
pytest tests/test_tick_aggregation.py -v
pytest tests/test_parquet_store.py -v
pytest tests/test_duckdb_store.py -v
```

Tests are grouped by component. All use synthetic data — no CSV files,
no network access, no external services required.
