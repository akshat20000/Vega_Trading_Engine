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
| Phase 3 | Complete | Technical indicators from first principles: EMA, RSI, ATR, OBV |
| Phase 4–9 | Planned | Macro regime, strategies, risk, broker, backtest, API, dashboard |

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

## Technical Analysis Indicators

The engine computes technical indicators from first principles using pure Python and NumPy. No black-box indicator libraries are used in the core trading engine.

### Core Indicators

| Indicator | Module | Measures | Formula / Recurrence |
|---|---|---|---|
| **EMA** | [`vega.indicators.ema`](vega/indicators/ema.py) | Trend-following momentum | $EMA_t = \alpha P_t + (1 - \alpha) EMA_{t-1}$, with $\alpha = \frac{2}{period + 1}$. Initialized via SMA of first `period` prices at index `period - 1`. |
| **RSI** | [`vega.indicators.rsi`](vega/indicators/rsi.py) | Momentum & overbought/oversold levels | $RSI = 100 - \frac{100}{1 + RS}$, with Wilder-smoothed average gains and losses. Handles zero loss (100.0), zero gain (0.0), and flat prices (50.0). |
| **ATR** | [`vega.indicators.atr`](vega/indicators/atr.py) | Market volatility & gap risk | $TR_t = \max(H_t - L_t, \|H_t - C_{t-1}\|, \|L_t - C_{t-1}\|)$ with $TR_0 = H_0 - L_0$. Smoothed via Wilder recursion: $ATR_t = \frac{ATR_{t-1}(N-1) + TR_t}{N}$. |
| **OBV** | [`vega.indicators.obv`](vega/indicators/obv.py) | Cumulative volume flow | $OBV_t = OBV_{t-1} \pm V_t$ based on close comparison ($C_t > C_{t-1} \implies +V$, $C_t < C_{t-1} \implies -V$). Seeded at $OBV_0 = V_0$. |

### Why Implement From First Principles?

1. **Explainability in Interviews**: Every formula, seed convention, and recurrence relation is explicit and readable in 30 seconds.
2. **Transparency on Edge Cases**: Black-box libraries often mask warmup periods, silently return NaNs, or use undocumented seed heuristics. Vega makes all warmup delays and flat-price conventions explicit.
3. **No Framework Lock-in**: The core calculation logic operates on pure Python numeric sequences and `vega.data.models.Bar` objects, with zero reliance on Pandas Series metadata or TA-Lib C-extensions.

### Independent Reference Validation (`pandas-ta`)

[`validation/validate_indicators.py`](validation/validate_indicators.py) uses `pandas-ta` **only as an independent reference**:
- **EMA & OBV**: Bit-identical matches against `pandas-ta` (`max_diff = 0.0`).
- **RSI**: Bit-identical match from index 14 onwards (`max_diff < 1e-13`). `pandas-ta` seeds one bar earlier at index 13 by taking an initial mean over 13 differences (due to `diff()[0]` being NaN), whereas Vega strictly adheres to Wilder's 1978 textbook definition requiring 15 price points to form 14 price changes.
- **ATR**: Evaluates True Range with $TR_0 = H_0 - L_0$ in the initial SMA seed. `pandas-ta` drops $TR_0$ as NaN, causing a minor seed difference (~0.018) that decays exponentially by $\frac{13}{14}$ per bar and converges to $< 0.001$ after warmup.

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

## Testing & Validation

```bash
# Run all unit and validation tests
pytest -v

# Run specific test suites
pytest tests/test_tick_aggregation.py -v
pytest tests/test_parquet_store.py -v
pytest tests/test_duckdb_store.py -v
pytest tests/test_indicators.py -v
pytest tests/test_indicator_validation.py -v

# Run standalone independent cross-validation against pandas-ta
python validation/validate_indicators.py
```

Tests are grouped by component. All use synthetic data — no CSV files,
no network access, no external services required.
