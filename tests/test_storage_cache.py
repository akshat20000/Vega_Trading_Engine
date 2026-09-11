"""
Tests for Phase 13: Redis Latest-State Cache & Storage Integration.

Verifies:
    1. RedisStateCache operations:
       - Set and get latest prices.
       - Set and get positions.
       - Set and get portfolio valuation metrics.
       - Set and get system status.
       - Empty cache behavior (returns None).
       - Clear cache behavior (returns None).
       - Rehydration from domain Portfolio into Redis cache.
    2. Failure Isolation Invariant:
       - Redis failure does not break or corrupt domain state (Portfolio/Risk/Broker).
    3. Storage Integration:
       - Historical analytical path (Parquet + DuckDB) operates alongside transient Redis cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import tempfile
import pytest

from vega.cache.redis_cache import InMemoryRedisClient, RedisStateCache
from vega.data.models import Bar
from vega.data.duckdb_store import DuckDBStore
from vega.data.parquet_store import ParquetStore
from vega.orders.models import Fill, OrderSide
from vega.portfolio.portfolio import Portfolio


# ─── 1. Basic Cache CRUD & Schema Tests ───────────────────────────────────────


def test_set_and_get_latest_price() -> None:
    """Verify setting and retrieving an instrument's latest market price."""
    cache = RedisStateCache.create_in_memory()
    ts = datetime(2026, 4, 1, 9, 30, 0, tzinfo=timezone.utc)

    assert cache.set_latest_price("NIFTY50", 22000.25, timestamp=ts) is True
    data = cache.get_latest_price("NIFTY50")

    assert data is not None
    assert data["symbol"] == "NIFTY50"
    assert data["price"] == 22000.25
    assert data["timestamp"] == ts.isoformat()


def test_set_and_get_position() -> None:
    """Verify caching position quantity and average cost basis."""
    cache = RedisStateCache.create_in_memory()

    assert cache.set_position("BANKNIFTY", quantity=15, avg_entry_price=48000.5) is True
    pos = cache.get_position("BANKNIFTY")

    assert pos is not None
    assert pos["symbol"] == "BANKNIFTY"
    assert pos["quantity"] == 15
    assert pos["avg_entry_price"] == 48000.5


def test_set_and_get_portfolio() -> None:
    """Verify caching top-level portfolio valuation metrics."""
    cache = RedisStateCache.create_in_memory()

    assert (
        cache.set_portfolio(
            equity=1050000.0,
            cash=950000.0,
            unrealized_pnl=100000.0,
            realized_pnl=50000.0,
            drawdown_pct=1.25,
        )
        is True
    )

    port = cache.get_portfolio()
    assert port is not None
    assert port["equity"] == 1050000.0
    assert port["cash"] == 950000.0
    assert port["unrealized_pnl"] == 100000.0
    assert port["realized_pnl"] == 50000.0
    assert port["drawdown_pct"] == 1.25
    assert "timestamp" in port


def test_set_and_get_system_status() -> None:
    """Verify caching system status and diagnostic details."""
    cache = RedisStateCache.create_in_memory()

    assert cache.set_system_status("RUNNING", {"kill_switch": False, "mode": "SIMULATION"}) is True
    status = cache.get_system_status()

    assert status is not None
    assert status["status"] == "RUNNING"
    assert status["kill_switch"] is False
    assert status["mode"] == "SIMULATION"


# ─── 2. Boundary & Lifecycle Tests ───────────────────────────────────────────


def test_empty_cache_returns_none() -> None:
    """Verify queries against an empty cache return None without errors."""
    cache = RedisStateCache.create_in_memory()

    assert cache.get_latest_price("UNKNOWN") is None
    assert cache.get_position("UNKNOWN") is None
    assert cache.get_portfolio() is None
    assert cache.get_system_status() is None


def test_clear_cache_returns_none() -> None:
    """Verify clearing cache removes all keys matching the namespace."""
    cache = RedisStateCache.create_in_memory()

    cache.set_latest_price("NIFTY50", 22000.0)
    cache.set_position("NIFTY50", 25, 22000.0)
    cache.set_portfolio(1000000.0, 1000000.0)
    cache.set_system_status("ACTIVE")

    assert cache.get_latest_price("NIFTY50") is not None

    cache.clear()

    assert cache.get_latest_price("NIFTY50") is None
    assert cache.get_position("NIFTY50") is None
    assert cache.get_portfolio() is None
    assert cache.get_system_status() is None


def test_rehydrate_restores_all_keys() -> None:
    """
    Verify rehydrating Redis from in-memory domain state:
        Domain (Portfolio) ──> Redis Cache
    """
    portfolio = Portfolio(initial_cash=1_000_000.0)

    # Process execution fills into domain portfolio
    fill1 = Fill(
        order_id="O-1",
        symbol="NIFTY",
        side=OrderSide.BUY,
        filled_qty=25,
        filled_price=22000.0,
        timestamp=datetime(2026, 4, 1, 9, 30),
        brokerage=20.0,
        slippage=0.0,
    )
    fill2 = Fill(
        order_id="O-2",
        symbol="BANKNIFTY",
        side=OrderSide.BUY,
        filled_qty=15,
        filled_price=48000.0,
        timestamp=datetime(2026, 4, 1, 9, 31),
        brokerage=20.0,
        slippage=0.0,
    )
    portfolio.process_fill(fill1)
    portfolio.process_fill(fill2)

    cache = RedisStateCache.create_in_memory()
    # Cache starts empty
    assert cache.get_position("NIFTY") is None

    # Rehydrate
    rehydrated_count = cache.rehydrate(
        portfolio=portfolio,
        latest_prices={"NIFTY": 22100.0, "BANKNIFTY": 48200.0},
        system_status="RUNNING",
    )

    # 2 positions + 1 portfolio equity + 2 latest prices + 1 system status = 6
    assert rehydrated_count == 6

    nifty_pos = cache.get_position("NIFTY")
    assert nifty_pos is not None
    assert nifty_pos["quantity"] == 25
    assert nifty_pos["avg_entry_price"] == 22000.0

    bn_pos = cache.get_position("BANKNIFTY")
    assert bn_pos is not None
    assert bn_pos["quantity"] == 15
    assert bn_pos["avg_entry_price"] == 48000.0

    port_data = cache.get_portfolio()
    assert port_data is not None
    assert port_data["cash"] < 1_000_000.0

    assert cache.get_latest_price("NIFTY")["price"] == 22100.0
    assert cache.get_system_status()["status"] == "RUNNING"


# ─── 3. Failure Isolation & Architectural Invariant Tests ────────────────────


class FailingRedisClient:
    """Mock client that deliberately simulates broken network / server failure."""

    def get(self, name: str) -> str:
        raise ConnectionError("Connection refused by Redis server: 127.0.0.1:6379")

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        raise ConnectionError("Connection refused by Redis server: 127.0.0.1:6379")

    def delete(self, *names: str) -> int:
        raise ConnectionError("Connection refused by Redis server: 127.0.0.1:6379")


def test_redis_failure_does_not_break_domain_operation() -> None:
    """
    Critical Invariant:
        Redis failure must NOT crash or corrupt trading engine operations.
        Portfolio/Risk/Broker continue unaffected.
    """
    portfolio = Portfolio(initial_cash=500_000.0)
    failing_client = FailingRedisClient()

    # Create cache pointing to the severed Redis connection with fail_silent=True
    cache = RedisStateCache(redis_client=failing_client, fail_silent=True)

    # 1. Domain operation proceeds normally
    fill = Fill(
        order_id="O-100",
        symbol="NIFTY",
        side=OrderSide.BUY,
        filled_qty=25,
        filled_price=22000.0,
        timestamp=datetime(2026, 4, 1, 9, 30),
        brokerage=20.0,
        slippage=0.0,
    )
    realized_pnl = portfolio.process_fill(fill)

    # Domain state is valid
    pos = portfolio.get_position("NIFTY")
    assert pos.quantity == 25
    assert pos.average_entry_price == Decimal("22000.00")
    assert realized_pnl == Decimal("0.00")

    # 2. Cache writes fail gracefully without raising unhandled exceptions
    ok_price = cache.set_latest_price("NIFTY", 22000.0)
    ok_pos = cache.set_position("NIFTY", pos.quantity, float(pos.average_entry_price))
    ok_port = cache.set_portfolio(float(portfolio.get_equity()), float(portfolio.get_cash()))

    assert ok_price is False
    assert ok_pos is False
    assert ok_port is False

    # Cache read also fails gracefully
    assert cache.get_latest_price("NIFTY") is None
    assert cache.get_position("NIFTY") is None

    # 3. Domain state remains completely uncorrupted
    assert portfolio.get_position("NIFTY").quantity == 25
    assert portfolio.get_cash() == Decimal("500000.00") - Decimal("550020.00")  # Cash adjusted accurately


# ─── 4. Storage Integration Test (Parquet + DuckDB Historical Path) ──────────


def test_duckdb_parquet_storage_path(tmp_path: Path) -> None:
    """
    Verify historical data persistence (Parquet) and analytical SQL querying (DuckDB)
    alongside the live telemetry cache.
    """
    parquet_file = tmp_path / "NIFTY50.parquet"
    bars = [
        Bar(datetime(2026, 1, 2, 9, 30), 21500.0, 21550.0, 21480.0, 21520.0, 50000.0),
        Bar(datetime(2026, 1, 2, 9, 31), 21520.0, 21600.0, 21510.0, 21580.0, 75000.0),
        Bar(datetime(2026, 1, 2, 9, 32), 21580.0, 21610.0, 21560.0, 21570.0, 60000.0),
    ]

    # Save to Parquet
    p_store = ParquetStore()
    p_store.save(bars, parquet_file)
    assert parquet_file.exists()

    # Query via DuckDBStore
    duck_store = DuckDBStore()
    loaded_bars = duck_store.query_bars(parquet_file)
    assert len(loaded_bars) == 3
    assert loaded_bars[0].close == 21520.0
    assert loaded_bars[1].high == 21600.0

    # Execute analytical aggregations using DuckDB
    assert duck_store.count_bars(parquet_file) == 3
    assert duck_store.max_high(parquet_file) == 21610.0
    assert duck_store.min_low(parquet_file) == 21480.0
