"""
Tests for Phase 14: FastAPI REST Application & System Integration.

Verifies:
    1. /health endpoint: Process liveness (always 200).
    2. /status endpoint: Returns cached state or graceful domain fallback.
    3. /portfolio endpoint: Reads from cache, falling back cleanly to domain authority.
    4. /positions endpoint: Returns active positions from Portfolio authority.
    5. /trades endpoint: Returns standardized trade records from TradeBlotter.
    6. /metrics endpoint: Returns calculated blotter trading performance metrics.
    7. Empty cache fallback behavior.
    8. Redis failure fallback behavior:
       Redis unavailable ──> GET /portfolio ──> 200 ──> correct domain values.
    9. OpenAPI documentation endpoints (/docs and /openapi.json).
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from vega.api.app import create_app
from vega.cache.redis_cache import InMemoryRedisClient, RedisStateCache
from vega.orders.models import Fill, OrderSide
from vega.portfolio.portfolio import Portfolio
from vega.reporting.blotter import TradeBlotter, TradeBlotterEntry


class FailingClient:
    """Simulates severed Redis connection for failure testing."""

    def get(self, name: str) -> str:
        raise ConnectionError("Redis server connection severed")

    def set(self, name: str, value: str, ex: int | None = None) -> bool:
        raise ConnectionError("Redis server connection severed")


# ─── 1. Core Endpoint Tests ───────────────────────────────────────────────────


def test_health_endpoint() -> None:
    """Verify /health returns 200 with liveness metadata."""
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_status_endpoint_from_cache() -> None:
    """Verify /status reads from cache when available."""
    cache = RedisStateCache.create_in_memory()
    cache.set_system_status("ACTIVE", {"kill_switch": False, "mode": "LIVE"})

    app = create_app(cache=cache)
    client = TestClient(app)

    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ACTIVE"
    assert data["kill_switch"] is False
    assert data["source"] == "cache"
    assert data["details"]["mode"] == "LIVE"


def test_status_endpoint_domain_fallback() -> None:
    """Verify /status falls back gracefully to domain status when cache is empty."""
    cache = RedisStateCache.create_in_memory()  # empty
    app = create_app(cache=cache, system_status="RUNNING", kill_switch=False)
    client = TestClient(app)

    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "RUNNING"
    assert data["kill_switch"] is False
    assert data["source"] == "domain"


def test_portfolio_endpoint_from_cache() -> None:
    """Verify /portfolio reads cached equity when populated."""
    cache = RedisStateCache.create_in_memory()
    cache.set_portfolio(
        equity=1050000.0,
        cash=950000.0,
        unrealized_pnl=100000.0,
        realized_pnl=50000.0,
        drawdown_pct=2.5,
    )

    app = create_app(cache=cache)
    client = TestClient(app)

    response = client.get("/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert data["equity"] == 1050000.0
    assert data["cash"] == 950000.0
    assert data["realized_pnl"] == 50000.0
    assert data["unrealized_pnl"] == 100000.0
    assert data["drawdown_pct"] == 2.5
    assert data["source"] == "cache"


def test_portfolio_endpoint_redis_unavailable_fallback() -> None:
    """
    Critical Invariant:
        Redis unavailable ──> GET /portfolio ──> 200 ──> correct domain values.
    """
    portfolio = Portfolio(initial_cash=750_000.0)
    # Buy 25 NIFTY @ 20,000 with 20 brokerage
    fill = Fill(
        order_id="ORD-1",
        symbol="NIFTY",
        side=OrderSide.BUY,
        filled_qty=25,
        filled_price=20000.0,
        timestamp=datetime(2026, 4, 1, 9, 30),
        brokerage=20.0,
        slippage=0.0,
    )
    portfolio.process_fill(fill)

    # Cache with broken connection and fail_silent=True
    failing_cache = RedisStateCache(redis_client=FailingClient(), fail_silent=True)

    app = create_app(cache=failing_cache, portfolio=portfolio)
    client = TestClient(app)

    # Must return 200 (not 500) and authoritative domain equity
    response = client.get("/portfolio")
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "domain"
    assert data["equity"] == 749980.0  # Cash 249,980 + 500,000 market value = 749,980
    assert data["cash"] == 249980.0


def test_positions_endpoint() -> None:
    """Verify /positions returns open positions from Portfolio authority."""
    portfolio = Portfolio(initial_cash=1_000_000.0)
    fill = Fill(
        order_id="ORD-POS-1",
        symbol="BANKNIFTY",
        side=OrderSide.BUY,
        filled_qty=15,
        filled_price=48000.0,
        timestamp=datetime(2026, 4, 1, 9, 30),
        brokerage=20.0,
        slippage=0.0,
    )
    portfolio.process_fill(fill)

    app = create_app(portfolio=portfolio)
    client = TestClient(app)

    response = client.get("/positions")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    pos = data["positions"][0]
    assert pos["symbol"] == "BANKNIFTY"
    assert pos["quantity"] == 15
    assert pos["avg_entry_price"] == 48000.0


def test_trades_endpoint() -> None:
    """Verify /trades returns all 11 standardized fields from TradeBlotter."""
    ts = datetime(2026, 4, 1, 9, 30, tzinfo=timezone.utc)
    entry = TradeBlotterEntry(
        timestamp=ts,
        symbol="NIFTY",
        side="BUY",
        quantity=25,
        signal_price=22000.0,
        execution_price=22002.5,
        slippage=2.5,
        brokerage=20.0,
        realized_pnl=0.0,
        strategy="ATRGrid",
        reason="BUY_LEVEL_1",
    )
    blotter = TradeBlotter([entry])

    app = create_app(blotter=blotter)
    client = TestClient(app)

    response = client.get("/trades")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trades"] == 1
    trade = data["trades"][0]
    assert trade["symbol"] == "NIFTY"
    assert trade["side"] == "BUY"
    assert trade["quantity"] == 25
    assert trade["signal_price"] == 22000.0
    assert trade["execution_price"] == 22002.5
    assert trade["slippage"] == 2.5
    assert trade["brokerage"] == 20.0
    assert trade["realized_pnl"] == 0.0
    assert trade["strategy"] == "ATRGrid"
    assert trade["reason"] == "BUY_LEVEL_1"


def test_metrics_endpoint() -> None:
    """Verify /metrics returns calculated blotter performance indicators."""
    ts = datetime(2026, 4, 1, 9, 30)
    blotter = TradeBlotter(
        [
            TradeBlotterEntry(ts, "NIFTY", "BUY", 25, 22000.0, 22000.0, 0.0, 20.0, 0.0),
            TradeBlotterEntry(ts, "NIFTY", "SELL", 25, 22200.0, 22200.0, 0.0, 20.0, 5000.0),
        ]
    )

    app = create_app(blotter=blotter)
    client = TestClient(app)

    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["total_trades"] == 2
    assert data["closed_trades"] == 1
    assert data["win_rate"] == 1.0
    assert data["gross_realized_pnl"] == 5000.0
    assert data["total_brokerage"] == 40.0
    assert data["net_pnl"] == 4960.0


def test_empty_cache_fallback() -> None:
    """Verify empty cache yields source='domain' for portfolio and status."""
    cache = RedisStateCache.create_in_memory()
    app = create_app(cache=cache)
    client = TestClient(app)

    res_status = client.get("/status")
    assert res_status.status_code == 200
    assert res_status.json()["source"] == "domain"

    res_port = client.get("/portfolio")
    assert res_port.status_code == 200
    assert res_port.json()["source"] == "domain"


def test_openapi_docs_endpoint() -> None:
    """Verify FastAPI automatically exposes Swagger /docs and OpenAPI schema."""
    app = create_app()
    client = TestClient(app)

    res_docs = client.get("/docs")
    assert res_docs.status_code == 200

    res_schema = client.get("/openapi.json")
    assert res_schema.status_code == 200
    schema = res_schema.json()
    assert "/health" in schema["paths"]
    assert "/status" in schema["paths"]
    assert "/portfolio" in schema["paths"]
    assert "/positions" in schema["paths"]
    assert "/trades" in schema["paths"]
    assert "/metrics" in schema["paths"]
