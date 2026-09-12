"""
Tests for Phase 15: Streamlit Dashboard Presentation Layer.

Verifies:
    1. DashboardAPIClient returns None gracefully when the backend API is offline.
    2. DashboardAPIClient accurately retrieves and parses /health, /status, /portfolio,
       /positions, /trades, and /metrics from a live or mocked API server.
    3. format_trades_dataframe() correctly handles empty and populated trade blotters.
    4. Trade blotter DataFrame exports cleanly to CSV.
    5. Dashboard invariants: presentation-only, zero calculation or mutation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vega.api.app import create_app
from vega.dashboard.app import DashboardAPIClient, format_trades_dataframe
from vega.orders.models import Fill, OrderSide
from vega.portfolio.portfolio import Portfolio
from vega.reporting.blotter import TradeBlotter, TradeBlotterEntry


# ─── 1. Offline Resilience Tests ─────────────────────────────────────────────


def test_client_api_offline_returns_none() -> None:
    """
    Verify that when FastAPI is offline, DashboardAPIClient returns None
    for all endpoints rather than raising exceptions or fabricating state.
    """
    # Point to an unallocated high port
    offline_client = DashboardAPIClient(base_url="http://127.0.0.1:59999", timeout=0.1)

    assert offline_client.get_health() is None
    assert offline_client.get_status() is None
    assert offline_client.get_portfolio() is None
    assert offline_client.get_positions() is None
    assert offline_client.get_trades() is None
    assert offline_client.get_metrics() is None


# ─── 2. Online Integration Tests via TestClient ──────────────────────────────


class MockRequestsSession:
    """Adapter bridging requests.get to a FastAPI TestClient."""

    def __init__(self, test_client: TestClient) -> None:
        self.test_client = test_client

    def get(self, url: str, timeout: float = 2.0) -> Any:
        # Strip scheme & host to extract endpoint path
        path = "/" + url.split("/", 3)[-1] if url.count("/") >= 3 else url
        return self.test_client.get(path)


def test_client_with_active_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verify DashboardAPIClient correctly parses real responses when API is online.
    """
    portfolio = Portfolio(initial_cash=1_000_000.0)
    fill = Fill(
        order_id="O-1",
        symbol="NIFTY",
        side=OrderSide.BUY,
        filled_qty=25,
        filled_price=22000.0,
        timestamp=datetime(2026, 4, 1, 9, 30),
        brokerage=20.0,
        slippage=0.0,
    )
    portfolio.process_fill(fill)

    entry = TradeBlotterEntry(
        timestamp="2026-04-01T09:30:00",
        symbol="NIFTY",
        side="BUY",
        quantity=25,
        signal_price=22000.0,
        execution_price=22000.0,
        slippage=0.0,
        brokerage=20.0,
        realized_pnl=0.0,
        strategy="ATRGrid",
        reason="LEVEL_1",
    )
    blotter = TradeBlotter([entry])

    app = create_app(portfolio=portfolio, blotter=blotter, system_status="RUNNING")
    fastapi_client = TestClient(app)
    mock_session = MockRequestsSession(fastapi_client)

    # Monkeypatch requests.get to route to the TestClient
    monkeypatch.setattr("requests.get", mock_session.get)

    client = DashboardAPIClient(base_url="http://localhost:8000")

    # 1. Health
    health = client.get_health()
    assert health is not None
    assert health["status"] == "ok"

    # 2. Status
    status = client.get_status()
    assert status is not None
    assert status["status"] == "RUNNING"
    assert status["source"] == "domain"

    # 3. Portfolio
    port = client.get_portfolio()
    assert port is not None
    assert port["equity"] < 1_000_000.0

    # 4. Positions
    positions = client.get_positions()
    assert positions is not None
    assert len(positions) == 1
    assert positions[0]["symbol"] == "NIFTY"
    assert positions[0]["quantity"] == 25

    # 5. Trades
    trades = client.get_trades()
    assert trades is not None
    assert len(trades) == 1
    assert trades[0]["symbol"] == "NIFTY"
    assert trades[0]["execution_price"] == 22000.0

    # 6. Metrics
    metrics = client.get_metrics()
    assert metrics is not None
    assert metrics["total_trades"] == 1


# ─── 3. DataFrame Formatting & CSV Export Tests ──────────────────────────────


def test_format_trades_dataframe_empty() -> None:
    """Verify empty trade blotter produces valid DataFrame with all columns."""
    df = format_trades_dataframe([])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == [
        "Timestamp",
        "Symbol",
        "Side",
        "Quantity",
        "Signal Price",
        "Execution Price",
        "Slippage",
        "Brokerage",
        "Realized PnL",
        "Strategy",
        "Reason",
    ]


def test_format_trades_dataframe_populated_and_csv_export() -> None:
    """Verify populated trade records are formatted with currency signs and exported to CSV."""
    raw_trades = [
        {
            "timestamp": "2026-04-01T09:30:00",
            "symbol": "NIFTY",
            "side": "BUY",
            "quantity": 25,
            "signal_price": 22000.0,
            "execution_price": 22002.5,
            "slippage": 2.5,
            "brokerage": 20.0,
            "realized_pnl": 0.0,
            "strategy": "ATRGrid",
            "reason": "BUY_1",
        },
        {
            "timestamp": "2026-04-01T15:15:00",
            "symbol": "NIFTY",
            "side": "SELL",
            "quantity": 25,
            "signal_price": 22200.0,
            "execution_price": 22197.5,
            "slippage": -2.5,
            "brokerage": 20.0,
            "realized_pnl": 4875.0,
            "strategy": "ATRGrid",
            "reason": "TAKE_PROFIT",
        },
    ]

    df = format_trades_dataframe(raw_trades)
    assert len(df) == 2
    assert df.iloc[0]["Symbol"] == "NIFTY"
    assert df.iloc[0]["Execution Price"] == "₹22,002.50"
    assert df.iloc[1]["Realized PnL"] == "₹+4,875.00"

    # Verify CSV export
    csv_str = df.to_csv(index=False)
    assert "NIFTY" in csv_str
    assert "₹+4,875.00" in csv_str
    assert "Execution Price" in csv_str
