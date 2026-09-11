"""
Tests for Phase 11: Observability, Structured Logging, and Trade Blotter.

Verifies:
    Trade Blotter:
        - All 11 standardized trade fields are present.
        - Accounting integrity: Gross Realized P&L - Brokerage = Net P&L (no slippage double-counting).
        - Edge cases: Empty blotter, zero closed trades, all-win trades without zero-division error.
        - Exports: to_dict(), to_dataframe(), and format_table().
        - BacktestEngine integration: result.blotter produces cached trade records.
    Structured Logging & Alerts:
        - Fixed JSON schema for domain events.
        - Logging != Alerting invariant: routine events do not trigger alerts; only critical events (Kill Switch, CRITICAL) fire alert callbacks.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import List

import pandas as pd
import pytest

from vega.engine.backtest import BacktestEngine, BacktestResult, TradeRecord
from vega.orders.models import OrderSide
from vega.reporting.blotter import TradeBlotter, TradeBlotterEntry
from vega.reporting.logger import LogEvent, LogEventType, StructuredLogger
from vega.strategy.stop_and_reverse import StopAndReverseStrategy


# ─── 1. Trade Blotter Accounting & Field Verification ─────────────────────────


def test_trade_blotter_entry_fields() -> None:
    """Verify TradeBlotterEntry contains all 11 required fields with accurate typing."""
    ts = datetime(2023, 1, 2, 9, 30, tzinfo=timezone.utc)
    entry = TradeBlotterEntry(
        timestamp=ts,
        symbol="NIFTY50",
        side="BUY",
        quantity=50,
        signal_price=18000.0,
        execution_price=18002.5,
        slippage=2.5,
        brokerage=20.0,
        realized_pnl=0.0,
        strategy="ATRGridStrategy",
        reason="GRID_BUY_LEVEL_1",
    )

    d = entry.to_dict()
    assert d["timestamp"] == ts.isoformat()
    assert d["symbol"] == "NIFTY50"
    assert d["side"] == "BUY"
    assert d["quantity"] == 50
    assert d["signal_price"] == 18000.0
    assert d["execution_price"] == 18002.5
    assert d["slippage"] == 2.5
    assert d["brokerage"] == 20.0
    assert d["realized_pnl"] == 0.0
    assert d["strategy"] == "ATRGridStrategy"
    assert d["reason"] == "GRID_BUY_LEVEL_1"


def test_trade_blotter_accounting_no_double_counting() -> None:
    """
    Verify precise accounting rules:
        - Entry order has realized_pnl = 0.0
        - Closing order has actual realized P&L
        - Gross Realized P&L = sum(realized_pnl)
        - Net P&L = Gross Realized P&L - Brokerage
        - Slippage is NOT subtracted again from net P&L (already embedded in execution price)
    """
    ts = datetime(2023, 1, 2, 9, 30)

    # Buy 10 @ 100 with 10 brokerage, slippage 0.5 (execution 100.5) -> entry, realized PnL = 0
    t1 = TradeBlotterEntry(
        timestamp=ts,
        symbol="NIFTY50",
        side="BUY",
        quantity=10,
        signal_price=100.0,
        execution_price=100.5,
        slippage=0.5,
        brokerage=10.0,
        realized_pnl=0.0,
        strategy="TestStrategy",
        reason="ENTRY",
    )

    # Sell 10 @ 110 with 10 brokerage, slippage 0.5 (execution 109.5) -> closing, gross trading gain = 100
    t2 = TradeBlotterEntry(
        timestamp=ts,
        symbol="NIFTY50",
        side="SELL",
        quantity=10,
        signal_price=110.0,
        execution_price=109.5,
        slippage=-0.5,
        brokerage=10.0,
        realized_pnl=100.0,
        strategy="TestStrategy",
        reason="EXIT",
    )

    blotter = TradeBlotter([t1, t2])

    assert blotter.total_trades == 2
    assert blotter.closed_trades_count == 1
    assert blotter.winning_trades_count == 1
    assert blotter.losing_trades_count == 0

    # Key accounting invariant:
    assert blotter.gross_realized_pnl == 100.0
    assert blotter.total_brokerage == 20.0
    assert blotter.net_pnl == 80.0  # 100 - 20 = 80 (brokerage not double counted)

    # Informational slippage impact: 10 * 0.5 + 10 * 0.5 = 10.0
    assert blotter.total_slippage_cost == 10.0


def test_trade_blotter_empty_and_zero_cases() -> None:
    """Verify blotter metrics on edge cases: empty, only entries, zero losses."""
    # 1. Empty blotter
    empty_blotter = TradeBlotter()
    assert empty_blotter.total_trades == 0
    assert empty_blotter.closed_trades_count == 0
    assert empty_blotter.win_rate == 0.0
    assert empty_blotter.profit_factor == 0.0
    assert empty_blotter.gross_realized_pnl == 0.0
    assert empty_blotter.total_brokerage == 0.0
    assert empty_blotter.net_pnl == 0.0
    assert empty_blotter.format_table() == "(Trade blotter is empty)"
    assert empty_blotter.to_dict() == []
    df_empty = empty_blotter.to_dataframe()
    assert isinstance(df_empty, pd.DataFrame)
    assert len(df_empty) == 0

    # 2. Only entry trades (all realized_pnl == 0.0)
    ts = datetime(2023, 1, 2)
    entry_only = TradeBlotter(
        [
            TradeBlotterEntry(ts, "NIFTY50", "BUY", 10, 100.0, 100.0, 0.0, 10.0, 0.0),
            TradeBlotterEntry(ts, "NIFTY50", "BUY", 10, 105.0, 105.0, 0.0, 10.0, 0.0),
        ]
    )
    assert entry_only.total_trades == 2
    assert entry_only.closed_trades_count == 0
    # Win rate should not raise division by zero
    assert entry_only.win_rate == 0.0
    assert entry_only.profit_factor == 0.0
    assert entry_only.gross_realized_pnl == 0.0
    assert entry_only.total_brokerage == 20.0
    assert entry_only.net_pnl == -20.0

    # 3. Profitable trades with zero losses (no losing trades)
    profitable = TradeBlotter(
        [
            TradeBlotterEntry(ts, "NIFTY50", "SELL", 10, 110.0, 110.0, 0.0, 10.0, 50.0),
        ]
    )
    assert profitable.closed_trades_count == 1
    assert profitable.winning_trades_count == 1
    assert profitable.losing_trades_count == 0
    assert profitable.win_rate == 1.0
    assert profitable.profit_factor == float("inf")

    # 4. Losing trades with zero wins (no winning trades)
    losing = TradeBlotter(
        [
            TradeBlotterEntry(ts, "NIFTY50", "SELL", 10, 90.0, 90.0, 0.0, 10.0, -50.0),
        ]
    )
    assert losing.closed_trades_count == 1
    assert losing.winning_trades_count == 0
    assert losing.losing_trades_count == 1
    assert losing.win_rate == 0.0
    assert losing.profit_factor == 0.0



def test_trade_blotter_formatting_and_exports() -> None:
    """Verify DataFrame export, dict export, and table formatting."""
    ts = datetime(2023, 1, 2, 9, 30)
    blotter = TradeBlotter(
        [
            TradeBlotterEntry(ts, "NIFTY50", "BUY", 10, 100.0, 100.5, 0.5, 15.0, 0.0, "ATRGrid", "LEVEL_1"),
            TradeBlotterEntry(ts, "NIFTY50", "SELL", 10, 110.0, 109.5, -0.5, 15.0, 90.0, "ATRGrid", "LEVEL_2"),
        ]
    )

    df = blotter.to_dataframe()
    assert len(df) == 2
    assert list(df.columns) == [
        "timestamp",
        "symbol",
        "side",
        "quantity",
        "signal_price",
        "execution_price",
        "slippage",
        "brokerage",
        "realized_pnl",
        "strategy",
        "reason",
    ]

    table = blotter.format_table()
    assert "NIFTY50" in table
    assert "ATRGrid" in table
    assert "LEVEL_1" in table
    assert "+90.00" in table

    summary_text = blotter.summary()
    assert "Win Rate:" in summary_text
    assert "Net PnL:" in summary_text


# ─── 2. BacktestEngine Integration Tests ──────────────────────────────────────


def test_backtest_result_blotter_integration() -> None:
    """
    Run an actual backtest and verify:
        - BacktestResult.blotter is populated.
        - BacktestResult.blotter is cached across repeated accesses.
        - Blotter entries have signal_price, strategy, and reason properly assigned.
    """
    from vega.data.models import Bar

    # Construct synthetic bars causing SAR to trade
    bars = [
        Bar(datetime(2023, 1, 2), 100.0, 105.0, 95.0, 102.0, 1000.0),
        Bar(datetime(2023, 1, 3), 102.0, 115.0, 101.0, 112.0, 1500.0),
        Bar(datetime(2023, 1, 4), 112.0, 114.0, 85.0, 88.0, 2000.0),
        Bar(datetime(2023, 1, 5), 88.0, 90.0, 80.0, 82.0, 1200.0),
    ]

    strategy = StopAndReverseStrategy(fast_period=2, slow_period=3, symbol="NIFTY50")
    engine = BacktestEngine(strategy=strategy)
    result = engine.run(bars)

    assert isinstance(result, BacktestResult)
    # Cached property check
    blotter1 = result.blotter
    blotter2 = result.blotter
    assert blotter1 is blotter2
    assert isinstance(blotter1, TradeBlotter)

    if blotter1.total_trades > 0:
        first_trade = blotter1.entries[0]
        assert first_trade.strategy == "StopAndReverseStrategy"
        assert first_trade.reason != ""
        assert first_trade.signal_price is not None


# ─── 3. Structured Logger & Alerting Tests ────────────────────────────────────


def test_structured_logger_json_schema() -> None:
    """Verify structured log events produce valid JSON matching the fixed schema."""
    fixed_time = datetime(2023, 1, 2, 9, 30, 0, tzinfo=timezone.utc)
    logger = StructuredLogger(clock_fn=lambda: fixed_time)

    event = logger.log_fill(
        order_id="ORD-001",
        symbol="NIFTY50",
        side="BUY",
        quantity=50,
        price=18000.0,
        brokerage=20.0,
        slippage=1.5,
        realized_pnl=0.0,
        strategy="ATRGrid",
    )

    json_str = event.to_json()
    parsed = json.loads(json_str)

    assert parsed["timestamp"] == fixed_time.isoformat()
    assert parsed["event_type"] == "FILL"
    assert parsed["level"] == "INFO"
    assert "Fill executed" in parsed["message"]
    assert parsed["data"]["order_id"] == "ORD-001"
    assert parsed["data"]["symbol"] == "NIFTY50"
    assert parsed["data"]["price"] == 18000.0


def test_logging_not_equal_to_alerting_invariant() -> None:
    """
    Critical invariant: Logging != Alerting.
        - Routine events (ORDER, FILL, POSITION_CHANGE, PNL_UPDATE) are logged ONLY.
        - Critical events (KILL_SWITCH, CRITICAL error, CRITICAL alert) trigger alert callback.
    """
    alerts_received: List[LogEvent] = []

    def on_alert(event: LogEvent) -> None:
        alerts_received.append(event)

    logger = StructuredLogger(on_alert=on_alert)

    # 1. Routine events must NOT trigger alert
    logger.log_order("ORD-1", "NIFTY50", "BUY", 10, 100.0, "PENDING")
    logger.log_fill("ORD-1", "NIFTY50", "BUY", 10, 100.0, 10.0, 0.0, 0.0)
    logger.log_position_change("NIFTY50", 0, 10, 100.0)
    logger.log_pnl_update(100000.0, 99000.0, 0.0, 0.0, 0.0, 0.0)
    logger.log_alert("Minor memory usage notice", level="WARNING")
    logger.log_error("Non-critical lookup warning", is_critical=False)

    assert len(logger.events) == 6
    # Zero alerts triggered so far
    assert len(alerts_received) == 0
    assert len(logger.alert_events) == 0

    # 2. Kill Switch MUST trigger alert immediately
    logger.log_kill_switch("Drawdown exceeded maximum allowed limit 20%", triggered_by="risk_manager")

    assert len(alerts_received) == 1
    assert alerts_received[0].event_type == LogEventType.KILL_SWITCH.value
    assert alerts_received[0].level == "CRITICAL"
    assert "KILL SWITCH ACTIVATED" in alerts_received[0].message

    # 3. Critical Error MUST trigger alert
    logger.log_error("Broker socket severed with unrecoverable handshake failure", is_critical=True)

    assert len(alerts_received) == 2
    assert alerts_received[1].event_type == LogEventType.ERROR.value
    assert alerts_received[1].level == "CRITICAL"

    # 4. Critical Alert MUST trigger alert
    logger.log_alert("India VIX jumped above circuit breaker threshold 24.0", level="CRITICAL")

    assert len(alerts_received) == 3
    assert alerts_received[2].event_type == LogEventType.ALERT.value
    assert alerts_received[2].level == "CRITICAL"
