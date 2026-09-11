"""
Backtesting and execution engine package for Vega Quant Trading Engine.

Exports:
    - BacktestEngine: Deterministic historical bar-by-bar orchestrator
    - BacktestResult: Output container with metrics, equity curve, orders, and trades
    - EquityPoint: Snapshot of equity and drawdown at a specific bar
    - TradeRecord: Executed trade blotter entry
    - OrderRecord: Audit record for an order generated during backtesting
"""

from vega.engine.backtest import (
    BacktestEngine,
    BacktestResult,
    EquityPoint,
    OrderRecord,
    TradeRecord,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "EquityPoint",
    "TradeRecord",
    "OrderRecord",
]
