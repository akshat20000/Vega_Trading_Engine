"""
Trade Blotter for the Vega Quant Trading Engine.

Provides:
    - TradeBlotterEntry: Standardized trade execution record.
    - TradeBlotter: Container for trade records with performance metrics,
      ASCII/Markdown table formatting, and DataFrame export.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class TradeBlotterEntry:
    """
    A single executed trade record within the Trade Blotter.

    Attributes:
        timestamp: Execution timestamp (e.g. t+1 OPEN).
        symbol: Ticker symbol (e.g. 'NIFTY50').
        side: Direction ('BUY' or 'SELL').
        quantity: Executed quantity in units.
        signal_price: Market price when the trading signal was generated (or None).
        execution_price: Actual fill price including slippage.
        slippage: Per-unit slippage applied by broker.
        brokerage: Commission / fee charged for this execution.
        realized_pnl: Realized P&L produced by this fill (0.0 for position openings).
        strategy: Name of the strategy generating the signal.
        reason: Signal rationale or order generation rule.
    """

    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    signal_price: float | None
    execution_price: float
    slippage: float
    brokerage: float
    realized_pnl: float
    strategy: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary with ISO-formatted timestamp."""
        return {
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "signal_price": self.signal_price,
            "execution_price": self.execution_price,
            "slippage": self.slippage,
            "brokerage": self.brokerage,
            "realized_pnl": self.realized_pnl,
            "strategy": self.strategy,
            "reason": self.reason,
        }


class TradeBlotter:
    """
    Collection of TradeBlotterEntry records with accounting and summary metrics.

    Accounting Semantics:
        - Entry fills have realized_pnl = 0.0.
        - Closing / reducing fills reflect actual portfolio realized P&L.
        - Gross Realized P&L = sum(realized_pnl)
        - Total Brokerage = sum(brokerage)
        - Net P&L = Gross Realized P&L - Total Brokerage
        - Slippage is already factored into execution_price; total_slippage_cost
          is reported as informational impact and is NOT deducted a second time.
    """

    def __init__(self, entries: Sequence[TradeBlotterEntry] | None = None) -> None:
        self.entries: list[TradeBlotterEntry] = list(entries or [])

    def add_entry(self, entry: TradeBlotterEntry) -> None:
        """Append a new trade record to the blotter."""
        self.entries.append(entry)

    @classmethod
    def from_trades(
        cls,
        trades: Sequence[Any],
        default_strategy: str = "",
    ) -> TradeBlotter:
        """
        Construct a TradeBlotter from BacktestEngine TradeRecord items.
        """
        entries: list[TradeBlotterEntry] = []
        for t in trades:
            side_str = t.side.value if hasattr(t.side, "value") else str(t.side)
            entry = TradeBlotterEntry(
                timestamp=t.execution_time,
                symbol=t.symbol,
                side=side_str,
                quantity=t.quantity,
                signal_price=getattr(t, "signal_price", None),
                execution_price=t.execution_price,
                slippage=t.slippage,
                brokerage=t.brokerage,
                realized_pnl=t.realized_pnl,
                strategy=getattr(t, "strategy", default_strategy) or default_strategy,
                reason=getattr(t, "reason", "") or "",
            )
            entries.append(entry)
        return cls(entries)

    # ─── Accounting Metrics ───────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        """Total number of executed orders on the blotter."""
        return len(self.entries)

    @property
    def gross_realized_pnl(self) -> float:
        """Gross cumulative realized trading P&L across all closed positions."""
        return sum(e.realized_pnl for e in self.entries)

    @property
    def total_brokerage(self) -> float:
        """Total transaction costs / broker commissions paid."""
        return sum(e.brokerage for e in self.entries)

    @property
    def net_pnl(self) -> float:
        """
        Net economic P&L: Gross Realized P&L minus Total Brokerage.
        Slippage is not deducted here because it is already embedded in execution prices.
        """
        return self.gross_realized_pnl - self.total_brokerage

    @property
    def total_slippage_cost(self) -> float:
        """
        Cumulative informational slippage impact: sum(abs(slippage) * quantity).
        Note: Already reflected in execution_price; not subtracted again from net_pnl.
        """
        return sum(abs(e.slippage) * e.quantity for e in self.entries)

    @property
    def closed_trades_count(self) -> int:
        """Number of trades with non-zero realized P&L (position reductions / closes)."""
        return sum(1 for e in self.entries if e.realized_pnl != 0.0)

    @property
    def winning_trades_count(self) -> int:
        """Number of trades with strictly positive realized P&L."""
        return sum(1 for e in self.entries if e.realized_pnl > 0.0)

    @property
    def losing_trades_count(self) -> int:
        """Number of trades with strictly negative realized P&L."""
        return sum(1 for e in self.entries if e.realized_pnl < 0.0)

    @property
    def win_rate(self) -> float:
        """
        Proportion of winning closed trades: winning_trades / closed_trades.
        Returns 0.0 if no closed trades exist (avoids distorting with zero-P&L entry trades).
        """
        if self.closed_trades_count == 0:
            return 0.0
        return self.winning_trades_count / self.closed_trades_count

    @property
    def profit_factor(self) -> float:
        """
        Ratio of gross profits to gross losses: gross_profits / abs(gross_losses).
        Returns float('inf') if profitable with zero losses, 0.0 if no trades.
        """
        gross_profits = sum(e.realized_pnl for e in self.entries if e.realized_pnl > 0.0)
        gross_losses = abs(sum(e.realized_pnl for e in self.entries if e.realized_pnl < 0.0))

        if gross_losses == 0.0:
            return float("inf") if gross_profits > 0.0 else 0.0
        return gross_profits / gross_losses

    # ─── Serialization & Formatting ───────────────────────────────────────────

    def to_dict(self) -> list[dict[str, Any]]:
        """Export blotter entries as a list of dictionaries."""
        return [e.to_dict() for e in self.entries]

    def to_dataframe(self) -> pd.DataFrame:
        """Export blotter as a pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame(
                columns=[
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
            )
        return pd.DataFrame(self.to_dict())

    def format_table(self, max_rows: int = 50) -> str:
        """Format the trade blotter as a clean ASCII text table."""
        if not self.entries:
            return "(Trade blotter is empty)"

        headers = [
            "Timestamp",
            "Symbol",
            "Side",
            "Qty",
            "Sig Price",
            "Exec Price",
            "Slippage",
            "Brokerage",
            "Realized PnL",
            "Strategy",
            "Reason",
        ]

        rows: list[list[str]] = []
        for e in self.entries[:max_rows]:
            ts_str = e.timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(e.timestamp, datetime) else str(e.timestamp)
            sig_str = f"{e.signal_price:.2f}" if e.signal_price is not None else "-"
            rows.append(
                [
                    ts_str,
                    e.symbol,
                    e.side,
                    str(e.quantity),
                    sig_str,
                    f"{e.execution_price:.2f}",
                    f"{e.slippage:.2f}",
                    f"{e.brokerage:.2f}",
                    f"{e.realized_pnl:+.2f}",
                    e.strategy,
                    e.reason,
                ]
            )

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        row_lines = [" | ".join(row[i].ljust(col_widths[i]) for i in range(len(headers))) for row in rows]

        table = "\n".join([header_line, sep_line] + row_lines)
        if len(self.entries) > max_rows:
            table += f"\n... ({len(self.entries) - max_rows} additional trades omitted)"
        return table

    def summary(self) -> str:
        """Return a formatted string summary of blotter trading metrics."""
        return (
            f"Total Trades:       {self.total_trades}\n"
            f"Closed Trades:      {self.closed_trades_count}\n"
            f"Winning / Losing:   {self.winning_trades_count} / {self.losing_trades_count}\n"
            f"Win Rate:           {self.win_rate * 100:.1f}%\n"
            f"Profit Factor:      {self.profit_factor:.2f}\n"
            f"Gross Realized PnL: ₹{self.gross_realized_pnl:,.2f}\n"
            f"Total Brokerage:    ₹{self.total_brokerage:,.2f}\n"
            f"Net PnL:            ₹{self.net_pnl:,.2f}\n"
            f"Total Slippage:     ₹{self.total_slippage_cost:,.2f}"
        )
