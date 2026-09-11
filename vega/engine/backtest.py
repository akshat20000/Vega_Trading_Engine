"""
Backtest Engine for the Vega Quant Trading Engine.

Acts as the central, deterministic historical orchestrator coordinating:
    Market Data -> Indicators / Strategy -> Order -> RiskManager -> PaperBroker -> Fill -> Portfolio

Execution Model & No-Lookahead Guarantees:
    - Signal generation at bar t uses data <= t.
    - Orders generated at bar t remain PENDING.
    - Execution strictly occurs at bar t+1 OPEN at bar[t+1].open price.
    - PaperBroker applies configured slippage and brokerage models.
    - Generated Fills are dispatched to Portfolio (accounting authority)
      and Strategy (via on_fill callback).
    - Portfolio is marked to market at each bar's CLOSE.
    - At the final bar: pending orders from the previous bar are executed at final OPEN,
      final-bar signals are evaluated and approved orders placed as pending,
      portfolio is marked to market at final CLOSE, and all remaining pending orders
      are CANCELLED. Orders generated on the final bar are NEVER filled.

Risk Orchestration:
    - Maintains local projected position and projected pyramid state during
      same-bar batch order validation to prevent collective breaches of
      position cap or pyramiding limits.
    - Projected state is used ONLY for sequential RiskManager checks and
      never mutates Portfolio or Strategy state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Sequence

from vega.broker.paper import PaperBroker
from vega.config import VegaConfig
from vega.data.models import Bar
from vega.macro.models import MacroSnapshot, Regime
from vega.orders.models import Fill, Order, OrderSide, OrderStatus
from vega.portfolio.portfolio import Portfolio
from vega.risk.manager import OrderEffect, RiskManager
from vega.strategy.base import BaseStrategy


@dataclass(frozen=True)
class EquityPoint:
    """
    A single snapshot of portfolio equity and drawdown at the end of a bar.

    Attributes:
        timestamp: Time of the evaluated bar.
        equity: Total portfolio equity in INR.
        cash: Current available cash in INR.
        drawdown: Current absolute drawdown from peak equity in INR.
        drawdown_pct: Current drawdown percentage from peak equity.
    """

    timestamp: datetime
    equity: float
    cash: float
    drawdown: float
    drawdown_pct: float


@dataclass
class TradeRecord:
    """
    Execution record for a filled order (Trade Blotter entry).

    Attributes:
        order_id: The client_order_id of the executed order.
        symbol: Ticker symbol traded.
        side: OrderSide (BUY or SELL).
        quantity: Executed units/lots.
        signal_time: Timestamp of the bar where the order was generated.
        execution_time: Timestamp of execution (t+1 OPEN).
        execution_price: Execution price after slippage.
        brokerage: Commission charged by broker.
        slippage: Absolute slippage per unit applied by broker.
        realized_pnl: Realized P&L produced by this fill (from Portfolio).
    """

    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    signal_time: datetime
    execution_time: datetime
    execution_price: float
    brokerage: float
    slippage: float
    realized_pnl: float


@dataclass
class OrderRecord:
    """
    Audit log entry for an order generated during backtest.

    Attributes:
        client_order_id: Idempotency identifier.
        symbol: Instrument symbol.
        side: OrderSide.
        quantity: Units.
        price: Proposed order price (0.0 for market).
        signal_time: Bar timestamp where order was created.
        status: Status ("PENDING", "FILLED", "CANCELLED", "REJECTED").
        rejection_reason: Reason if rejected or cancelled.
    """

    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    signal_time: datetime
    status: str
    rejection_reason: str = ""


@dataclass
class BacktestResult:
    """
    Comprehensive output of a backtest run.

    Attributes:
        initial_cash: Starting capital in INR.
        final_equity: Ending portfolio equity in INR.
        total_return: Total return percentage ((final_equity - initial_cash) / initial_cash * 100).
        realized_pnl: Net cumulative trading P&L in INR.
        unrealized_pnl: Open mark-to-market P&L at the end of backtest.
        total_brokerage: Total transaction fees paid in INR.
        max_drawdown: Maximum observed absolute drawdown in INR.
        max_drawdown_pct: Maximum observed drawdown percentage.
        number_of_orders: Total orders generated across the backtest.
        number_of_fills: Total filled trades executed.
        number_of_rejected_orders: Total orders blocked by risk controls.
        equity_curve: Chronological list of EquityPoint records for each bar.
        trades: List of executed TradeRecord items (trade blotter).
        orders: Complete order audit log.
        rejected_orders: List of tuples (Order, rejection_reason).
    """

    initial_cash: float
    final_equity: float
    total_return: float
    realized_pnl: float
    unrealized_pnl: float
    total_brokerage: float
    max_drawdown: float
    max_drawdown_pct: float
    number_of_orders: int
    number_of_fills: int
    number_of_rejected_orders: int
    equity_curve: list[EquityPoint] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    orders: list[OrderRecord] = field(default_factory=list)
    rejected_orders: list[tuple[Order, str]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted text summary of backtest performance."""
        return (
            f"=== Backtest Performance Summary ===\n"
            f"Initial Capital:     {self.initial_cash:,.2f} INR\n"
            f"Final Equity:        {self.final_equity:,.2f} INR\n"
            f"Total Return:        {self.total_return:+.2f}%\n"
            f"Realized P&L:        {self.realized_pnl:+,.2f} INR\n"
            f"Unrealized P&L:      {self.unrealized_pnl:+,.2f} INR\n"
            f"Total Brokerage:     {self.total_brokerage:,.2f} INR\n"
            f"Max Drawdown:        {self.max_drawdown:,.2f} INR ({self.max_drawdown_pct:.2f}%)\n"
            f"Total Orders:        {self.number_of_orders}\n"
            f"Total Fills:         {self.number_of_fills}\n"
            f"Rejected Orders:     {self.number_of_rejected_orders}\n"
        )


class BacktestEngine:
    """
    Deterministic Historical Backtest Engine.

    Coordinates market data, strategies, pre-order risk validation,
    paper execution, and average-cost portfolio accounting.
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        broker: PaperBroker | None = None,
        risk_manager: RiskManager | None = None,
        portfolio: Portfolio | None = None,
        config: VegaConfig | None = None,
        initial_cash: float = 1_000_000.0,
        macro_snapshots: Sequence[MacroSnapshot] | None = None,
    ) -> None:
        """
        Initialize the Backtest Engine.

        Args:
            strategy: Trading strategy conforming to BaseStrategy interface.
            broker: Optional PaperBroker instance. If None, created from config.
            risk_manager: Optional RiskManager instance. If None, created from config.
            portfolio: Optional Portfolio instance. If None, initialized with initial_cash.
            config: Optional VegaConfig object.
            initial_cash: Starting cash balance if portfolio is not provided.
            macro_snapshots: Optional sequence of MacroSnapshot objects for regime awareness.
        """
        self.strategy: BaseStrategy = strategy
        self.config: VegaConfig | None = config
        self.broker: PaperBroker = broker if broker is not None else PaperBroker(config=config)
        self.risk_manager: RiskManager = (
            risk_manager if risk_manager is not None else RiskManager(config=config)
        )
        self.portfolio: Portfolio = (
            portfolio if portfolio is not None else Portfolio(initial_cash=initial_cash)
        )
        self.macro_snapshots: Sequence[MacroSnapshot] | None = macro_snapshots

        # Pre-index macro snapshots by date if provided
        self._macro_by_date: dict[date, MacroSnapshot] = {}
        if self.macro_snapshots:
            for snap in self.macro_snapshots:
                s_date = snap.date if isinstance(snap.date, date) else snap.date.date()
                self._macro_by_date[s_date] = snap

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        """
        Execute the historical backtest across a sequence of bars.

        Strict Bar-by-Bar Sequencing:
            For each bar t in bars:
                1. At bar t OPEN: execute pending orders from bar t-1.
                   Fills update Portfolio accounting and call strategy.on_fill(fill).
                2. At bar t: evaluate strategy.on_bar(bar, regime=regime).
                3. Validate batch of orders sequentially against RiskManager
                   maintaining local projected position and pyramid count.
                4. Approved orders placed with PaperBroker as PENDING.
                5. At bar t CLOSE: mark portfolio to market and record EquityPoint.
                6. At final bar N-1: cancel all remaining pending orders.
                   Orders generated on the final bar are NEVER filled.

        Args:
            bars: Chronological sequence of Bar objects.

        Returns:
            BacktestResult object containing performance metrics, equity curve,
            orders, and trade blotter.
        """
        initial_cash = float(self.portfolio.get_initial_cash())

        # 1. Handle empty data safely
        if not bars:
            return BacktestResult(
                initial_cash=initial_cash,
                final_equity=initial_cash,
                total_return=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_brokerage=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                number_of_orders=0,
                number_of_fills=0,
                number_of_rejected_orders=0,
            )

        trades: list[TradeRecord] = []
        order_records: list[OrderRecord] = []
        rejected_orders: list[tuple[Order, str]] = []
        equity_curve: list[EquityPoint] = []
        order_signal_times: dict[str, datetime] = {}

        n_bars = len(bars)

        for idx, bar in enumerate(bars):
            is_final_bar = (idx == n_bars - 1)

            # ── Step 1: Execute Pending Orders from Previous Bar at Bar t OPEN ──
            if idx > 0:
                # Execute strictly at t OPEN (bar.open) with PaperBroker costs
                fills = self.broker.execute_pending_orders(
                    market_price=bar.open,
                    timestamp=bar.timestamp,
                )
                for fill in fills:
                    # Portfolio is the accounting authority
                    realized_pnl = self.portfolio.process_fill(fill)
                    # Strategy updates filled position tracking
                    self.strategy.on_fill(fill)

                    sig_time = order_signal_times.get(fill.order_id, bar.timestamp)
                    trade = TradeRecord(
                        order_id=fill.order_id,
                        symbol=fill.symbol,
                        side=fill.side,
                        quantity=fill.filled_qty,
                        signal_time=sig_time,
                        execution_time=fill.timestamp,
                        execution_price=fill.filled_price,
                        brokerage=fill.brokerage,
                        slippage=fill.slippage,
                        realized_pnl=float(realized_pnl),
                    )
                    trades.append(trade)

                    # Update order record status
                    for o_rec in order_records:
                        if o_rec.client_order_id == fill.order_id:
                            o_rec.status = "FILLED"

            # ── Step 2: Extract Market & Macro Regime Information at Bar t ─────
            b_date = bar.timestamp.date() if isinstance(bar.timestamp, datetime) else bar.timestamp
            macro_snap = self._macro_by_date.get(b_date)
            regime = macro_snap.regime if macro_snap is not None else None
            circuit_breaker = False
            if macro_snap is not None:
                # Circuit breaker triggers when BEARISH and India VIX exceeds threshold (20.0)
                circuit_breaker = bool(macro_snap.regime == Regime.BEARISH and macro_snap.india_vix > 20.0)

            # ── Step 3: Strategy Evaluates Bar t ──────────────────────────────
            candidate_orders = self.strategy.on_bar(bar, regime=regime)

            # ── Step 4: Batch Risk Validation with Projected Risk State ───────
            # Review Correction 1:
            # Maintain local projected position and projected pyramid state while
            # processing a batch of strategy orders so multiple orders cannot
            # collectively breach position cap or pyramiding limits.
            actual_pos = self.portfolio.get_position(self.strategy.symbol).quantity
            actual_pyramids = self.strategy.get_pyramid_count()
            daily_pnl = float(self.portfolio.get_daily_realized_pnl(b_date))

            projected_pos = actual_pos
            projected_pyramids = actual_pyramids

            for order in candidate_orders:
                order_signal_times[order.client_order_id] = bar.timestamp

                decision = self.risk_manager.validate_order(
                    order=order,
                    current_position=projected_pos,
                    current_pyramids=projected_pyramids,
                    daily_realized_pnl=daily_pnl,
                    macro_circuit_breaker=circuit_breaker,
                )

                if not decision.is_allowed:
                    # Order rejected by risk gate
                    order.mark_rejected(reason=decision.reason)
                    order_records.append(
                        OrderRecord(
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            price=order.price,
                            signal_time=bar.timestamp,
                            status="REJECTED",
                            rejection_reason=decision.reason,
                        )
                    )
                    rejected_orders.append((order, decision.reason))
                    continue

                # Approved: Submit to PaperBroker as PENDING
                placed_order = self.broker.place_order(order)
                order_records.append(
                    OrderRecord(
                        client_order_id=placed_order.client_order_id,
                        symbol=placed_order.symbol,
                        side=placed_order.side,
                        quantity=placed_order.quantity,
                        price=placed_order.price,
                        signal_time=bar.timestamp,
                        status="PENDING",
                    )
                )

                # Update local projected risk state for subsequent orders in this batch
                if order.side == OrderSide.BUY:
                    projected_pos += order.quantity
                else:
                    projected_pos -= order.quantity

                # Update projected pyramid count based on classified order effect
                if decision.order_effect in (OrderEffect.NEW_ENTRY, OrderEffect.REVERSAL):
                    projected_pyramids = 1
                elif decision.order_effect == OrderEffect.SAME_DIRECTION_ENTRY:
                    projected_pyramids += 1
                elif decision.order_effect == OrderEffect.COMPLETE_REDUCTION:
                    projected_pyramids = 0

            # ── Step 5: Mark to Market at Bar t CLOSE & Record Equity ─────────
            self.portfolio.mark_to_market(self.strategy.symbol, bar.close)
            eq_pt = EquityPoint(
                timestamp=bar.timestamp,
                equity=float(self.portfolio.get_equity()),
                cash=float(self.portfolio.get_cash()),
                drawdown=float(self.portfolio.get_drawdown()),
                drawdown_pct=float(self.portfolio.get_drawdown_pct()),
            )
            equity_curve.append(eq_pt)

            # ── Step 6: End-Of-Data Handling on Final Bar ─────────────────────
            # Review Correction 7:
            # On final bar, cancel all remaining pending orders; never fill them.
            if is_final_bar:
                for pending_order in self.broker.get_all_orders():
                    if pending_order.status == OrderStatus.PENDING:
                        cancelled = self.broker.cancel_order(pending_order.client_order_id)
                        for o_rec in order_records:
                            if o_rec.client_order_id == cancelled.client_order_id:
                                o_rec.status = "CANCELLED"
                                o_rec.rejection_reason = "Cancelled at end of data (no t+1 bar)"

        # Calculate final summary metrics
        final_equity = float(self.portfolio.get_equity())
        total_ret = ((final_equity - initial_cash) / initial_cash * 100.0) if initial_cash > 0 else 0.0

        # Max drawdown across equity curve
        max_dd = max((pt.drawdown for pt in equity_curve), default=float(self.portfolio.get_drawdown()))
        max_dd_pct = max((pt.drawdown_pct for pt in equity_curve), default=float(self.portfolio.get_drawdown_pct()))

        return BacktestResult(
            initial_cash=initial_cash,
            final_equity=final_equity,
            total_return=round(total_ret, 4),
            realized_pnl=float(self.portfolio.get_realized_pnl()),
            unrealized_pnl=float(self.portfolio.get_unrealized_pnl()),
            total_brokerage=float(self.portfolio.get_total_brokerage()),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            number_of_orders=len(order_records),
            number_of_fills=len(trades),
            number_of_rejected_orders=len(rejected_orders),
            equity_curve=equity_curve,
            trades=trades,
            orders=order_records,
            rejected_orders=rejected_orders,
        )
