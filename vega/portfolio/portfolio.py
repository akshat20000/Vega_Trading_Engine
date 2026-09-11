"""
Portfolio and Accounting Layer for the Vega Quant Trading Engine.

Tracks per-symbol positions, cash balances, transaction costs, and portfolio-level
metrics (equity, peak equity, drawdown, daily realized P&L) from execution Fill events.

Accounting Principles:
    - Exact average-cost accounting for long and short positions.
    - Cash adjusts directly for execution value and brokerage:
        BUY:  cash decreases by (fill_price * quantity + brokerage)
        SELL: cash increases by (fill_price * quantity - brokerage)
    - Reversals (crossing through zero) first close the existing position,
      realizing its P&L, then establish the residual position at the new fill price.
    - Mark-to-market revaluation computes unrealized P&L and updates portfolio equity.
    - All monetary values are maintained in Decimal with 2-decimal banker's rounding (ROUND_HALF_EVEN).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_EVEN

from vega.orders.models import Fill, OrderSide

ROUND_PAISE = Decimal("0.01")


def to_decimal(val: float | int | str | Decimal) -> Decimal:
    """
    Safely convert numeric values to Decimal.

    Converts incoming floats via str representation (`Decimal(str(val))`)
    to prevent binary floating-point representation artifacts.
    """
    if isinstance(val, Decimal):
        return val
    return Decimal(str(val))


def round_money(val: Decimal) -> Decimal:
    """Round a Decimal value to 2 decimal places using banker's rounding (ROUND_HALF_EVEN)."""
    return val.quantize(ROUND_PAISE, rounding=ROUND_HALF_EVEN)


@dataclass
class Position:
    """
    State of an open or closed position for a single symbol.

    Attributes:
        symbol: Instrument identifier (e.g. 'NIFTY50').
        quantity: Net position in units/lots (> 0 for LONG, < 0 for SHORT, 0 for FLAT).
        average_entry_price: Average cost basis per unit for currently held position.
        realized_pnl: Cumulative realized trading P&L for this symbol.
        unrealized_pnl: Current unrealized P&L based on the latest market price.
        total_quantity_traded: Total cumulative volume (units/lots) traded for this symbol.
    """

    symbol: str
    quantity: int = 0
    average_entry_price: Decimal = Decimal("0.00")
    realized_pnl: Decimal = Decimal("0.00")
    unrealized_pnl: Decimal = Decimal("0.00")
    total_quantity_traded: int = 0

    @property
    def is_long(self) -> bool:
        """Return True if position is net long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Return True if position is net short."""
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        """Return True if position is flat (quantity == 0)."""
        return self.quantity == 0


class Portfolio:
    """
    Portfolio accounting engine.

    Responsibilities:
        1. Maintain cash balance with deterministic deduction/addition of transaction costs.
        2. Average-cost accounting for opening, increasing, reducing, closing, and reversing positions.
        3. Mark-to-market valuations and unrealized P&L calculation.
        4. Real-time equity, peak equity, and drawdown percentage tracking.
        5. Daily realized P&L tracking for risk management and performance evaluation.
        6. Multi-symbol support with strict Decimal monetary precision.
    """

    def __init__(self, initial_cash: float | Decimal | str = 1_000_000.0) -> None:
        """
        Initialize Portfolio with initial cash balance.

        Args:
            initial_cash: Starting capital in INR. Defaults to 10,00,000 INR (10 lakhs).

        Raises:
            ValueError: If initial_cash is negative.
        """
        init_dec = to_decimal(initial_cash)
        if init_dec < Decimal("0.00"):
            raise ValueError(f"initial_cash must be non-negative, got {initial_cash}")

        self._initial_cash: Decimal = round_money(init_dec)
        self._cash: Decimal = self._initial_cash
        self._peak_equity: Decimal = self._initial_cash
        self._drawdown: Decimal = Decimal("0.00")
        self._drawdown_pct: Decimal = Decimal("0.00")

        # Internal collections
        self._positions: dict[str, Position] = {}
        self._latest_market_prices: dict[str, Decimal] = {}
        self._daily_realized_pnl: dict[date, Decimal] = {}
        self._latest_trading_date: date | None = None
        self._total_brokerage: Decimal = Decimal("0.00")

    # ── Fill Processing & Average-Cost Accounting ─────────────────────────────

    def process_fill(
        self,
        fill: Fill,
        symbol: str | None = None,
        side: OrderSide | None = None,
    ) -> Decimal:
        """
        Process an execution Fill and update portfolio state.

        Args:
            fill: The execution Fill object.
            symbol: Optional symbol override (defaults to fill.symbol).
            side: Optional OrderSide override (defaults to fill.side).

        Returns:
            Decimal: Realized P&L produced by this fill (0.00 if opening or adding).

        Raises:
            TypeError: If fill is not an instance of Fill.
            ValueError: If symbol or side cannot be determined, or fill parameters are invalid.
        """
        if not isinstance(fill, Fill):
            raise TypeError(f"fill must be an instance of Fill, got {type(fill).__name__}")

        resolved_symbol = symbol or getattr(fill, "symbol", "")
        if not resolved_symbol or not isinstance(resolved_symbol, str):
            raise ValueError("Fill must have a non-empty symbol or symbol must be provided.")

        resolved_side = side or getattr(fill, "side", None)
        if not isinstance(resolved_side, OrderSide):
            raise ValueError(f"side must be an OrderSide (BUY/SELL), got {resolved_side}")

        if fill.filled_qty <= 0:
            raise ValueError(f"filled_qty must be positive, got {fill.filled_qty}")

        if fill.filled_price <= 0.0:
            raise ValueError(f"filled_price must be positive, got {fill.filled_price}")

        fill_price = to_decimal(fill.filled_price)
        qty = fill.filled_qty
        brokerage = round_money(to_decimal(fill.brokerage))

        # 1. Update Cash Balance
        # BUY:  cash decreases by (fill_price * quantity + brokerage)
        # SELL: cash increases by (fill_price * quantity - brokerage)
        if resolved_side == OrderSide.BUY:
            cash_outflow = round_money(fill_price * Decimal(qty) + brokerage)
            self._cash = round_money(self._cash - cash_outflow)
        else:
            cash_inflow = round_money(fill_price * Decimal(qty) - brokerage)
            self._cash = round_money(self._cash + cash_inflow)

        self._total_brokerage = round_money(self._total_brokerage + brokerage)

        # 2. Retrieve or initialize position
        pos = self._positions.get(resolved_symbol)
        if pos is None:
            pos = Position(symbol=resolved_symbol)
            self._positions[resolved_symbol] = pos

        pos.total_quantity_traded += qty
        realized_pnl = Decimal("0.00")

        # 3. Average-Cost Accounting Transitions
        if pos.quantity == 0:
            # Open new position from flat
            pos.quantity = qty if resolved_side == OrderSide.BUY else -qty
            pos.average_entry_price = round_money(fill_price)

        elif pos.quantity > 0:
            # Currently LONG
            if resolved_side == OrderSide.BUY:
                # Increasing Long -> update weighted average entry price
                old_qty = Decimal(pos.quantity)
                new_qty = Decimal(qty)
                total_qty = old_qty + new_qty
                weighted_cost = (old_qty * pos.average_entry_price) + (new_qty * fill_price)
                pos.average_entry_price = round_money(weighted_cost / total_qty)
                pos.quantity = int(total_qty)
            else:
                # SELL against Long
                if qty < pos.quantity:
                    # Partial Long Reduction -> average entry price unchanged
                    closed_qty = Decimal(qty)
                    realized_pnl = round_money((fill_price - pos.average_entry_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)
                    pos.quantity -= qty
                elif qty == pos.quantity:
                    # Complete Long Exit -> position becomes flat
                    closed_qty = Decimal(qty)
                    realized_pnl = round_money((fill_price - pos.average_entry_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)
                    pos.quantity = 0
                    pos.average_entry_price = Decimal("0.00")
                    pos.unrealized_pnl = Decimal("0.00")
                else:
                    # Long -> Short Reversal: close long, residual establishes short
                    closed_qty = Decimal(pos.quantity)
                    realized_pnl = round_money((fill_price - pos.average_entry_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)

                    residual_qty = qty - pos.quantity
                    pos.quantity = -residual_qty
                    pos.average_entry_price = round_money(fill_price)

        else:
            # Currently SHORT (pos.quantity < 0)
            current_abs = abs(pos.quantity)
            if resolved_side == OrderSide.SELL:
                # Increasing Short -> update weighted average entry price
                old_abs = Decimal(current_abs)
                new_qty = Decimal(qty)
                total_abs = old_abs + new_qty
                weighted_cost = (old_abs * pos.average_entry_price) + (new_qty * fill_price)
                pos.average_entry_price = round_money(weighted_cost / total_abs)
                pos.quantity = -int(total_abs)
            else:
                # BUY against Short
                if qty < current_abs:
                    # Partial Short Reduction -> average entry price unchanged
                    closed_qty = Decimal(qty)
                    realized_pnl = round_money((pos.average_entry_price - fill_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)
                    pos.quantity = -(current_abs - qty)
                elif qty == current_abs:
                    # Complete Short Exit -> position becomes flat
                    closed_qty = Decimal(qty)
                    realized_pnl = round_money((pos.average_entry_price - fill_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)
                    pos.quantity = 0
                    pos.average_entry_price = Decimal("0.00")
                    pos.unrealized_pnl = Decimal("0.00")
                else:
                    # Short -> Long Reversal: close short, residual establishes long
                    closed_qty = Decimal(current_abs)
                    realized_pnl = round_money((pos.average_entry_price - fill_price) * closed_qty)
                    pos.realized_pnl = round_money(pos.realized_pnl + realized_pnl)

                    residual_qty = qty - current_abs
                    pos.quantity = residual_qty
                    pos.average_entry_price = round_money(fill_price)

        # 4. Daily Realized P&L Tracking
        t_date = fill.timestamp.date() if isinstance(fill.timestamp, datetime) else fill.timestamp
        self._latest_trading_date = t_date
        if realized_pnl != Decimal("0.00"):
            current_daily = self._daily_realized_pnl.get(t_date, Decimal("0.00"))
            self._daily_realized_pnl[t_date] = round_money(current_daily + realized_pnl)

        # 5. Mark Position to Latest Fill Price
        self._latest_market_prices[resolved_symbol] = fill_price
        if pos.quantity > 0:
            pos.unrealized_pnl = round_money((fill_price - pos.average_entry_price) * Decimal(pos.quantity))
        elif pos.quantity < 0:
            pos.unrealized_pnl = round_money((pos.average_entry_price - fill_price) * Decimal(abs(pos.quantity)))
        else:
            pos.unrealized_pnl = Decimal("0.00")

        # 6. Recompute Equity, Peak, and Drawdown
        self._update_equity_and_drawdown()

        return realized_pnl

    # ── Mark-To-Market & Valuations ───────────────────────────────────────────

    def mark_to_market(self, symbol: str, market_price: float | Decimal | str) -> Decimal:
        """
        Update the reference market price for an instrument and revalue unrealized P&L.

        For LONG:  unrealized_pnl = (market_price - average_entry_price) * quantity
        For SHORT: unrealized_pnl = (average_entry_price - market_price) * quantity
        For FLAT:  unrealized_pnl = 0

        Args:
            symbol: Instrument symbol.
            market_price: Current market price.

        Returns:
            The symbol's updated unrealized P&L (Decimal).

        Raises:
            ValueError: If market_price <= 0.
        """
        mkt_px = to_decimal(market_price)
        if mkt_px <= Decimal("0.00"):
            raise ValueError(f"market_price must be positive, got {market_price}")

        self._latest_market_prices[symbol] = mkt_px

        pos = self._positions.get(symbol)
        if pos is not None and pos.quantity != 0:
            if pos.quantity > 0:
                pos.unrealized_pnl = round_money((mkt_px - pos.average_entry_price) * Decimal(pos.quantity))
            else:
                pos.unrealized_pnl = round_money((pos.average_entry_price - mkt_px) * Decimal(abs(pos.quantity)))
        elif pos is not None:
            pos.unrealized_pnl = Decimal("0.00")

        self._update_equity_and_drawdown()
        return pos.unrealized_pnl if pos is not None else Decimal("0.00")

    def mark_to_market_all(self, market_prices: dict[str, float | Decimal | str]) -> Decimal:
        """
        Batch mark-to-market update for multiple instruments.

        Args:
            market_prices: Mapping of symbol to market price.

        Returns:
            The portfolio's updated total equity (Decimal).
        """
        for sym, px in market_prices.items():
            self.mark_to_market(sym, px)
        return self.get_equity()

    # ── Equity & Drawdown Internal Calculations ───────────────────────────────

    def _calculate_equity(self) -> Decimal:
        """
        Calculate total portfolio equity:
            equity = cash + sum(market_value_of_open_positions)
            market_value = position_quantity * market_price
        """
        total_market_value = Decimal("0.00")
        for sym, pos in self._positions.items():
            if pos.quantity != 0:
                mkt_px = self._latest_market_prices.get(sym, pos.average_entry_price)
                market_val = Decimal(pos.quantity) * mkt_px
                total_market_value += market_val
        return round_money(self._cash + total_market_value)

    def _update_equity_and_drawdown(self) -> None:
        """Update current equity, high-water mark (peak equity), and drawdown metrics."""
        current_equity = self._calculate_equity()
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._drawdown = Decimal("0.00")
            self._drawdown_pct = Decimal("0.00")
        else:
            self._drawdown = round_money(self._peak_equity - current_equity)
            if self._peak_equity > Decimal("0.00"):
                raw_pct = (self._drawdown / self._peak_equity) * Decimal("100")
                self._drawdown_pct = round_money(raw_pct)
            else:
                self._drawdown_pct = Decimal("0.00")

    # ── Public Query API ──────────────────────────────────────────────────────

    def get_cash(self) -> Decimal:
        """Return the current unallocated cash balance in INR."""
        return self._cash

    def get_initial_cash(self) -> Decimal:
        """Return the starting capital in INR."""
        return self._initial_cash

    def get_equity(self) -> Decimal:
        """Return current portfolio equity (cash + open positions market value)."""
        return self._calculate_equity()

    def get_peak_equity(self) -> Decimal:
        """Return the highest observed portfolio equity (high-water mark)."""
        return self._peak_equity

    def get_drawdown(self) -> Decimal:
        """Return current drawdown in INR (peak_equity - current_equity)."""
        return self._drawdown

    def get_drawdown_pct(self) -> Decimal:
        """Return current drawdown percentage (drawdown / peak_equity * 100)."""
        return self._drawdown_pct

    def get_total_brokerage(self) -> Decimal:
        """Return cumulative brokerage commissions deducted."""
        return self._total_brokerage

    def get_position(self, symbol: str) -> Position:
        """
        Return an immutable snapshot of the position for a given symbol.

        If the symbol has never been traded, returns a clean flat Position object.
        Callers cannot mutate internal portfolio state via the returned object.
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return Position(symbol=symbol)
        return Position(
            symbol=pos.symbol,
            quantity=pos.quantity,
            average_entry_price=pos.average_entry_price,
            realized_pnl=pos.realized_pnl,
            unrealized_pnl=pos.unrealized_pnl,
            total_quantity_traded=pos.total_quantity_traded,
        )

    def get_all_positions(self) -> dict[str, Position]:
        """
        Return snapshot copies of all tracked positions.

        Returns:
            Dictionary mapping symbol to Position snapshot.
        """
        return {sym: self.get_position(sym) for sym in self._positions}

    def get_realized_pnl(self, symbol: str | None = None) -> Decimal:
        """
        Return cumulative realized trading P&L.

        Args:
            symbol: If specified, returns realized P&L for that symbol only.
                    If None, returns sum of realized P&L across all symbols.

        Returns:
            Decimal realized P&L in INR.
        """
        if symbol is not None:
            pos = self._positions.get(symbol)
            return pos.realized_pnl if pos is not None else Decimal("0.00")
        return sum(
            (p.realized_pnl for p in self._positions.values()),
            start=Decimal("0.00"),
        )

    def get_unrealized_pnl(self, symbol: str | None = None) -> Decimal:
        """
        Return current unrealized mark-to-market P&L.

        Args:
            symbol: If specified, returns unrealized P&L for that symbol only.
                    If None, returns sum of unrealized P&L across all symbols.

        Returns:
            Decimal unrealized P&L in INR.
        """
        if symbol is not None:
            pos = self._positions.get(symbol)
            return pos.unrealized_pnl if pos is not None else Decimal("0.00")
        return sum(
            (p.unrealized_pnl for p in self._positions.values()),
            start=Decimal("0.00"),
        )

    def get_daily_realized_pnl(self, trading_date: date | str | None = None) -> Decimal:
        """
        Return realized P&L for a specific or current trading date.

        Used by RiskManager to enforce daily loss limits.

        Args:
            trading_date: Optional date (as date object or 'YYYY-MM-DD' string).
                          If None, defaults to the latest recorded trading date.

        Returns:
            Realized P&L for that date in INR (Decimal).
        """
        if trading_date is None:
            if self._latest_trading_date is None:
                return Decimal("0.00")
            return self._daily_realized_pnl.get(self._latest_trading_date, Decimal("0.00"))

        if isinstance(trading_date, str):
            target_date = date.fromisoformat(trading_date)
        elif isinstance(trading_date, date):
            target_date = trading_date
        else:
            raise TypeError(f"trading_date must be a date, str, or None, got {type(trading_date).__name__}")

        return self._daily_realized_pnl.get(target_date, Decimal("0.00"))
