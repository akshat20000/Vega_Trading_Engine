"""
Indian Market Expiry and Contract Rollover Engine.

Provides:
    - TradingCalendar: Holiday-aware Indian trading calendar with previous-trading-day adjustment.
    - ExpiryRule: Configurable expiry calculation (supports current NSE Tuesday specification).
    - RolloverDecision & RolloverManager: Automated calendar spread order generation
      preserving position direction and quantity when Days To Expiry (DTE) hits threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import uuid

from vega.markets.contract_master import ContractSpec
from vega.orders.models import Order, OrderSide, OrderStatus


# Default canonical Indian market holidays for 2026
CANONICAL_INDIAN_HOLIDAYS_2026: set[date] = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri
    date(2026, 3, 4),    # Holi
    date(2026, 3, 20),   # Id-ul-Fitr
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 27),   # Bakri Id
    date(2026, 6, 26),   # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 26),   # Milad-un-Nabi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 8),   # Diwali (Laxmi Pujan)
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}


class TradingCalendar:
    """
    Deterministic Indian market trading calendar.

    Handles weekday trading sessions and holiday backwards adjustments.
    """

    def __init__(self, holidays: set[date] | None = None) -> None:
        self.holidays: set[date] = set(holidays) if holidays is not None else set(CANONICAL_INDIAN_HOLIDAYS_2026)

    def is_trading_day(self, d: date) -> bool:
        """Return True if date is a weekday (Monday-Friday) and not an exchange holiday."""
        # 0 = Monday, 4 = Friday, 5 = Saturday, 6 = Sunday
        if d.weekday() >= 5:
            return False
        return d not in self.holidays

    def previous_trading_day(self, d: date) -> date:
        """
        Find the strictly preceding valid trading day.
        """
        cur = d - timedelta(days=1)
        while not self.is_trading_day(cur):
            cur -= timedelta(days=1)
        return cur

    def adjust_if_holiday(self, d: date) -> date:
        """
        If d is already a valid trading day, return it; otherwise return
        the latest preceding trading day.
        """
        if self.is_trading_day(d):
            return d
        return self.previous_trading_day(d)


class ExpiryRule:
    """
    Configurable expiry rule for Indian derivatives.

    Attributes:
        weekday: Target expiry day of week (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri).
                 Current NSE index and equity derivatives expire on Tuesday (1).
        calendar: TradingCalendar used for holiday adjustments.
    """

    def __init__(
        self,
        weekday: int = 1,  # Default: Tuesday
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.weekday = weekday
        self.calendar = calendar or TradingCalendar()

    def get_monthly_expiry(self, year: int, month: int) -> date:
        """
        Compute the monthly expiry date for a given year and month.

        Calculates the nominal last specified weekday of the month,
        then shifts to the previous trading day if it falls on an exchange holiday.
        """
        if month == 12:
            last_day = date(year, 12, 31)
        else:
            last_day = date(year, month + 1, 1) - timedelta(days=1)

        # Walk backward to find the last target weekday
        cur = last_day
        while cur.weekday() != self.weekday:
            cur -= timedelta(days=1)

        # Holiday adjustment
        return self.calendar.adjust_if_holiday(cur)

    def get_weekly_expiry(self, reference_date: date) -> date:
        """
        Compute the weekly expiry date containing reference_date.
        """
        diff = self.weekday - reference_date.weekday()
        nominal = reference_date + timedelta(days=diff)
        return self.calendar.adjust_if_holiday(nominal)


@dataclass(frozen=True)
class RolloverDecision:
    """
    Result of a rollover horizon evaluation.

    Attributes:
        requires_rollover: True if DTE <= threshold and position != 0.
        days_to_expiry: Current calendar days remaining until near contract expiry.
        close_order: Order to unwind the near-month contract (or None).
        open_order: Order to establish the same position in next-month contract (or None).
        reason: Explanatory rationale.
    """

    requires_rollover: bool
    days_to_expiry: int
    close_order: Order | None = None
    open_order: Order | None = None
    reason: str = ""


class RolloverManager:
    """
    Automated rollover rule engine for derivative contracts.

    Invariants:
        - When Days To Expiry (DTE) <= threshold, triggers rollover.
        - Long position (qty > 0) -> SELL near-month, BUY next-month.
        - Short position (qty < 0) -> BUY near-month, SELL next-month.
        - Exactly preserves position direction and quantity.
    """

    def __init__(self, dte_threshold: int = 1) -> None:
        self.dte_threshold = dte_threshold

    def check_rollover(
        self,
        current_date: date,
        near_contract: ContractSpec,
        next_contract: ContractSpec,
        position_quantity: int,
        dte_threshold: int | None = None,
    ) -> RolloverDecision:
        """
        Evaluate whether a position in near_contract requires rolling into next_contract.

        Args:
            current_date: Evaluation date.
            near_contract: Currently held contract expiring soon.
            next_contract: Successor contract to roll into.
            position_quantity: Signed units (+ for long, - for short, 0 for flat).
            dte_threshold: Optional override for DTE threshold.

        Returns:
            RolloverDecision with generated paired orders if rolling is required.
        """
        if near_contract.expiry_date is None:
            raise ValueError(
                f"Near contract '{near_contract.symbol}' has no expiry_date set."
            )

        threshold = self.dte_threshold if dte_threshold is None else dte_threshold
        dte = (near_contract.expiry_date - current_date).days

        # If position is flat or DTE is above threshold, no rollover is needed
        if position_quantity == 0 or dte > threshold:
            return RolloverDecision(
                requires_rollover=False,
                days_to_expiry=dte,
                close_order=None,
                open_order=None,
                reason="DTE above threshold or position is flat",
            )

        abs_qty = abs(position_quantity)
        run_id = uuid.uuid4().hex[:8].upper()

        if position_quantity > 0:
            # LONG rollover: SELL near, BUY next
            close_order = Order(
                client_order_id=f"ROLL-CLOSE-{near_contract.symbol}-{run_id}",
                symbol=near_contract.symbol,
                side=OrderSide.SELL,
                quantity=abs_qty,
                price=0.0,  # Market order
                status=OrderStatus.PENDING,
                reason=f"ROLLOVER_CLOSE: DTE={dte} <= {threshold}",
            )
            open_order = Order(
                client_order_id=f"ROLL-OPEN-{next_contract.symbol}-{run_id}",
                symbol=next_contract.symbol,
                side=OrderSide.BUY,
                quantity=abs_qty,
                price=0.0,
                status=OrderStatus.PENDING,
                reason=f"ROLLOVER_OPEN: Roll from {near_contract.symbol}",
            )
        else:
            # SHORT rollover: BUY near, SELL next
            close_order = Order(
                client_order_id=f"ROLL-CLOSE-{near_contract.symbol}-{run_id}",
                symbol=near_contract.symbol,
                side=OrderSide.BUY,
                quantity=abs_qty,
                price=0.0,
                status=OrderStatus.PENDING,
                reason=f"ROLLOVER_CLOSE: DTE={dte} <= {threshold}",
            )
            open_order = Order(
                client_order_id=f"ROLL-OPEN-{next_contract.symbol}-{run_id}",
                symbol=next_contract.symbol,
                side=OrderSide.SELL,
                quantity=abs_qty,
                price=0.0,
                status=OrderStatus.PENDING,
                reason=f"ROLLOVER_OPEN: Roll from {near_contract.symbol}",
            )

        return RolloverDecision(
            requires_rollover=True,
            days_to_expiry=dte,
            close_order=close_order,
            open_order=open_order,
            reason=f"Rollover triggered (DTE {dte} <= {threshold})",
        )
