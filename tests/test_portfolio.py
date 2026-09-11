"""
Comprehensive unit tests for Portfolio and Position accounting.

Tests:
    - Initial portfolio state
    - Open long, increase long with weighted average entry price
    - Partial long exit and complete long exit (profitable & losing)
    - Open short, increase short with weighted average entry price
    - Partial short exit and complete short exit (profitable & losing)
    - Long -> short reversal and short -> long reversal
    - Brokerage deductions and cash/equity consistency
    - Mark-to-market unrealized P&L for long, short, and flat
    - Equity, peak equity, drawdown, and drawdown percentage
    - Multi-symbol independence
    - Daily realized P&L query by date or string
    - Input validation and Decimal precision handling
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
import pytest

from vega.orders.models import Fill, OrderSide
from vega.portfolio.portfolio import Portfolio, Position, round_money, to_decimal


@pytest.fixture
def dt_day1() -> datetime:
    return datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def dt_day2() -> datetime:
    return datetime(2026, 9, 12, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def portfolio() -> Portfolio:
    return Portfolio(initial_cash=1_000_000.0)


# ── Initial State ────────────────────────────────────────────────────────────


def test_initial_portfolio_state(portfolio: Portfolio) -> None:
    """Verify initial portfolio state and zeroed metrics."""
    assert portfolio.get_initial_cash() == Decimal("1000000.00")
    assert portfolio.get_cash() == Decimal("1000000.00")
    assert portfolio.get_equity() == Decimal("1000000.00")
    assert portfolio.get_peak_equity() == Decimal("1000000.00")
    assert portfolio.get_drawdown() == Decimal("0.00")
    assert portfolio.get_drawdown_pct() == Decimal("0.00")
    assert portfolio.get_realized_pnl() == Decimal("0.00")
    assert portfolio.get_unrealized_pnl() == Decimal("0.00")
    assert portfolio.get_total_brokerage() == Decimal("0.00")
    assert portfolio.get_all_positions() == {}

    # Untraded symbol returns clean flat position
    pos = portfolio.get_position("NIFTY50")
    assert pos.symbol == "NIFTY50"
    assert pos.quantity == 0
    assert pos.is_flat
    assert not pos.is_long
    assert not pos.is_short
    assert pos.average_entry_price == Decimal("0.00")
    assert pos.realized_pnl == Decimal("0.00")
    assert pos.unrealized_pnl == Decimal("0.00")
    assert pos.total_quantity_traded == 0


def test_portfolio_negative_initial_cash_raises() -> None:
    """Negative initial cash must raise ValueError."""
    with pytest.raises(ValueError, match="initial_cash must be non-negative"):
        Portfolio(initial_cash=-1000)


# ── Long Positions: Open, Increase, Reduce, Close ───────────────────────────


def test_open_long(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Open Long:
        BUY 2 @ 20,000.00, brokerage 12.00
        Cash decreases by: 20000 * 2 + 12 = 40,012.00
        Cash becomes: 1,000,000 - 40,012 = 959,988.00
        Position: +2 @ 20,000.00
        Equity = cash (959,988) + market_value (40,000) = 999,988.00
    """
    fill = Fill(
        order_id="BUY-1",
        filled_price=20000.0,
        filled_qty=2,
        timestamp=dt_day1,
        brokerage=12.0,
        symbol="NIFTY50",
        side=OrderSide.BUY,
    )
    pnl = portfolio.process_fill(fill)
    assert pnl == Decimal("0.00")

    assert portfolio.get_cash() == Decimal("959988.00")
    assert portfolio.get_total_brokerage() == Decimal("12.00")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 2
    assert pos.is_long
    assert pos.average_entry_price == Decimal("20000.00")
    assert pos.realized_pnl == Decimal("0.00")
    assert pos.unrealized_pnl == Decimal("0.00")
    assert pos.total_quantity_traded == 2

    assert portfolio.get_equity() == Decimal("999988.00")


def test_increase_long_weighted_average(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Increase Long:
        Step 1: BUY 2 @ 20,000.00, brokerage 12.00
        Step 2: BUY 3 @ 21,000.00, brokerage 18.90
        New average entry price:
            ((2 * 20000) + (3 * 21000)) / 5 = (40000 + 63000) / 5 = 20,600.00
        Cash outflow: 63,018.90 -> cash = 959,988 - 63,018.90 = 896,969.10
    """
    f1 = Fill("B1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.BUY)
    f2 = Fill("B2", 21000.0, 3, dt_day1, brokerage=18.90, symbol="NIFTY50", side=OrderSide.BUY)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)

    assert portfolio.get_cash() == Decimal("896969.10")
    assert portfolio.get_total_brokerage() == Decimal("30.90")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 5
    assert pos.average_entry_price == Decimal("20600.00")
    assert pos.total_quantity_traded == 5


def test_partial_long_exit_profitable(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Partial Long Exit:
        Position: +5 @ 20,600.00
        SELL 2 @ 22,000.00, brokerage 13.20
        Realized P&L = (22000 - 20600) * 2 = +2,800.00
        Cash inflow = 22000 * 2 - 13.20 = 43,986.80
        Cash becomes: 896,969.10 + 43,986.80 = 940,955.90
        Remaining position: +3 @ 20,600.00 (average price unchanged!)
    """
    f1 = Fill("B1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.BUY)
    f2 = Fill("B2", 21000.0, 3, dt_day1, brokerage=18.90, symbol="NIFTY50", side=OrderSide.BUY)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)

    f3 = Fill("S1", 22000.0, 2, dt_day1, brokerage=13.20, symbol="NIFTY50", side=OrderSide.SELL)
    pnl = portfolio.process_fill(f3)

    assert pnl == Decimal("2800.00")
    assert portfolio.get_cash() == Decimal("940955.90")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("2800.00")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 3
    assert pos.average_entry_price == Decimal("20600.00")
    assert pos.realized_pnl == Decimal("2800.00")
    assert pos.total_quantity_traded == 7


def test_complete_long_exit_losing(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Complete Long Exit:
        Position: +3 @ 20,600.00
        SELL 3 @ 19,600.00, brokerage 17.64
        Realized P&L = (19600 - 20600) * 3 = -3,000.00
        Cumulative Realized P&L = 2800 + (-3000) = -200.00
        Cash inflow = 19600 * 3 - 17.64 = 58,782.36
        Cash becomes: 940,955.90 + 58,782.36 = 999,738.26
        Remaining position: 0 (flat), avg entry price reset to 0.00
        Net cash check: 1,000,000 - 200 (loss) - 61.74 (total fees) = 999,738.26
    """
    f1 = Fill("B1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.BUY)
    f2 = Fill("B2", 21000.0, 3, dt_day1, brokerage=18.90, symbol="NIFTY50", side=OrderSide.BUY)
    f3 = Fill("S1", 22000.0, 2, dt_day1, brokerage=13.20, symbol="NIFTY50", side=OrderSide.SELL)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)
    portfolio.process_fill(f3)

    f4 = Fill("S2", 19600.0, 3, dt_day1, brokerage=17.64, symbol="NIFTY50", side=OrderSide.SELL)
    pnl = portfolio.process_fill(f4)

    assert pnl == Decimal("-3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("-200.00")
    assert portfolio.get_realized_pnl() == Decimal("-200.00")
    assert portfolio.get_cash() == Decimal("999738.26")
    assert portfolio.get_total_brokerage() == Decimal("61.74")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 0
    assert pos.is_flat
    assert pos.average_entry_price == Decimal("0.00")
    assert pos.unrealized_pnl == Decimal("0.00")
    assert pos.total_quantity_traded == 10

    # Equity equals cash when flat
    assert portfolio.get_equity() == Decimal("999738.26")


def test_complete_round_trip_economic_pnl(dt_day1: datetime) -> None:
    """
    High-value regression test for a complete round-trip trade:
        Initial cash: ₹100,000
        BUY 10 at ₹100 with ₹10 brokerage
        SELL 10 at ₹110 with ₹10 brokerage

    Verifies:
        - final position = 0
        - gross trading P&L = ₹100
        - total transaction costs = ₹20
        - net economic P&L = ₹80
        - final cash/equity = ₹100,080
    """
    p = Portfolio(initial_cash=100_000.0)

    # 1. BUY 10 at ₹100, brokerage ₹10
    # Cash outflow: 10 * 100 + 10 = 1,010 -> cash = 98,990.00
    p.process_fill(Fill("B1", 100.0, 10, dt_day1, brokerage=10.0, symbol="TEST", side=OrderSide.BUY))
    assert p.get_cash() == Decimal("98990.00")
    assert p.get_position("TEST").quantity == 10

    # 2. SELL 10 at ₹110, brokerage ₹10
    # Cash inflow: 10 * 110 - 10 = 1,090 -> cash = 98,990 + 1,090 = 100,080.00
    realized = p.process_fill(Fill("S1", 110.0, 10, dt_day1, brokerage=10.0, symbol="TEST", side=OrderSide.SELL))

    # Assertions:
    pos = p.get_position("TEST")
    assert pos.quantity == 0
    assert pos.is_flat

    gross_trading_pnl = p.get_realized_pnl("TEST")
    assert gross_trading_pnl == Decimal("100.00")
    assert realized == Decimal("100.00")

    total_transaction_costs = p.get_total_brokerage()
    assert total_transaction_costs == Decimal("20.00")

    net_economic_pnl = gross_trading_pnl - total_transaction_costs
    assert net_economic_pnl == Decimal("80.00")

    final_cash = p.get_cash()
    final_equity = p.get_equity()
    assert final_cash == Decimal("100080.00")
    assert final_equity == Decimal("100080.00")
    assert final_cash == p.get_initial_cash() + net_economic_pnl


def test_profitable_long(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Explicit test: Opening and closing a profitable long trade."""
    portfolio.process_fill(Fill("B1", 20000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    pnl = portfolio.process_fill(Fill("S1", 21500.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.SELL))
    assert pnl == Decimal("3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("3000.00")
    assert portfolio.get_position("NIFTY50").is_flat


def test_losing_long(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Explicit test: Opening and closing a losing long trade."""
    portfolio.process_fill(Fill("B1", 20000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    pnl = portfolio.process_fill(Fill("S1", 19000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.SELL))
    assert pnl == Decimal("-2000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("-2000.00")
    assert portfolio.get_position("NIFTY50").is_flat


def test_brokerage_affects_cash_and_pnl_correctly(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Explicit test: Brokerage affects cash, P&L, and accounting consistently.
        Initial cash: 1,000,000.00
        BUY 1 @ 20,000.00, fee 10.00 -> cash decreases by 20,010.00 -> cash = 979,990.00
        SELL 1 @ 21,000.00, fee 10.50 -> cash increases by 20,989.50 -> cash = 1,000,979.50
        Realized trading P&L = (21000 - 20000) * 1 = +1,000.00
        Total brokerage = 10.00 + 10.50 = 20.50
        Net cash gain = 1000.00 - 20.50 = +979.50
        Final cash = 1,000,000.00 + 979.50 = 1,000,979.50
    """
    portfolio.process_fill(Fill("B1", 20000.0, 1, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    assert portfolio.get_cash() == Decimal("979990.00")

    pnl = portfolio.process_fill(Fill("S1", 21000.0, 1, dt_day1, brokerage=10.50, symbol="NIFTY50", side=OrderSide.SELL))
    assert pnl == Decimal("1000.00")
    assert portfolio.get_total_brokerage() == Decimal("20.50")
    assert portfolio.get_cash() == Decimal("1000979.50")
    assert portfolio.get_equity() == Decimal("1000979.50")


# ── Short Positions: Open, Increase, Reduce, Close ──────────────────────────


def test_open_short(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Open Short:
        SELL 2 @ 20,000.00, brokerage 12.00
        Cash increases by: 20000 * 2 - 12 = 39,988.00
        Cash becomes: 1,000,000 + 39,988 = 1,039,988.00
        Position: -2 @ 20,000.00
        Market value: -2 * 20000 = -40,000.00
        Equity = 1,039,988 - 40,000 = 999,988.00 (initial - 12 fee)
    """
    fill = Fill(
        order_id="SHORT-1",
        filled_price=20000.0,
        filled_qty=2,
        timestamp=dt_day1,
        brokerage=12.0,
        symbol="NIFTY50",
        side=OrderSide.SELL,
    )
    pnl = portfolio.process_fill(fill)
    assert pnl == Decimal("0.00")

    assert portfolio.get_cash() == Decimal("1039988.00")
    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == -2
    assert pos.is_short
    assert pos.average_entry_price == Decimal("20000.00")
    assert pos.realized_pnl == Decimal("0.00")

    assert portfolio.get_equity() == Decimal("999988.00")


def test_increase_short_weighted_average(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Increase Short:
        Step 1: SELL 2 @ 20,000.00, brokerage 12.00
        Step 2: SELL 3 @ 19,000.00, brokerage 17.10
        New average entry price:
            ((2 * 20000) + (3 * 19000)) / 5 = (40000 + 57000) / 5 = 19,400.00
        Cash inflow: 19000 * 3 - 17.10 = 56,982.90
        Cash becomes: 1,039,988 + 56,982.90 = 1,096,970.90
    """
    f1 = Fill("S1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.SELL)
    f2 = Fill("S2", 19000.0, 3, dt_day1, brokerage=17.10, symbol="NIFTY50", side=OrderSide.SELL)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)

    assert portfolio.get_cash() == Decimal("1096970.90")
    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == -5
    assert pos.average_entry_price == Decimal("19400.00")


def test_partial_short_exit_profitable(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Partial Short Exit:
        Position: -5 @ 19,400.00
        BUY 2 @ 18,400.00, brokerage 11.04
        Realized P&L = (19400 - 18400) * 2 = +2,000.00
        Cash outflow = 18400 * 2 + 11.04 = 36,811.04
        Cash becomes: 1,096,970.90 - 36,811.04 = 1,060,159.86
        Remaining position: -3 @ 19,400.00 (average price unchanged!)
    """
    f1 = Fill("S1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.SELL)
    f2 = Fill("S2", 19000.0, 3, dt_day1, brokerage=17.10, symbol="NIFTY50", side=OrderSide.SELL)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)

    f3 = Fill("B1", 18400.0, 2, dt_day1, brokerage=11.04, symbol="NIFTY50", side=OrderSide.BUY)
    pnl = portfolio.process_fill(f3)

    assert pnl == Decimal("2000.00")
    assert portfolio.get_cash() == Decimal("1060159.86")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == -3
    assert pos.average_entry_price == Decimal("19400.00")
    assert pos.realized_pnl == Decimal("2000.00")


def test_complete_short_exit_losing(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Complete Short Exit:
        Position: -3 @ 19,400.00
        BUY 3 @ 20,400.00, brokerage 18.36
        Realized P&L = (19400 - 20400) * 3 = -3,000.00
        Cumulative Realized P&L = 2000 + (-3000) = -1,000.00
        Cash outflow = 20400 * 3 + 18.36 = 61,218.36
        Cash becomes: 1,060,159.86 - 61,218.36 = 998,941.50
        Net cash check: 1,000,000 - 1000 (loss) - 58.50 (total fees) = 998,941.50
    """
    f1 = Fill("S1", 20000.0, 2, dt_day1, brokerage=12.0, symbol="NIFTY50", side=OrderSide.SELL)
    f2 = Fill("S2", 19000.0, 3, dt_day1, brokerage=17.10, symbol="NIFTY50", side=OrderSide.SELL)
    f3 = Fill("B1", 18400.0, 2, dt_day1, brokerage=11.04, symbol="NIFTY50", side=OrderSide.BUY)
    portfolio.process_fill(f1)
    portfolio.process_fill(f2)
    portfolio.process_fill(f3)

    f4 = Fill("B2", 20400.0, 3, dt_day1, brokerage=18.36, symbol="NIFTY50", side=OrderSide.BUY)
    pnl = portfolio.process_fill(f4)

    assert pnl == Decimal("-3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("-1000.00")
    assert portfolio.get_cash() == Decimal("998941.50")
    assert portfolio.get_total_brokerage() == Decimal("58.50")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 0
    assert pos.is_flat
    assert portfolio.get_equity() == Decimal("998941.50")


def test_profitable_short(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Explicit test: Opening and closing a profitable short trade."""
    portfolio.process_fill(Fill("S1", 20000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.SELL))
    pnl = portfolio.process_fill(Fill("B1", 18500.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    assert pnl == Decimal("3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("3000.00")
    assert portfolio.get_position("NIFTY50").is_flat


def test_losing_short(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Explicit test: Opening and closing a losing short trade."""
    portfolio.process_fill(Fill("S1", 20000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.SELL))
    pnl = portfolio.process_fill(Fill("B1", 21000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    assert pnl == Decimal("-2000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("-2000.00")
    assert portfolio.get_position("NIFTY50").is_flat


# ── Position Reversals ───────────────────────────────────────────────────────


def test_reversal_long_to_short(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Reversal Long -> Short:
        Position: +3 @ 20,000.00
        SELL 5 @ 21,000.00, brokerage 31.50
        Step 1: Closes 3 LONG units @ 21,000.00
            realized_pnl = (21000 - 20000) * 3 = +3,000.00
        Step 2: Residual 2 units open as SHORT @ 21,000.00
            new quantity = -2, avg_price = 21,000.00
        Cash increases by: 21000 * 5 - 31.50 = 104,968.50
    """
    # Open long 3 @ 20,000 (fee 18.00)
    # Cash outflow: 60,018.00 -> cash = 939,982.00
    portfolio.process_fill(Fill("B1", 20000.0, 3, dt_day1, brokerage=18.0, symbol="NIFTY50", side=OrderSide.BUY))

    # Reversal order: SELL 5 @ 21,000
    pnl = portfolio.process_fill(Fill("S1", 21000.0, 5, dt_day1, brokerage=31.50, symbol="NIFTY50", side=OrderSide.SELL))

    assert pnl == Decimal("3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("3000.00")

    # Cash: 939,982.00 + 104,968.50 = 1,044,950.50
    assert portfolio.get_cash() == Decimal("1044950.50")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == -2
    assert pos.is_short
    assert pos.average_entry_price == Decimal("21000.00")


def test_reversal_short_to_long(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Reversal Short -> Long:
        Position: -3 @ 20,000.00
        BUY 5 @ 19,000.00, brokerage 28.50
        Step 1: Closes 3 SHORT units @ 19,000.00
            realized_pnl = (20000 - 19000) * 3 = +3,000.00
        Step 2: Residual 2 units open as LONG @ 19,000.00
            new quantity = +2, avg_price = 19,000.00
    """
    # Open short 3 @ 20,000 (fee 18.00)
    # Cash inflow: 60,000 - 18 = 59,982.00 -> cash = 1,059,982.00
    portfolio.process_fill(Fill("S1", 20000.0, 3, dt_day1, brokerage=18.0, symbol="NIFTY50", side=OrderSide.SELL))

    # Reversal order: BUY 5 @ 19,000
    # Cash outflow: 19000 * 5 + 28.50 = 95,028.50 -> cash = 1,059,982 - 95,028.50 = 964,953.50
    pnl = portfolio.process_fill(Fill("B1", 19000.0, 5, dt_day1, brokerage=28.50, symbol="NIFTY50", side=OrderSide.BUY))

    assert pnl == Decimal("3000.00")
    assert portfolio.get_realized_pnl("NIFTY50") == Decimal("3000.00")
    assert portfolio.get_cash() == Decimal("964953.50")

    pos = portfolio.get_position("NIFTY50")
    assert pos.quantity == 2
    assert pos.is_long
    assert pos.average_entry_price == Decimal("19000.00")


# ── Mark-To-Market & Unrealized P&L ──────────────────────────────────────────


def test_unrealized_pnl_long_and_short(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Verify unrealized P&L calculation for both Long and Short positions:
        For LONG (+2 @ 20,000.00):
            market_price 20,500.00 -> unrealized = +1,000.00
            market_price 19,500.00 -> unrealized = -1,000.00
        For SHORT (-2 @ 20,000.00):
            market_price 19,500.00 -> unrealized = +1,000.00
            market_price 20,500.00 -> unrealized = -1,000.00
    """
    # 1. Long test
    portfolio.process_fill(Fill("B1", 20000.0, 2, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.BUY))
    u1 = portfolio.mark_to_market("NIFTY50", 20500.0)
    assert u1 == Decimal("1000.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("1000.00")

    u2 = portfolio.mark_to_market("NIFTY50", 19500.0)
    assert u2 == Decimal("-1000.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("-1000.00")

    # Close long to flat
    portfolio.process_fill(Fill("S1", 19500.0, 2, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.SELL))
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("0.00")

    # 2. Short test
    portfolio.process_fill(Fill("S2", 20000.0, 2, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.SELL))
    u3 = portfolio.mark_to_market("NIFTY50", 19500.0)
    assert u3 == Decimal("1000.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("1000.00")

    u4 = portfolio.mark_to_market("NIFTY50", 20500.0)
    assert u4 == Decimal("-1000.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("-1000.00")


def test_flat_position_unrealized_pnl_zero(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Explicit test: Flat position has exactly 0.00 unrealized P&L regardless of market price."""
    # Never traded
    assert portfolio.get_position("NIFTY50").is_flat
    assert portfolio.mark_to_market("NIFTY50", 25000.0) == Decimal("0.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("0.00")

    # Traded and closed
    portfolio.process_fill(Fill("B1", 20000.0, 1, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.BUY))
    portfolio.process_fill(Fill("S1", 20000.0, 1, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.SELL))
    assert portfolio.get_position("NIFTY50").is_flat
    assert portfolio.mark_to_market("NIFTY50", 30000.0) == Decimal("0.00")
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("0.00")


# ── Equity, Peak Equity, and Drawdown ────────────────────────────────────────


def test_equity_peak_and_drawdown_mechanics(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Track peak equity and drawdown percentages:
        Initial: Equity = 1,000,000.00, Peak = 1,000,000.00, DD = 0.00, DD% = 0.00
        Buy 10 @ 20,000.00 (fee 0): Cash = 800,000.00
        MTM @ 22,000.00:
            Equity = 800,000 + 220,000 = 1,020,000.00 (New Peak!)
            DD = 0.00, DD% = 0.00
        MTM @ 21,000.00:
            Equity = 800,000 + 210,000 = 1,010,000.00
            Peak = 1,020,000.00
            DD = 1,020,000 - 1,010,000 = 10,000.00
            DD% = (10,000 / 1,020,000) * 100 = 0.98%
        MTM @ 18,000.00:
            Equity = 800,000 + 180,000 = 980,000.00
            DD = 1,020,000 - 980,000 = 40,000.00
            DD% = (40,000 / 1,020,000) * 100 = 3.92%
    """
    portfolio.process_fill(Fill("B1", 20000.0, 10, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.BUY))

    # Rally to 22,000
    portfolio.mark_to_market("NIFTY50", 22000.0)
    assert portfolio.get_equity() == Decimal("1020000.00")
    assert portfolio.get_peak_equity() == Decimal("1020000.00")
    assert portfolio.get_drawdown() == Decimal("0.00")
    assert portfolio.get_drawdown_pct() == Decimal("0.00")

    # Pullback to 21,000
    portfolio.mark_to_market("NIFTY50", 21000.0)
    assert portfolio.get_equity() == Decimal("1010000.00")
    assert portfolio.get_peak_equity() == Decimal("1020000.00")
    assert portfolio.get_drawdown() == Decimal("10000.00")
    assert portfolio.get_drawdown_pct() == Decimal("0.98")

    # Decline to 18,000
    portfolio.mark_to_market("NIFTY50", 18000.0)
    assert portfolio.get_equity() == Decimal("980000.00")
    assert portfolio.get_peak_equity() == Decimal("1020000.00")
    assert portfolio.get_drawdown() == Decimal("40000.00")
    assert portfolio.get_drawdown_pct() == Decimal("3.92")


def test_zero_peak_equity_drawdown_safe() -> None:
    """Zero initial/peak equity must not raise ZeroDivisionError for drawdown_pct."""
    p = Portfolio(initial_cash=0.0)
    assert p.get_peak_equity() == Decimal("0.00")
    assert p.get_drawdown() == Decimal("0.00")
    assert p.get_drawdown_pct() == Decimal("0.00")


# ── Multiple Symbols ─────────────────────────────────────────────────────────


def test_multiple_symbols_independence(portfolio: Portfolio, dt_day1: datetime) -> None:
    """
    Verify portfolio handles multiple instruments concurrently and independently:
        NIFTY50: BUY 2 @ 20,000.00
        BANKNIFTY: SELL 1 @ 45,000.00
    """
    portfolio.process_fill(Fill("N1", 20000.0, 2, dt_day1, brokerage=10.0, symbol="NIFTY50", side=OrderSide.BUY))
    portfolio.process_fill(Fill("B1", 45000.0, 1, dt_day1, brokerage=15.0, symbol="BANKNIFTY", side=OrderSide.SELL))

    # Cash: 1,000,000 - 40,010 + 44,985 = 1,004,975.00
    assert portfolio.get_cash() == Decimal("1004975.00")

    all_pos = portfolio.get_all_positions()
    assert len(all_pos) == 2
    assert all_pos["NIFTY50"].quantity == 2
    assert all_pos["BANKNIFTY"].quantity == -1

    # Mark both to market
    portfolio.mark_to_market_all({"NIFTY50": 21000.0, "BANKNIFTY": 44000.0})
    # NIFTY unrealized: (21000 - 20000) * 2 = +2,000.00
    # BANKNIFTY unrealized: (45000 - 44000) * 1 = +1,000.00
    assert portfolio.get_unrealized_pnl("NIFTY50") == Decimal("2000.00")
    assert portfolio.get_unrealized_pnl("BANKNIFTY") == Decimal("1000.00")
    assert portfolio.get_unrealized_pnl() == Decimal("3000.00")

    # Equity = cash (1,004,975) + NIFTY mkt_val (42,000) + BANKNIFTY mkt_val (-44,000) = 1,002,975.00
    assert portfolio.get_equity() == Decimal("1002975.00")


# ── Daily Realized P&L ───────────────────────────────────────────────────────


def test_daily_realized_pnl_tracking(
    portfolio: Portfolio, dt_day1: datetime, dt_day2: datetime
) -> None:
    """
    Verify daily realized P&L is cleanly aggregated by trading date:
        Day 1 (2026-09-11): Realized P&L = +2,800.00
        Day 2 (2026-09-12): Realized P&L = -3,000.00
    """
    # Day 1: Open long 2 @ 20,000 and sell 2 @ 21,400 -> realized +2,800
    portfolio.process_fill(Fill("B1", 20000.0, 2, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.BUY))
    portfolio.process_fill(Fill("S1", 21400.0, 2, dt_day1, brokerage=0.0, symbol="NIFTY50", side=OrderSide.SELL))

    # Day 2: Open long 3 @ 21,000 and sell 3 @ 20,000 -> realized -3,000
    portfolio.process_fill(Fill("B2", 21000.0, 3, dt_day2, brokerage=0.0, symbol="NIFTY50", side=OrderSide.BUY))
    portfolio.process_fill(Fill("S2", 20000.0, 3, dt_day2, brokerage=0.0, symbol="NIFTY50", side=OrderSide.SELL))

    # Query by date object
    assert portfolio.get_daily_realized_pnl(date(2026, 9, 11)) == Decimal("2800.00")
    assert portfolio.get_daily_realized_pnl(date(2026, 9, 12)) == Decimal("-3000.00")

    # Query by ISO date string
    assert portfolio.get_daily_realized_pnl("2026-09-11") == Decimal("2800.00")
    assert portfolio.get_daily_realized_pnl("2026-09-12") == Decimal("-3000.00")

    # Query latest trading day (default)
    assert portfolio.get_daily_realized_pnl() == Decimal("-3000.00")

    # Untraded date returns 0.00
    assert portfolio.get_daily_realized_pnl(date(2026, 9, 13)) == Decimal("0.00")


# ── Precision & Validation ───────────────────────────────────────────────────


def test_decimal_precision_no_float_drift() -> None:
    """Float prices must convert strictly via Decimal(str(val)) without binary float noise."""
    val = to_decimal(22000.15)
    assert str(val) == "22000.15"
    assert isinstance(val, Decimal)

    rounded = round_money(Decimal("10.125"))
    # Banker's rounding: 10.125 -> 10.12 (nearest even last digit)
    assert rounded == Decimal("10.12")
    assert round_money(Decimal("10.135")) == Decimal("10.14")


def test_invalid_fill_inputs_raise(portfolio: Portfolio, dt_day1: datetime) -> None:
    """Portfolio rejects non-Fill objects, missing symbols, and invalid prices/quantities."""
    with pytest.raises(TypeError, match="fill must be an instance of Fill"):
        portfolio.process_fill("not_a_fill")  # type: ignore[arg-type]

    # Missing symbol
    bad_fill = Fill("F1", 20000.0, 1, dt_day1, symbol="", side=OrderSide.BUY)
    with pytest.raises(ValueError, match="Fill must have a non-empty symbol"):
        portfolio.process_fill(bad_fill)

    # Missing side
    bad_fill2 = Fill("F2", 20000.0, 1, dt_day1, symbol="NIFTY50", side=None)
    with pytest.raises(ValueError, match="side must be an OrderSide"):
        portfolio.process_fill(bad_fill2)

    # Non-positive market price
    with pytest.raises(ValueError, match="market_price must be positive"):
        portfolio.mark_to_market("NIFTY50", -100.0)
