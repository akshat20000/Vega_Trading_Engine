"""
Tests for Phase 12: Indian Market Plumbing (NSE F&O, MCX, Statutory Costs, and Rollover).

Verifies:
    Contract Master:
        - Canonical contracts (NIFTY, BANKNIFTY, MCX).
        - Tick size validation and rounding.
        - Quantity / lot size validation (rejects invalid multiples).
    Statutory & Brokerage Costs:
        - Itemized cost breakdown (NSE & MCX 2026 rates).
        - STT applied strictly on SELL side.
        - Stamp duty applied strictly on BUY side.
        - GST base (brokerage + exchange + sebi).
        - Brokerage cap (₹20 max).
        - Paisa-level numerical discipline using Decimal.
        - Indicative margin estimation.
    Expiry & Holiday Adjustment:
        - Configurable expiry weekday (current NSE Tuesday rule).
        - Holiday moves expiry backward to previous trading day.
    Contract Rollover:
        - No rollover before DTE threshold.
        - Rollover triggered at DTE <= threshold.
        - Preserves exact direction (Long: Sell near, Buy next; Short: Buy near, Sell next).
        - Preserves exact quantity.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
import pytest

from vega.markets.contract_master import (
    ContractSpec,
    Exchange,
    InstrumentType,
    StaticContractRegistry,
)
from vega.markets.costs import (
    CostBreakdown,
    CostEngine,
    CostSchedule,
    MarginEstimate,
    MCX_FUTURES_SCHEDULE_2026,
    NSE_EQUITY_INTRADAY_SCHEDULE_2026,
    NSE_FUTURES_SCHEDULE_2026,
    estimate_margin,
)
from vega.markets.rollover import (
    ExpiryRule,
    RolloverDecision,
    RolloverManager,
    TradingCalendar,
)
from vega.orders.models import OrderSide


# ─── 1. Contract Master & Tick/Lot Tests ─────────────────────────────────────


def test_known_nifty_contract() -> None:
    """Verify registry contains canonical NIFTY futures with verified specifications."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("NIFTY26APRFUT")

    assert spec.symbol == "NIFTY26APRFUT"
    assert spec.underlying == "NIFTY"
    assert spec.exchange == "NFO"
    assert spec.instrument_type == InstrumentType.INDEX_FUT
    assert spec.lot_size == 25
    assert spec.tick_size == Decimal("0.05")
    assert spec.expiry_weekday == 1  # Tuesday
    assert spec.expiry_date == date(2026, 4, 28)


def test_known_banknifty_contract() -> None:
    """Verify BANKNIFTY contract has lot size 15 and tick size 0.05."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("BANKNIFTY26APRFUT")

    assert spec.underlying == "BANKNIFTY"
    assert spec.lot_size == 15
    assert spec.tick_size == Decimal("0.05")
    assert spec.margin_rate == Decimal("0.15")


def test_known_mcx_contract() -> None:
    """Verify MCX commodity futures specification (CRUDEOIL lot 100, tick 1.00)."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("CRUDEOIL26APRFUT")

    assert spec.exchange == "MCX"
    assert spec.underlying == "CRUDEOIL"
    assert spec.instrument_type == InstrumentType.COMMODITY_FUT
    assert spec.lot_size == 100
    assert spec.tick_size == Decimal("1.00")


def test_tick_size_validation() -> None:
    """
    Verify tick size validation:
        tick = 0.05
        100.00 -> valid
        100.05 -> valid
        100.07 -> invalid
    """
    reg = StaticContractRegistry.create_default()
    spec = reg.get("NIFTY26APRFUT")

    assert spec.validate_price(100.00) is True
    assert spec.validate_price(100.05) is True
    assert spec.validate_price(100.07) is False
    assert spec.validate_price(Decimal("22500.15")) is True
    assert spec.validate_price(Decimal("22500.12")) is False

    # Negative and zero prices are invalid
    assert spec.validate_price(0.0) is False
    assert spec.validate_price(-100.0) is False


def test_tick_size_rounding() -> None:
    """Verify half-up rounding aligns prices to valid tick multiples."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("NIFTY26APRFUT")

    assert spec.round_to_tick(100.07) == Decimal("100.05")
    assert spec.round_to_tick(100.08) == Decimal("100.10")
    assert spec.round_to_tick(100.02) == Decimal("100.00")
    assert spec.round_to_tick(Decimal("22000.04")) == Decimal("22000.05")

    # MCX Crude Oil (tick = 1.00)
    crude = reg.get("CRUDEOIL26APRFUT")
    assert crude.round_to_tick(6500.4) == Decimal("6500.00")
    assert crude.round_to_tick(6500.6) == Decimal("6501.00")


def test_invalid_quantity() -> None:
    """Verify non-multiples of lot size are rejected without silent rounding."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("NIFTY26APRFUT")  # lot = 25

    assert spec.validate_quantity(30) is False
    assert spec.validate_quantity(70) is False
    assert spec.validate_quantity(0) is False
    assert spec.validate_quantity(-25) is False
    assert spec.validate_quantity(12) is False


def test_valid_multiple_of_lot_size() -> None:
    """Verify positive exact multiples of lot size are accepted."""
    reg = StaticContractRegistry.create_default()
    spec = reg.get("NIFTY26APRFUT")  # lot = 25

    for valid_qty in [25, 50, 75, 100, 250, 500]:
        assert spec.validate_quantity(valid_qty) is True


# ─── 2. Statutory Costs & Financial Accuracy Tests ───────────────────────────


def test_nse_futures_cost_breakdown() -> None:
    """
    Verify complete cost calculation for NSE Futures (2026 rates).
    Trade: Sell 1 lot of NIFTY @ 22,000 (qty = 25, turnover = ₹550,000).
    """
    breakdown = CostEngine.calculate_costs(
        price=22000.0,
        quantity=25,
        side="SELL",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    assert breakdown.turnover == Decimal("550000.00")
    # Brokerage: min(550000 * 0.0003 = 165, 20) = 20.00
    assert breakdown.brokerage == Decimal("20.00")
    # STT on sell side (0.05% from April 2026): 550000 * 0.0005 = 275.00
    assert breakdown.stt == Decimal("275.00")
    assert breakdown.ctt == Decimal("0.00")
    # Exchange charges (0.00183%): 550000 * 0.0000183 = 10.065 -> 10.07
    assert breakdown.exchange_charges == Decimal("10.07")
    # SEBI charges (0.0001%): 550000 * 0.000001 = 0.55
    assert breakdown.sebi_charges == Decimal("0.55")
    # Stamp duty: 0.00 on SELL
    assert breakdown.stamp_duty == Decimal("0.00")
    # GST: 18% of (20.00 + 10.07 + 0.55 = 30.62) = 5.5116 -> 5.51
    assert breakdown.gst == Decimal("5.51")
    # Total = 20.00 + 275.00 + 10.07 + 0.55 + 0.00 + 5.51 = 311.13
    assert breakdown.total == Decimal("311.13")
    assert breakdown.statutory_total == Decimal("291.13")


def test_mcx_cost_breakdown() -> None:
    """
    Verify MCX commodity futures calculation.
    Trade: Sell 1 lot of CRUDEOIL @ 6000 (qty = 100, turnover = ₹600,000).
    """
    breakdown = CostEngine.calculate_costs(
        price=6000.0,
        quantity=100,
        side="SELL",
        schedule=MCX_FUTURES_SCHEDULE_2026,
    )

    assert breakdown.turnover == Decimal("600000.00")
    assert breakdown.brokerage == Decimal("20.00")
    # CTT (0.01% on non-agri sell turnover): 600000 * 0.0001 = 60.00
    assert breakdown.ctt == Decimal("60.00")
    assert breakdown.stt == Decimal("0.00")
    # Exchange turnover (0.0021%): 600000 * 0.000021 = 12.60
    assert breakdown.exchange_charges == Decimal("12.60")
    # SEBI: 0.60
    assert breakdown.sebi_charges == Decimal("0.60")
    assert breakdown.stamp_duty == Decimal("0.00")
    # GST: 18% of (20.00 + 12.60 + 0.60 = 33.20) = 5.976 -> 5.98
    assert breakdown.gst == Decimal("5.98")
    assert breakdown.total == Decimal("99.18")


def test_stt_sell_side_only() -> None:
    """Verify STT is charged strictly on the SELL side, never on BUY side."""
    buy_costs = CostEngine.calculate_costs(
        price=22000.0,
        quantity=25,
        side="BUY",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )
    sell_costs = CostEngine.calculate_costs(
        price=22000.0,
        quantity=25,
        side="SELL",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    assert buy_costs.stt == Decimal("0.00")
    assert sell_costs.stt > Decimal("0.00")
    assert sell_costs.stt == Decimal("275.00")


def test_stamp_duty_buy_side_only() -> None:
    """Verify State Stamp Duty is charged strictly on BUY side, never on SELL side."""
    buy_costs = CostEngine.calculate_costs(
        price=22000.0,
        quantity=25,
        side="BUY",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )
    sell_costs = CostEngine.calculate_costs(
        price=22000.0,
        quantity=25,
        side="SELL",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    # Buy side stamp duty: 550,000 * 0.00002 = 11.00
    assert buy_costs.stamp_duty == Decimal("11.00")
    assert sell_costs.stamp_duty == Decimal("0.00")


def test_gst_base() -> None:
    """
    Verify GST is applied strictly on (Brokerage + Exchange Charges + SEBI Fees),
    and NEVER on turnover or statutory taxes (STT / stamp duty).
    """
    costs = CostEngine.calculate_costs(
        price=20000.0,
        quantity=25,
        side="SELL",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    expected_gst_base = costs.brokerage + costs.exchange_charges + costs.sebi_charges
    expected_gst = (expected_gst_base * Decimal("0.18")).quantize(Decimal("0.01"))
    assert costs.gst == expected_gst


def test_brokerage_cap() -> None:
    """
    Verify brokerage cap behavior:
        - Small trade: 0.03% applied when below ₹20 cap.
        - Large trade: Capped at ₹20.
    """
    # Small trade: 10 shares @ ₹100 = turnover ₹1,000
    small = CostEngine.calculate_costs(
        price=100.0,
        quantity=10,
        side="BUY",
        schedule=NSE_EQUITY_INTRADAY_SCHEDULE_2026,
    )
    # 1,000 * 0.0003 = 0.30
    assert small.brokerage == Decimal("0.30")

    # Large trade: 10,000 shares @ ₹500 = turnover ₹5,000,000
    large = CostEngine.calculate_costs(
        price=500.0,
        quantity=10000,
        side="BUY",
        schedule=NSE_EQUITY_INTRADAY_SCHEDULE_2026,
    )
    # 5,000,000 * 0.0003 = 1,500.00, but capped at 20.00
    assert large.brokerage == Decimal("20.00")


def test_costs_use_decimal() -> None:
    """Verify all monetary outputs are instances of Decimal."""
    breakdown = CostEngine.calculate_costs(
        price=18050.25,
        quantity=50,
        side="SELL",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    assert isinstance(breakdown.turnover, Decimal)
    assert isinstance(breakdown.brokerage, Decimal)
    assert isinstance(breakdown.stt, Decimal)
    assert isinstance(breakdown.ctt, Decimal)
    assert isinstance(breakdown.exchange_charges, Decimal)
    assert isinstance(breakdown.sebi_charges, Decimal)
    assert isinstance(breakdown.stamp_duty, Decimal)
    assert isinstance(breakdown.gst, Decimal)
    assert isinstance(breakdown.total, Decimal)


def test_cost_calculation_matches_to_paisa() -> None:
    """
    Numerical discipline test: verify exact penny-accurate summation without
    binary floating-point representation errors.
    """
    breakdown = CostEngine.calculate_costs(
        price=21456.75,
        quantity=25,
        side="BUY",
        schedule=NSE_FUTURES_SCHEDULE_2026,
    )

    summed_components = (
        breakdown.brokerage
        + breakdown.stt
        + breakdown.ctt
        + breakdown.exchange_charges
        + breakdown.sebi_charges
        + breakdown.stamp_duty
        + breakdown.gst
    )

    # Exactly equals the total to the paisa
    assert breakdown.total == summed_components
    # Quantized to 2 decimal places
    assert breakdown.total == breakdown.total.quantize(Decimal("0.01"))


def test_estimate_margin() -> None:
    """Verify indicative margin calculation for derivative contracts."""
    reg = StaticContractRegistry.create_default()
    nifty = reg.get("NIFTY26APRFUT")  # lot = 25, margin_rate = 0.12

    est = estimate_margin(contract=nifty, price=22000.0, lots=2)
    # Contract value = 22000 * 25 * 2 = 1,100,000
    assert est.contract_value == Decimal("1100000.00")
    assert est.margin_rate == Decimal("0.12")
    # Margin = 1,100,000 * 0.12 = 132,000
    assert est.estimated_margin == Decimal("132000.00")
    assert "Estimated margin" in est.label


# ─── 3. Expiry & Calendar Tests ───────────────────────────────────────────────


def test_nse_expiry_weekday() -> None:
    """Verify current NSE specification uses Tuesday (weekday 1) for expiry."""
    rule = ExpiryRule(weekday=1)  # Tuesday
    # April 2026: Tuesdays are 7, 14, 21, 28
    # 28 is the last Tuesday of the month and not a holiday
    expiry = rule.get_monthly_expiry(2026, 4)

    assert expiry == date(2026, 4, 28)
    assert expiry.weekday() == 1


def test_holiday_moves_expiry_backward() -> None:
    """
    Verify holiday adjustment moves expiry backwards to the preceding trading day.
    If Tuesday 2026-04-28 is a trading holiday, expiry shifts to Monday 2026-04-27.
    """
    cal = TradingCalendar(holidays={date(2026, 4, 28)})
    rule = ExpiryRule(weekday=1, calendar=cal)

    expiry = rule.get_monthly_expiry(2026, 4)
    assert expiry == date(2026, 4, 27)  # Monday

    # If Monday was ALSO a holiday, shifts back to Friday
    cal2 = TradingCalendar(holidays={date(2026, 4, 28), date(2026, 4, 27)})
    rule2 = ExpiryRule(weekday=1, calendar=cal2)
    assert rule2.get_monthly_expiry(2026, 4) == date(2026, 4, 24)  # Friday


# ─── 4. Rollover Manager Tests ────────────────────────────────────────────────


def test_no_roll_before_threshold() -> None:
    """Verify rollover is NOT triggered when DTE is strictly greater than threshold."""
    reg = StaticContractRegistry.create_default()
    near = reg.get("NIFTY26APRFUT")  # Expiry: 2026-04-28
    next_c = reg.get("NIFTY26MAYFUT")

    manager = RolloverManager(dte_threshold=1)
    # Current date is 2026-04-25 -> DTE = 3 days > 1
    decision = manager.check_rollover(
        current_date=date(2026, 4, 25),
        near_contract=near,
        next_contract=next_c,
        position_quantity=50,
    )

    assert decision.requires_rollover is False
    assert decision.days_to_expiry == 3
    assert decision.close_order is None
    assert decision.open_order is None


def test_rollover_when_dte_threshold_reached() -> None:
    """Verify rollover triggers when DTE <= threshold."""
    reg = StaticContractRegistry.create_default()
    near = reg.get("NIFTY26APRFUT")  # Expiry: 2026-04-28
    next_c = reg.get("NIFTY26MAYFUT")

    manager = RolloverManager(dte_threshold=1)
    # Current date is 2026-04-27 -> DTE = 1 <= 1
    decision = manager.check_rollover(
        current_date=date(2026, 4, 27),
        near_contract=near,
        next_contract=next_c,
        position_quantity=50,
    )

    assert decision.requires_rollover is True
    assert decision.days_to_expiry == 1
    assert decision.close_order is not None
    assert decision.open_order is not None


def test_long_roll_preserves_direction() -> None:
    """
    Verify Long rollover mechanics:
        Long near (+50) -> SELL 50 near, BUY 50 next.
    """
    reg = StaticContractRegistry.create_default()
    near = reg.get("NIFTY26APRFUT")
    next_c = reg.get("NIFTY26MAYFUT")

    manager = RolloverManager(dte_threshold=1)
    decision = manager.check_rollover(
        current_date=date(2026, 4, 27),
        near_contract=near,
        next_contract=next_c,
        position_quantity=50,  # Long 50
    )

    assert decision.requires_rollover is True
    close = decision.close_order
    open_o = decision.open_order

    assert close is not None and open_o is not None
    assert close.symbol == "NIFTY26APRFUT"
    assert close.side == OrderSide.SELL
    assert close.quantity == 50

    assert open_o.symbol == "NIFTY26MAYFUT"
    assert open_o.side == OrderSide.BUY
    assert open_o.quantity == 50


def test_short_roll_preserves_direction() -> None:
    """
    Verify Short rollover mechanics:
        Short near (-50) -> BUY 50 near, SELL 50 next.
    """
    reg = StaticContractRegistry.create_default()
    near = reg.get("NIFTY26APRFUT")
    next_c = reg.get("NIFTY26MAYFUT")

    manager = RolloverManager(dte_threshold=1)
    decision = manager.check_rollover(
        current_date=date(2026, 4, 27),
        near_contract=near,
        next_contract=next_c,
        position_quantity=-50,  # Short 50
    )

    assert decision.requires_rollover is True
    close = decision.close_order
    open_o = decision.open_order

    assert close is not None and open_o is not None
    assert close.symbol == "NIFTY26APRFUT"
    assert close.side == OrderSide.BUY
    assert close.quantity == 50

    assert open_o.symbol == "NIFTY26MAYFUT"
    assert open_o.side == OrderSide.SELL
    assert open_o.quantity == 50


def test_roll_quantity_matches_position() -> None:
    """Verify rollover orders strictly match the absolute position size."""
    reg = StaticContractRegistry.create_default()
    near = reg.get("NIFTY26APRFUT")
    next_c = reg.get("NIFTY26MAYFUT")
    manager = RolloverManager(dte_threshold=1)

    for test_qty in [25, 75, 150, -25, -75, -150]:
        decision = manager.check_rollover(
            current_date=date(2026, 4, 28),
            near_contract=near,
            next_contract=next_c,
            position_quantity=test_qty,
        )
        assert decision.close_order is not None
        assert decision.open_order is not None
        assert decision.close_order.quantity == abs(test_qty)
        assert decision.open_order.quantity == abs(test_qty)
