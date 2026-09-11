"""
Unit tests for RiskManager and risk validation controls.

Tests:
    - Kill switch activation, deactivation, and order blocking
    - Macro circuit breaker rejection
    - Daily loss limit (inclusive boundary, positive vs negative PnL)
    - Single order quantity cap
    - Resulting net position cap (long and short)
    - Bearish regime position cap override
    - Pyramiding limit (same-direction additions vs position reductions)
    - RiskDecision boolean evaluation and explanation strings
"""

from __future__ import annotations

import pytest

from vega.config import VegaConfig
from vega.orders.models import Order, OrderSide
from vega.risk import OrderEffect, RiskDecision, RiskManager


@pytest.fixture
def risk_manager() -> RiskManager:
    """Fixture providing a standard RiskManager."""
    return RiskManager(
        max_position_size=5,
        max_order_quantity=5,
        max_pyramids=3,
        daily_loss_limit=10_000.0,
    )


@pytest.fixture
def sample_buy_order() -> Order:
    return Order(
        client_order_id="BUY-RISK-01",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=2,
    )


# ── Kill Switch Tests ────────────────────────────────────────────────────────


def test_kill_switch_default_inactive(risk_manager: RiskManager, sample_buy_order: Order) -> None:
    """Kill switch is inactive by default and does not block orders."""
    assert not risk_manager.is_kill_switch_active()
    decision = risk_manager.validate_order(sample_buy_order)
    assert decision.is_allowed
    assert risk_manager.is_order_allowed(sample_buy_order)


def test_kill_switch_activation_blocks_orders(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """Enabling kill switch rejects orders immediately with the specified reason."""
    risk_manager.enable_kill_switch(reason="Extreme market volatility detected")
    assert risk_manager.is_kill_switch_active()

    decision = risk_manager.validate_order(sample_buy_order)
    assert not decision.is_allowed
    assert not risk_manager.is_order_allowed(sample_buy_order)
    assert "Kill switch is ACTIVE" in decision.reason
    assert "Extreme market volatility" in decision.reason


def test_kill_switch_deactivation_resumes_orders(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """Disabling kill switch clears the active state and allows orders through."""
    risk_manager.enable_kill_switch()
    assert risk_manager.is_kill_switch_active()

    risk_manager.disable_kill_switch()
    assert not risk_manager.is_kill_switch_active()

    decision = risk_manager.validate_order(sample_buy_order)
    assert decision.is_allowed


# ── Macro Circuit Breaker Tests ──────────────────────────────────────────────


def test_macro_circuit_breaker_blocks_orders(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """Active macro circuit breaker must immediately reject proposed orders."""
    decision = risk_manager.validate_order(
        sample_buy_order,
        macro_circuit_breaker=True,
    )
    assert not decision.is_allowed
    assert "Macro circuit breaker is active" in decision.reason


def test_macro_circuit_breaker_inactive_allows_orders(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """Inactive macro circuit breaker does not block orders."""
    decision = risk_manager.validate_order(
        sample_buy_order,
        macro_circuit_breaker=False,
    )
    assert decision.is_allowed


# ── Daily Loss Limit Tests ───────────────────────────────────────────────────


def test_daily_loss_limit_under_threshold(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """daily_realized_pnl of -8,000 INR with limit 10,000 INR is allowed."""
    decision = risk_manager.validate_order(
        sample_buy_order,
        daily_realized_pnl=-8_000.0,
    )
    assert decision.is_allowed


def test_daily_loss_limit_at_boundary_is_rejected(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """
    Inclusive boundary: daily_realized_pnl of -10,000 INR with limit 10,000 INR
    is rejected (daily_realized_pnl <= -limit).
    """
    decision = risk_manager.validate_order(
        sample_buy_order,
        daily_realized_pnl=-10_000.0,
    )
    assert not decision.is_allowed
    assert "Daily loss limit breached" in decision.reason


def test_daily_loss_limit_breached_is_rejected(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """daily_realized_pnl of -12,000 INR with limit 10,000 INR is rejected."""
    decision = risk_manager.validate_order(
        sample_buy_order,
        daily_realized_pnl=-12_000.0,
    )
    assert not decision.is_allowed
    assert "Daily loss limit breached" in decision.reason


def test_daily_loss_limit_positive_pnl_allowed(
    risk_manager: RiskManager, sample_buy_order: Order
) -> None:
    """Profitable day allows new orders without hindrance."""
    decision = risk_manager.validate_order(
        sample_buy_order,
        daily_realized_pnl=25_000.0,
    )
    assert decision.is_allowed


# ── Single Order Quantity Limit Tests ────────────────────────────────────────


def test_max_order_quantity_within_limit(risk_manager: RiskManager) -> None:
    """Order with quantity <= max_order_quantity (5) is allowed."""
    order = Order(client_order_id="O-5", symbol="NIFTY50", side=OrderSide.BUY, quantity=5)
    decision = risk_manager.validate_order(order)
    assert decision.is_allowed


def test_max_order_quantity_exceeded(risk_manager: RiskManager) -> None:
    """Order with quantity > max_order_quantity (5) is rejected."""
    order = Order(client_order_id="O-6", symbol="NIFTY50", side=OrderSide.BUY, quantity=6)
    decision = risk_manager.validate_order(order)
    assert not decision.is_allowed
    assert "exceeds max order quantity" in decision.reason


# ── Position Cap Tests ───────────────────────────────────────────────────────


def test_position_cap_scenarios(risk_manager: RiskManager) -> None:
    """
    Verify position cap mechanics (max_position_size = 5):
        - current position = +3, BUY 2 -> resulting +5 -> allowed
        - current position = +3, BUY 3 -> resulting +6 -> rejected
        - current position = +3, SELL 5 -> resulting -2 -> allowed (| -2 | <= 5)
    """
    # 1. current = +3, BUY 2 -> +5 <= 5 (allowed)
    buy2 = Order(client_order_id="BUY-2", symbol="NIFTY50", side=OrderSide.BUY, quantity=2)
    assert risk_manager.validate_order(buy2, current_position=3).is_allowed

    # 2. current = +3, BUY 3 -> +6 > 5 (rejected)
    buy3 = Order(client_order_id="BUY-3", symbol="NIFTY50", side=OrderSide.BUY, quantity=3)
    d2 = risk_manager.validate_order(buy3, current_position=3)
    assert not d2.is_allowed
    assert "exceeds position cap" in d2.reason

    # 3. current = +3, SELL 5 -> -2 (| -2 | <= 5, allowed)
    sell5 = Order(client_order_id="SELL-5", symbol="NIFTY50", side=OrderSide.SELL, quantity=5)
    assert risk_manager.validate_order(sell5, current_position=3).is_allowed


def test_position_cap_short_side(risk_manager: RiskManager) -> None:
    """Short positions also obey the absolute position cap."""
    # current = -3, SELL 3 -> -6 (rejected)
    sell3 = Order(client_order_id="SELL-3", symbol="NIFTY50", side=OrderSide.SELL, quantity=3)
    d = risk_manager.validate_order(sell3, current_position=-3)
    assert not d.is_allowed
    assert "exceeds position cap" in d.reason

    # current = -3, BUY 4 -> +1 (allowed)
    buy4 = Order(client_order_id="BUY-4", symbol="NIFTY50", side=OrderSide.BUY, quantity=4)
    assert risk_manager.validate_order(buy4, current_position=-3).is_allowed


def test_position_cap_override_bearish_regime(risk_manager: RiskManager) -> None:
    """Macro regime override (e.g. bearish cap = 2) constrains position size further."""
    buy3 = Order(client_order_id="BUY-3", symbol="NIFTY50", side=OrderSide.BUY, quantity=3)
    # Default cap is 5, but override is 2
    decision = risk_manager.validate_order(buy3, current_position=0, position_cap_override=2)
    assert not decision.is_allowed
    assert "exceeds position cap (2)" in decision.reason

    # Order of 2 units passes under cap of 2
    buy2 = Order(client_order_id="BUY-2", symbol="NIFTY50", side=OrderSide.BUY, quantity=2)
    assert risk_manager.validate_order(buy2, current_position=0, position_cap_override=2).is_allowed


# ── Pyramiding, Reduction, and Reversal Regression Tests ────────────────────


def test_classify_order_effect_coverage(risk_manager: RiskManager) -> None:
    """Verify explicit classification of order effect across all scenarios."""
    # 1. NEW_ENTRY (from flat)
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O1", symbol="NIFTY50", side=OrderSide.BUY, quantity=2),
        current_position=0,
    ) == OrderEffect.NEW_ENTRY

    # 2. SAME_DIRECTION_ENTRY (adding long or adding short)
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O2", symbol="NIFTY50", side=OrderSide.BUY, quantity=1),
        current_position=3,
    ) == OrderEffect.SAME_DIRECTION_ENTRY
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O3", symbol="NIFTY50", side=OrderSide.SELL, quantity=1),
        current_position=-3,
    ) == OrderEffect.SAME_DIRECTION_ENTRY

    # 3. PARTIAL_REDUCTION
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O4", symbol="NIFTY50", side=OrderSide.SELL, quantity=1),
        current_position=3,
    ) == OrderEffect.PARTIAL_REDUCTION
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O5", symbol="NIFTY50", side=OrderSide.BUY, quantity=1),
        current_position=-3,
    ) == OrderEffect.PARTIAL_REDUCTION

    # 4. COMPLETE_REDUCTION
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O6", symbol="NIFTY50", side=OrderSide.SELL, quantity=3),
        current_position=3,
    ) == OrderEffect.COMPLETE_REDUCTION
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O7", symbol="NIFTY50", side=OrderSide.BUY, quantity=3),
        current_position=-3,
    ) == OrderEffect.COMPLETE_REDUCTION

    # 5. REVERSAL
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O8", symbol="NIFTY50", side=OrderSide.SELL, quantity=5),
        current_position=3,
    ) == OrderEffect.REVERSAL
    assert risk_manager.classify_order_effect(
        Order(client_order_id="O9", symbol="NIFTY50", side=OrderSide.BUY, quantity=5),
        current_position=-3,
    ) == OrderEffect.REVERSAL


def test_same_direction_entry_at_pyramid_limit(risk_manager: RiskManager) -> None:
    """
    1. Same-direction entry at pyramid limit:
       Current LONG +3, max_pyramids = 3:
       BUY 1 is a SAME_DIRECTION_ENTRY.
       When current_pyramids == 3, pyramiding check must reject it.
    """
    buy = Order(client_order_id="BUY-LIM", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)

    # Below limit: 2 open legs -> allowed
    d_ok = risk_manager.validate_order(buy, current_position=3, current_pyramids=2)
    assert d_ok.is_allowed
    assert d_ok.order_effect == OrderEffect.SAME_DIRECTION_ENTRY

    # At limit: 3 open legs -> rejected
    d_rej = risk_manager.validate_order(buy, current_position=3, current_pyramids=3)
    assert not d_rej.is_allowed
    assert d_rej.order_effect == OrderEffect.SAME_DIRECTION_ENTRY
    assert "Pyramiding limit reached" in d_rej.reason


def test_reduction_allowed_at_pyramid_limit(risk_manager: RiskManager) -> None:
    """
    2. Reduction allowed at pyramid limit:
       Current LONG +3, max_pyramids = 3, current_pyramids = 3:
       SELL 1 is a PARTIAL_REDUCTION.
       Pyramiding check must NOT block it.
    """
    sell1 = Order(client_order_id="SELL-RED-1", symbol="NIFTY50", side=OrderSide.SELL, quantity=1)
    d_long = risk_manager.validate_order(sell1, current_position=3, current_pyramids=3)
    assert d_long.is_allowed
    assert d_long.order_effect == OrderEffect.PARTIAL_REDUCTION

    # Same for short side: Current SHORT -3, BUY 1 is a PARTIAL_REDUCTION
    buy1 = Order(client_order_id="BUY-RED-1", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    d_short = risk_manager.validate_order(buy1, current_position=-3, current_pyramids=3)
    assert d_short.is_allowed
    assert d_short.order_effect == OrderEffect.PARTIAL_REDUCTION


def test_complete_reduction(risk_manager: RiskManager) -> None:
    """
    3. Complete reduction:
       Current LONG +3:
       SELL 3 completely reduces/flattens the position to 0.
       Must be explicitly classified as COMPLETE_REDUCTION and allowed.
    """
    sell3 = Order(client_order_id="SELL-FLAT", symbol="NIFTY50", side=OrderSide.SELL, quantity=3)
    d_long = risk_manager.validate_order(sell3, current_position=3, current_pyramids=3)
    assert d_long.is_allowed
    assert d_long.order_effect == OrderEffect.COMPLETE_REDUCTION

    # Same for short side: Current SHORT -3, BUY 3 flattens to 0
    buy3 = Order(client_order_id="BUY-FLAT", symbol="NIFTY50", side=OrderSide.BUY, quantity=3)
    d_short = risk_manager.validate_order(buy3, current_position=-3, current_pyramids=3)
    assert d_short.is_allowed
    assert d_short.order_effect == OrderEffect.COMPLETE_REDUCTION


def test_reversal_semantics(risk_manager: RiskManager) -> None:
    """
    4. Reversal semantics:
       Current LONG +3, max_position_size = 5, max_order_quantity = 5:
       SELL 5 closes 3 LONG and opens 2 SHORT.
       - Explicitly classified as REVERSAL (not merely a reduction).
       - Permitted if the resulting short position (-2) and order quantity (5) are within limits.
       - Not blocked by the old LONG position's pyramid count (3), because the old position is closed.
    """
    sell5 = Order(client_order_id="SELL-REV", symbol="NIFTY50", side=OrderSide.SELL, quantity=5)
    decision = risk_manager.validate_order(sell5, current_position=3, current_pyramids=3)
    assert decision.is_allowed
    assert decision.order_effect == OrderEffect.REVERSAL

    # Short to Long reversal: Current SHORT -3, BUY 5 closes 3 SHORT and opens 2 LONG
    buy5 = Order(client_order_id="BUY-REV", symbol="NIFTY50", side=OrderSide.BUY, quantity=5)
    d_rev_long = risk_manager.validate_order(buy5, current_position=-3, current_pyramids=3)
    assert d_rev_long.is_allowed
    assert d_rev_long.order_effect == OrderEffect.REVERSAL


def test_reversal_violating_position_cap_rejected(risk_manager: RiskManager) -> None:
    """
    Reversal that would create an oversized opposite position is rejected:
    Current LONG +3, max_position_size = 5, max_order_quantity = 10:
    SELL 9 closes 3 LONG and attempts to open 6 SHORT.
    Resulting position -6 exceeds position cap 5 -> rejected.
    """
    rm = RiskManager(max_position_size=5, max_order_quantity=10, max_pyramids=3)
    sell9 = Order(client_order_id="SELL-BIG-REV", symbol="NIFTY50", side=OrderSide.SELL, quantity=9)
    decision = rm.validate_order(sell9, current_position=3, current_pyramids=3)
    assert not decision.is_allowed
    assert decision.order_effect == OrderEffect.REVERSAL
    assert "exceeds position cap" in decision.reason
