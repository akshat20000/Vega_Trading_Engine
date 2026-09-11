"""
Unit tests for Order and Fill models and the Order State Machine.

Tests:
    - Order construction validation (client_order_id, symbol, quantity, price, side, status)
    - Valid state transitions: PENDING -> FILLED, PENDING -> CANCELLED, PENDING -> REJECTED
    - Invalid state transitions (e.g. FILLED -> PENDING, FILLED -> FILLED, CANCELLED -> FILLED, REJECTED -> FILLED)
    - Fill construction validation (order_id, filled_price, filled_qty, brokerage, timestamp)
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from vega.orders.models import (
    Fill,
    InvalidOrderStateTransitionError,
    Order,
    OrderSide,
    OrderStatus,
)


# ── Order Construction & Validation ──────────────────────────────────────────


def test_order_valid_construction() -> None:
    """Test valid Order instantiation with default and explicit parameters."""
    order = Order(
        client_order_id="ORD-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=2,
        price=22500.0,
    )
    assert order.client_order_id == "ORD-001"
    assert order.symbol == "NIFTY50"
    assert order.side == OrderSide.BUY
    assert order.quantity == 2
    assert order.price == 22500.0
    assert order.status == OrderStatus.PENDING
    assert order.reason == ""


def test_order_empty_client_order_id_raises() -> None:
    """An empty or whitespace client_order_id must raise ValueError."""
    with pytest.raises(ValueError, match="client_order_id must be a non-empty string"):
        Order(client_order_id="", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)

    with pytest.raises(ValueError, match="client_order_id must be a non-empty string"):
        Order(client_order_id="   ", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)


def test_order_empty_symbol_raises() -> None:
    """An empty or whitespace symbol must raise ValueError."""
    with pytest.raises(ValueError, match="symbol must be a non-empty string"):
        Order(client_order_id="ORD-001", symbol="", side=OrderSide.BUY, quantity=1)


def test_order_invalid_side_raises() -> None:
    """Non-OrderSide instance must raise ValueError."""
    with pytest.raises(ValueError, match="side must be an instance of OrderSide"):
        Order(
            client_order_id="ORD-001",
            symbol="NIFTY50",
            side="BUY",  # type: ignore[arg-type]
            quantity=1,
        )


def test_order_invalid_quantity_raises() -> None:
    """Quantity <= 0 or non-int must raise ValueError."""
    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        Order(client_order_id="ORD-001", symbol="NIFTY50", side=OrderSide.BUY, quantity=0)

    with pytest.raises(ValueError, match="quantity must be a positive integer"):
        Order(client_order_id="ORD-001", symbol="NIFTY50", side=OrderSide.BUY, quantity=-2)


def test_order_invalid_price_raises() -> None:
    """Negative price must raise ValueError."""
    with pytest.raises(ValueError, match="price must be >= 0.0"):
        Order(
            client_order_id="ORD-001",
            symbol="NIFTY50",
            side=OrderSide.BUY,
            quantity=1,
            price=-100.0,
        )


def test_order_invalid_status_raises() -> None:
    """Status that is not an OrderStatus enum member must raise ValueError."""
    with pytest.raises(ValueError, match="status must be an instance of OrderStatus"):
        Order(
            client_order_id="ORD-001",
            symbol="NIFTY50",
            side=OrderSide.BUY,
            quantity=1,
            status="PENDING",  # type: ignore[arg-type]
        )


# ── Order State Machine Transitions ──────────────────────────────────────────


def test_valid_transitions() -> None:
    """Test standard valid lifecycle transitions from PENDING."""
    # 1. PENDING -> FILLED
    o1 = Order(client_order_id="O1", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    assert o1.status == OrderStatus.PENDING
    o1.mark_filled()
    assert o1.status == OrderStatus.FILLED

    # 2. PENDING -> CANCELLED
    o2 = Order(client_order_id="O2", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    o2.mark_cancelled("User cancelled order")
    assert o2.status == OrderStatus.CANCELLED
    assert o2.reason == "User cancelled order"

    # 3. PENDING -> REJECTED
    o3 = Order(client_order_id="O3", symbol="NIFTY50", side=OrderSide.SELL, quantity=1)
    o3.mark_rejected("Risk check failed")
    assert o3.status == OrderStatus.REJECTED
    assert o3.reason == "Risk check failed"


@pytest.mark.parametrize(
    "initial_status,target_status",
    [
        (OrderStatus.FILLED, OrderStatus.PENDING),
        (OrderStatus.FILLED, OrderStatus.FILLED),
        (OrderStatus.FILLED, OrderStatus.CANCELLED),
        (OrderStatus.FILLED, OrderStatus.REJECTED),
        (OrderStatus.CANCELLED, OrderStatus.PENDING),
        (OrderStatus.CANCELLED, OrderStatus.FILLED),
        (OrderStatus.CANCELLED, OrderStatus.CANCELLED),
        (OrderStatus.CANCELLED, OrderStatus.REJECTED),
        (OrderStatus.REJECTED, OrderStatus.PENDING),
        (OrderStatus.REJECTED, OrderStatus.FILLED),
        (OrderStatus.REJECTED, OrderStatus.CANCELLED),
        (OrderStatus.REJECTED, OrderStatus.REJECTED),
    ],
)
def test_invalid_terminal_transitions_raise(
    initial_status: OrderStatus, target_status: OrderStatus
) -> None:
    """Terminal states (FILLED, CANCELLED, REJECTED) cannot transition to any other status."""
    order = Order(
        client_order_id="O-TERM",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
        status=initial_status,
    )
    with pytest.raises(InvalidOrderStateTransitionError, match="Cannot transition order"):
        order.transition_to(target_status)


def test_transition_to_invalid_type_raises() -> None:
    """Calling transition_to with a non-OrderStatus argument raises ValueError."""
    order = Order(client_order_id="O-BAD", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    with pytest.raises(ValueError, match="target_status must be an OrderStatus"):
        order.transition_to("FILLED")  # type: ignore[arg-type]


# ── Fill Construction & Validation ───────────────────────────────────────────


def test_fill_valid_construction() -> None:
    """Test valid Fill construction."""
    ts = datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)
    fill = Fill(
        order_id="ORD-100",
        filled_price=22550.0,
        filled_qty=2,
        timestamp=ts,
        slippage=22.55,
        brokerage=13.53,
    )
    assert fill.order_id == "ORD-100"
    assert fill.filled_price == 22550.0
    assert fill.filled_qty == 2
    assert fill.timestamp == ts
    assert fill.slippage == 22.55
    assert fill.brokerage == 13.53


def test_fill_invalid_parameters_raise() -> None:
    """Fill validation rejects empty order_id, non-positive price/qty, negative brokerage."""
    ts = datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)

    # Empty order_id
    with pytest.raises(ValueError, match="order_id must be a non-empty string"):
        Fill(order_id="", filled_price=22500.0, filled_qty=1, timestamp=ts)

    # Non-positive filled_price
    with pytest.raises(ValueError, match="filled_price must be positive"):
        Fill(order_id="O1", filled_price=0.0, filled_qty=1, timestamp=ts)

    with pytest.raises(ValueError, match="filled_price must be positive"):
        Fill(order_id="O1", filled_price=-10.0, filled_qty=1, timestamp=ts)

    # Non-positive filled_qty
    with pytest.raises(ValueError, match="filled_qty must be a positive integer"):
        Fill(order_id="O1", filled_price=22500.0, filled_qty=0, timestamp=ts)

    # Non-datetime timestamp
    with pytest.raises(ValueError, match="timestamp must be a datetime"):
        Fill(order_id="O1", filled_price=22500.0, filled_qty=1, timestamp="2026-09-11")  # type: ignore[arg-type]

    # Negative brokerage
    with pytest.raises(ValueError, match="brokerage must be non-negative"):
        Fill(order_id="O1", filled_price=22500.0, filled_qty=1, timestamp=ts, brokerage=-5.0)
