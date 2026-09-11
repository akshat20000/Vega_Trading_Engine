"""
Unit and regression tests for PaperBroker and Broker interfaces.

Tests:
    - Order submission and state tracking
    - Idempotency regression test (duplicate client_order_id)
    - Public order lookup without exposing internal collections
    - Cancellation of pending orders
    - State transition errors on cancelling filled orders
    - Separation of submission and execution
    - Deterministic slippage calculation for BUY and SELL
    - Deterministic brokerage calculation
    - Fills recording and filtering
    - Deterministic failure simulation (broker failure, broker rejection)
    - Safe KiteBroker skeleton behavior
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from vega.broker.kite import KiteBroker
from vega.broker.paper import BrokerError, IdempotencyConflictError, PaperBroker
from vega.config import VegaConfig
from vega.orders.models import (
    Fill,
    InvalidOrderStateTransitionError,
    Order,
    OrderSide,
    OrderStatus,
)


@pytest.fixture
def sample_timestamp() -> datetime:
    return datetime(2026, 9, 11, 9, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def paper_broker() -> PaperBroker:
    # Use deterministic custom rates: 0.1% slippage, 0.03% brokerage
    return PaperBroker(slippage_pct=0.001, brokerage_pct=0.0003)


# ── Idempotency Regression Tests ─────────────────────────────────────────────


def test_idempotency_case_a_identical_parameters(paper_broker: PaperBroker) -> None:
    """
    CASE A:
    Same client_order_id + identical order parameters (symbol, side, quantity, price)
    -> Accepted as idempotent duplicate
    -> Returns the existing order instance
    -> No duplicate order or extra position is created
    """
    order1 = Order(
        client_order_id="GRID-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=2,
        price=22000.0,
    )
    res1 = paper_broker.place_order(order1)
    assert res1.client_order_id == "GRID-001"
    assert len(paper_broker.get_all_orders()) == 1

    # Second submission with identical parameters
    order2 = Order(
        client_order_id="GRID-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=2,
        price=22000.0,
    )
    res2 = paper_broker.place_order(order2)

    # Must return the existing order instance
    assert res2 is res1
    assert res2.quantity == 2
    assert len(paper_broker.get_all_orders()) == 1


@pytest.mark.parametrize(
    "conflicting_kwargs",
    [
        {"side": OrderSide.SELL},
        {"quantity": 5},
        {"price": 22500.0},
        {"symbol": "BANKNIFTY"},
    ],
)
def test_idempotency_case_b_conflicting_parameters_rejected(
    paper_broker: PaperBroker, conflicting_kwargs: dict
) -> None:
    """
    CASE B:
    Same client_order_id + conflicting order parameters
    -> Rejected with IdempotencyConflictError
    -> Does NOT silently return original order
    -> Keeps the original order completely unchanged
    """
    base_kwargs = {
        "client_order_id": "GRID-001",
        "symbol": "NIFTY50",
        "side": OrderSide.BUY,
        "quantity": 2,
        "price": 22000.0,
    }
    order1 = Order(**base_kwargs)
    res1 = paper_broker.place_order(order1)
    assert len(paper_broker.get_all_orders()) == 1

    # Create conflicting order
    bad_kwargs = {**base_kwargs, **conflicting_kwargs}
    conflicting_order = Order(**bad_kwargs)

    with pytest.raises(IdempotencyConflictError, match="Idempotency conflict for client_order_id"):
        paper_broker.place_order(conflicting_order)

    # Original order must remain completely unchanged
    tracked = paper_broker.get_order("GRID-001")
    assert tracked is not None
    assert tracked.symbol == "NIFTY50"
    assert tracked.side == OrderSide.BUY
    assert tracked.quantity == 2
    assert tracked.price == 22000.0
    assert len(paper_broker.get_all_orders()) == 1


# ── Order Submission and Public Interface ────────────────────────────────────


def test_order_submission_status_is_pending(paper_broker: PaperBroker) -> None:
    """Submitted orders must be stored with PENDING status."""
    order = Order(
        client_order_id="SUB-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
    )
    placed = paper_broker.place_order(order)
    assert placed.status == OrderStatus.PENDING

    # Public lookup
    found = paper_broker.get_order("SUB-001")
    assert found is not None
    assert found.client_order_id == "SUB-001"

    # Non-existent order returns None
    assert paper_broker.get_order("DOES-NOT-EXIST") is None


def test_place_order_non_pending_raises(paper_broker: PaperBroker) -> None:
    """Submitting an order that is not in PENDING status must raise ValueError."""
    order = Order(
        client_order_id="ORD-FILLED",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
        status=OrderStatus.FILLED,
    )
    with pytest.raises(ValueError, match="New orders must be submitted in PENDING status"):
        paper_broker.place_order(order)


def test_public_methods_do_not_expose_internal_dict(paper_broker: PaperBroker) -> None:
    """get_all_orders returns a copy of the list; modifying it does not mutate broker state."""
    order = Order(
        client_order_id="PUB-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
    )
    paper_broker.place_order(order)

    orders_list = paper_broker.get_all_orders()
    assert len(orders_list) == 1
    orders_list.clear()  # Caller mutates returned list

    # Broker internal orders must remain unaffected
    assert len(paper_broker.get_all_orders()) == 1


# ── Cancellation ─────────────────────────────────────────────────────────────


def test_cancel_pending_order(paper_broker: PaperBroker) -> None:
    """Cancelling a PENDING order transitions its status to CANCELLED."""
    order = Order(
        client_order_id="CAN-001",
        symbol="NIFTY50",
        side=OrderSide.SELL,
        quantity=3,
    )
    paper_broker.place_order(order)
    cancelled = paper_broker.cancel_order("CAN-001")
    assert cancelled.status == OrderStatus.CANCELLED
    assert "Cancelled" in cancelled.reason


def test_cancel_non_existent_order_raises(paper_broker: PaperBroker) -> None:
    """Cancelling an unknown order raises KeyError."""
    with pytest.raises(KeyError, match="Order 'UNKNOWN' not found"):
        paper_broker.cancel_order("UNKNOWN")


def test_cancel_already_filled_order_raises(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """Attempting to cancel an order that has already been FILLED must raise InvalidOrderStateTransitionError."""
    order = Order(
        client_order_id="CAN-FILLED",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
    )
    paper_broker.place_order(order)
    paper_broker.execute_order("CAN-FILLED", market_price=22000.0, timestamp=sample_timestamp)

    with pytest.raises(InvalidOrderStateTransitionError):
        paper_broker.cancel_order("CAN-FILLED")


# ── Execution Semantics, Slippage & Brokerage ─────────────────────────────────


def test_submission_separate_from_execution(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """
    Placing an order enqueues it as PENDING.
    It is NOT executed until execute_order / execute_pending_orders is called.
    """
    order = Order(
        client_order_id="SEP-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=1,
    )
    paper_broker.place_order(order)
    assert order.status == OrderStatus.PENDING
    assert len(paper_broker.get_fills()) == 0

    fills = paper_broker.execute_pending_orders(market_price=20000.0, timestamp=sample_timestamp)
    assert len(fills) == 1
    assert order.status == OrderStatus.FILLED
    assert len(paper_broker.get_fills()) == 1


def test_slippage_and_brokerage_buy(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """
    BUY order slippage:
        fill_price = market_price * (1 + slippage_pct)
        slippage = abs(fill_price - market_price)
        brokerage = fill_price * quantity * brokerage_pct
    """
    order = Order(
        client_order_id="BUY-001",
        symbol="NIFTY50",
        side=OrderSide.BUY,
        quantity=2,
    )
    paper_broker.place_order(order)

    # market_price = 20,000.0, slippage_pct = 0.001 (0.1%), brokerage_pct = 0.0003 (0.03%)
    # expected raw fill_price = 20000 * 1.001 = 20020.0
    # expected slippage = 20.0
    # expected brokerage = 20020.0 * 2 * 0.0003 = 12.012 -> 12.01
    fill = paper_broker.execute_order("BUY-001", market_price=20000.0, timestamp=sample_timestamp)

    assert fill.order_id == "BUY-001"
    assert fill.filled_price == 20020.0
    assert fill.filled_qty == 2
    assert fill.slippage == 20.0
    assert fill.brokerage == 12.01
    assert fill.timestamp == sample_timestamp


def test_slippage_and_brokerage_sell(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """
    SELL order slippage:
        fill_price = market_price * (1 - slippage_pct)
        slippage = abs(fill_price - market_price)
        brokerage = fill_price * quantity * brokerage_pct
    """
    order = Order(
        client_order_id="SELL-001",
        symbol="NIFTY50",
        side=OrderSide.SELL,
        quantity=4,
    )
    paper_broker.place_order(order)

    # market_price = 20,000.0, slippage_pct = 0.001 (0.1%), brokerage_pct = 0.0003 (0.03%)
    # expected raw fill_price = 20000 * (1 - 0.001) = 19980.0
    # expected slippage = 20.0
    # expected brokerage = 19980.0 * 4 * 0.0003 = 23.976 -> 23.98
    fill = paper_broker.execute_order("SELL-001", market_price=20000.0, timestamp=sample_timestamp)

    assert fill.order_id == "SELL-001"
    assert fill.filled_price == 19980.0
    assert fill.filled_qty == 4
    assert fill.slippage == 20.0
    assert fill.brokerage == 23.98


def test_get_fills_filter_by_order_id(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """get_fills can return all fills or filter by a specific client_order_id."""
    o1 = Order(client_order_id="ORD-A", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    o2 = Order(client_order_id="ORD-B", symbol="NIFTY50", side=OrderSide.SELL, quantity=2)
    paper_broker.place_order(o1)
    paper_broker.place_order(o2)

    paper_broker.execute_pending_orders(market_price=21000.0, timestamp=sample_timestamp)

    assert len(paper_broker.get_fills()) == 2
    assert len(paper_broker.get_fills("ORD-A")) == 1
    assert paper_broker.get_fills("ORD-A")[0].order_id == "ORD-A"
    assert len(paper_broker.get_fills("NON-EXISTENT")) == 0


def test_execute_already_filled_order_raises(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """Executing an order that is already FILLED must raise ValueError."""
    order = Order(client_order_id="DOUBLE-EXEC", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    paper_broker.place_order(order)
    paper_broker.execute_order("DOUBLE-EXEC", market_price=20000.0, timestamp=sample_timestamp)

    with pytest.raises(ValueError, match="Cannot execute order 'DOUBLE-EXEC' with status FILLED"):
        paper_broker.execute_order("DOUBLE-EXEC", market_price=20000.0, timestamp=sample_timestamp)


def test_execute_invalid_market_price_raises(
    paper_broker: PaperBroker, sample_timestamp: datetime
) -> None:
    """Executing with non-positive market price must raise ValueError."""
    order = Order(client_order_id="NEG-PX", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    paper_broker.place_order(order)
    with pytest.raises(ValueError, match="market_price must be positive"):
        paper_broker.execute_order("NEG-PX", market_price=-10.0, timestamp=sample_timestamp)


# ── Failure & Rejection Simulation ───────────────────────────────────────────


def test_simulated_broker_failure(paper_broker: PaperBroker) -> None:
    """When simulate_broker_failure is active, place_order and cancel_order raise BrokerError."""
    paper_broker.simulate_broker_failure = True
    order = Order(client_order_id="FAIL-01", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)

    with pytest.raises(BrokerError, match="Simulated broker network/system failure"):
        paper_broker.place_order(order)

    # Disable failure to place order, then test cancel_order failure
    paper_broker.simulate_broker_failure = False
    paper_broker.place_order(order)

    paper_broker.simulate_broker_failure = True
    with pytest.raises(BrokerError, match="Simulated broker network/system failure"):
        paper_broker.cancel_order("FAIL-01")


def test_simulated_order_rejection(paper_broker: PaperBroker) -> None:
    """When simulate_order_rejection is active, placed orders transition to REJECTED."""
    paper_broker.simulate_order_rejection = True
    order = Order(client_order_id="REJ-01", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)
    placed = paper_broker.place_order(order)

    assert placed.status == OrderStatus.REJECTED
    assert "rejection" in placed.reason.lower()


# ── Safe KiteBroker Skeleton Verification ────────────────────────────────────


def test_safe_kite_broker_skeleton() -> None:
    """All KiteBroker operations must raise NotImplementedError for safety."""
    kite = KiteBroker()
    order = Order(client_order_id="KITE-01", symbol="NIFTY50", side=OrderSide.BUY, quantity=1)

    with pytest.raises(NotImplementedError, match="KiteBroker.place_order is disabled"):
        kite.place_order(order)

    with pytest.raises(NotImplementedError, match="KiteBroker.cancel_order is disabled"):
        kite.cancel_order("KITE-01")

    with pytest.raises(NotImplementedError, match="KiteBroker.get_order is disabled"):
        kite.get_order("KITE-01")

    with pytest.raises(NotImplementedError, match="KiteBroker.get_all_orders is disabled"):
        kite.get_all_orders()

    with pytest.raises(NotImplementedError, match="KiteBroker.get_fills is disabled"):
        kite.get_fills()
