"""
Order and Fill models for the Vega Quant Trading Engine.

Defines:
    - OrderSide: Enum (BUY, SELL)
    - OrderStatus: Enum (PENDING, FILLED, CANCELLED, REJECTED)
    - InvalidOrderStateTransitionError: Raised on illegal state machine transitions
    - Order: Dataclass representing an order with state machine transitions
    - Fill: Dataclass representing an execution fill
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    """Side of an order: BUY or SELL."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """
    Lifecycle status of an order.

    Normal lifecycles:
        PENDING -> FILLED
        PENDING -> CANCELLED
        PENDING -> REJECTED
    """

    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class InvalidOrderStateTransitionError(ValueError):
    """Raised when an illegal order status transition is attempted."""

    pass


# Legal state transitions from a given source status
_LEGAL_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
    },
    OrderStatus.FILLED: set(),     # Terminal state
    OrderStatus.CANCELLED: set(),  # Terminal state
    OrderStatus.REJECTED: set(),   # Terminal state
}


@dataclass
class Order:
    """
    Represents an order placed by a strategy or execution agent.

    Attributes:
        client_order_id: Unique strategy-generated idempotency identifier (e.g. "GRID-NIFTY-001").
        symbol: The instrument symbol (e.g. "NIFTY50").
        side: OrderSide.BUY or OrderSide.SELL.
        quantity: Positive integer number of units/lots.
        price: Order price. 0.0 represents a market order.
        status: Current OrderStatus. Defaults to PENDING.
        reason: Explanatory note (populated if rejected or cancelled).
    """

    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate order attributes on construction."""
        if not isinstance(self.client_order_id, str) or not self.client_order_id.strip():
            raise ValueError("client_order_id must be a non-empty string.")

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

        if not isinstance(self.side, OrderSide):
            raise ValueError(f"side must be an instance of OrderSide, got {self.side}")

        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError(f"quantity must be a positive integer, got {self.quantity}")

        if not isinstance(self.price, (int, float)) or self.price < 0.0:
            raise ValueError(f"price must be >= 0.0, got {self.price}")
        self.price = float(self.price)

        if not isinstance(self.status, OrderStatus):
            raise ValueError(f"status must be an instance of OrderStatus, got {self.status}")

    def transition_to(self, target_status: OrderStatus, reason: str = "") -> None:
        """
        Transition order to a new status according to state machine rules.

        Raises:
            InvalidOrderStateTransitionError: If the transition is not permitted.
        """
        if not isinstance(target_status, OrderStatus):
            raise ValueError(f"target_status must be an OrderStatus, got {target_status}")

        allowed = _LEGAL_TRANSITIONS.get(self.status, set())
        if target_status not in allowed:
            raise InvalidOrderStateTransitionError(
                f"Cannot transition order '{self.client_order_id}' from "
                f"{self.status.value} to {target_status.value}."
            )

        self.status = target_status
        if reason:
            self.reason = reason

    def mark_filled(self) -> None:
        """Transition order from PENDING to FILLED."""
        self.transition_to(OrderStatus.FILLED)

    def mark_cancelled(self, reason: str = "Cancelled by client") -> None:
        """Transition order from PENDING to CANCELLED."""
        self.transition_to(OrderStatus.CANCELLED, reason=reason)

    def mark_rejected(self, reason: str = "Rejected by risk or broker") -> None:
        """Transition order from PENDING to REJECTED."""
        self.transition_to(OrderStatus.REJECTED, reason=reason)


@dataclass
class Fill:
    """
    Represents an execution fill for an order.

    Attributes:
        order_id: The client_order_id of the filled order.
        filled_price: The executed price (including slippage).
        filled_qty: Number of units/lots executed.
        timestamp: Time of execution.
        slippage: Absolute slippage applied to execution price.
        brokerage: Commission / fee charged for this execution.
    """

    order_id: str
    filled_price: float
    filled_qty: int
    timestamp: datetime
    slippage: float = 0.0
    brokerage: float = 0.0

    def __post_init__(self) -> None:
        """Validate fill attributes on construction."""
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string.")

        if not isinstance(self.filled_price, (int, float)) or self.filled_price <= 0.0:
            raise ValueError(f"filled_price must be positive, got {self.filled_price}")
        self.filled_price = float(self.filled_price)

        if not isinstance(self.filled_qty, int) or self.filled_qty <= 0:
            raise ValueError(f"filled_qty must be a positive integer, got {self.filled_qty}")

        if not isinstance(self.timestamp, datetime):
            raise ValueError(f"timestamp must be a datetime, got {type(self.timestamp).__name__}")

        if not isinstance(self.slippage, (int, float)):
            raise ValueError(f"slippage must be numeric, got {self.slippage}")
        self.slippage = float(self.slippage)

        if not isinstance(self.brokerage, (int, float)) or self.brokerage < 0.0:
            raise ValueError(f"brokerage must be non-negative, got {self.brokerage}")
        self.brokerage = float(self.brokerage)
