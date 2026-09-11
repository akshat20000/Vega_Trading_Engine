"""
Orders package for the Vega Quant Trading Engine.

Exports:
    - OrderSide: BUY or SELL.
    - OrderStatus: PENDING, FILLED, CANCELLED, REJECTED.
    - InvalidOrderStateTransitionError: Error raised on illegal status transitions.
    - Order: Order representation with state machine validation.
    - Fill: Execution fill representation.
"""

from vega.orders.models import (
    Fill,
    InvalidOrderStateTransitionError,
    Order,
    OrderSide,
    OrderStatus,
)

__all__ = [
    "OrderSide",
    "OrderStatus",
    "InvalidOrderStateTransitionError",
    "Order",
    "Fill",
]
