"""
Abstract base broker interface for the Vega Quant Trading Engine.

Defines the contract for order placement, cancellation, and order inspection.
External callers (strategies, API layers) interact only with this public API,
never accessing private broker state (e.g. `_orders`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from vega.orders.models import Fill, Order


class AbstractBroker(ABC):
    """
    Abstract broker interface.

    Defines the public contract for all broker implementations (PaperBroker, KiteBroker).
    Execution environments determine fills; callers submit orders without specifying fill prices.
    """

    @abstractmethod
    def place_order(self, order: Order) -> Order:
        """
        Submit an order to the broker.

        Enforces idempotency based on `order.client_order_id`. If an order with
        the same `client_order_id` has already been placed, returns the existing order
        without duplicating state.

        Args:
            order: The Order instance to place.

        Returns:
            The placed or existing Order instance.
        """
        pass

    @abstractmethod
    def cancel_order(self, client_order_id: str) -> Order:
        """
        Request cancellation of an existing pending order.

        Args:
            client_order_id: The client-assigned idempotency identifier.

        Returns:
            The cancelled Order instance.

        Raises:
            KeyError: If no order with client_order_id exists.
            InvalidOrderStateTransitionError: If the order cannot be cancelled (e.g. already FILLED).
        """
        pass

    @abstractmethod
    def get_order(self, client_order_id: str) -> Order | None:
        """
        Look up an order by its client_order_id.

        Args:
            client_order_id: The client-assigned idempotency identifier.

        Returns:
            The Order if found, else None.
        """
        pass

    @abstractmethod
    def get_all_orders(self) -> list[Order]:
        """
        Return a list of all orders tracked by the broker.

        Returns:
            A list of Order objects. Does not expose private internal state collections.
        """
        pass

    @abstractmethod
    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        """
        Return execution fills recorded by the broker.

        Args:
            order_id: Optional client_order_id to filter fills for a specific order.
                      If None, returns all fills.

        Returns:
            A list of Fill objects.
        """
        pass
