"""
Safe skeleton for Zerodha Kite broker integration in the Vega Quant Trading Engine.

This module provides a stub conforming to the AbstractBroker interface for future
live execution phases. It strictly contains NO credentials, makes NO network calls,
and connects to NO real accounts.

All execution and placement methods raise NotImplementedError.
"""

from __future__ import annotations

from vega.broker.base import AbstractBroker
from vega.orders.models import Fill, Order


class KiteBroker(AbstractBroker):
    """
    KiteBroker skeleton conforming to AbstractBroker.

    Live connectivity to Zerodha Kite Connect will be implemented in future phases.
    All operations are currently disabled for safety.
    """

    def __init__(self, api_key: str | None = None, access_token: str | None = None) -> None:
        """Initialize safe KiteBroker skeleton without credentials."""
        self._api_key = api_key
        self._access_token = access_token

    def place_order(self, order: Order) -> Order:
        """Raise NotImplementedError: Live Zerodha order placement not yet implemented."""
        raise NotImplementedError(
            "KiteBroker.place_order is disabled in Phase 5A. "
            "Real broker connectivity is scheduled for a future phase."
        )

    def cancel_order(self, client_order_id: str) -> Order:
        """Raise NotImplementedError: Live Zerodha order cancellation not yet implemented."""
        raise NotImplementedError(
            "KiteBroker.cancel_order is disabled in Phase 5A. "
            "Real broker connectivity is scheduled for a future phase."
        )

    def get_order(self, client_order_id: str) -> Order | None:
        """Raise NotImplementedError: Live Zerodha order lookup not yet implemented."""
        raise NotImplementedError(
            "KiteBroker.get_order is disabled in Phase 5A. "
            "Real broker connectivity is scheduled for a future phase."
        )

    def get_all_orders(self) -> list[Order]:
        """Raise NotImplementedError: Live Zerodha order queries not yet implemented."""
        raise NotImplementedError(
            "KiteBroker.get_all_orders is disabled in Phase 5A. "
            "Real broker connectivity is scheduled for a future phase."
        )

    def get_fills(self, order_id: str | None = None) -> list[Fill]:
        """Raise NotImplementedError: Live Zerodha fill queries not yet implemented."""
        raise NotImplementedError(
            "KiteBroker.get_fills is disabled in Phase 5A. "
            "Real broker connectivity is scheduled for a future phase."
        )
