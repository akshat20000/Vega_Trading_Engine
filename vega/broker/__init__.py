"""
Broker package for the Vega Quant Trading Engine.

Exports:
    - AbstractBroker: Base broker interface contract.
    - PaperBroker: Deterministic paper trading simulation broker.
    - BrokerError: Exception raised during broker execution/system errors.
    - KiteBroker: Safe placeholder skeleton for future Zerodha Kite connectivity.
"""

from vega.broker.base import AbstractBroker
from vega.broker.kite import KiteBroker
from vega.broker.paper import BrokerError, IdempotencyConflictError, PaperBroker

__all__ = [
    "AbstractBroker",
    "PaperBroker",
    "BrokerError",
    "IdempotencyConflictError",
    "KiteBroker",
]
