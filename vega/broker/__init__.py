"""
Broker package for the Vega Quant Trading Engine.

Exports:
    - AbstractBroker: Base broker interface contract.
    - PaperBroker: Deterministic paper trading simulation broker.
    - BrokerError: Exception raised during broker execution/system errors.
    - KiteBroker: Safe placeholder skeleton for future Zerodha Kite connectivity.
    - RetryPolicy: Exponential backoff policy for resilient API calls.
    - retry_with_backoff: Resilient retry wrapper and decorator.
    - RateLimiter: Sliding window rate limiter with delay-on-exceed policy.
    - TokenProvider, InMemoryTokenProvider: Token management abstractions.
    - APIClient, APIResponse: Resilient broker API client.
    - AuthenticationError, TokenRefreshError, APIResponseError: Auth/API errors.
"""

from vega.broker.auth import (
    APIClient,
    APIResponse,
    APIResponseError,
    AuthenticationError,
    InMemoryTokenProvider,
    TokenProvider,
    TokenRefreshError,
)
from vega.broker.base import AbstractBroker
from vega.broker.kite import KiteBroker
from vega.broker.paper import BrokerError, IdempotencyConflictError, PaperBroker
from vega.broker.rate_limiter import RateLimiter
from vega.broker.retry import RetryPolicy, retry_with_backoff

__all__ = [
    "AbstractBroker",
    "PaperBroker",
    "BrokerError",
    "IdempotencyConflictError",
    "KiteBroker",
    "RetryPolicy",
    "retry_with_backoff",
    "RateLimiter",
    "TokenProvider",
    "InMemoryTokenProvider",
    "APIClient",
    "APIResponse",
    "AuthenticationError",
    "TokenRefreshError",
    "APIResponseError",
]
