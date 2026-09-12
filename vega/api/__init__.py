"""
REST API package for the Vega Quant Trading Engine.

Exports:
    - create_app: Application factory constructing the FastAPI server.
    - Response models: HealthResponse, StatusResponse, PortfolioResponse,
      PositionResponse, PositionsListResponse, TradeResponse, TradesListResponse,
      MetricsResponse.
"""

from vega.api.app import (
    HealthResponse,
    MetricsResponse,
    PortfolioResponse,
    PositionResponse,
    PositionsListResponse,
    StatusResponse,
    TradeResponse,
    TradesListResponse,
    create_app,
)

__all__ = [
    "create_app",
    "HealthResponse",
    "StatusResponse",
    "PortfolioResponse",
    "PositionResponse",
    "PositionsListResponse",
    "TradeResponse",
    "TradesListResponse",
    "MetricsResponse",
]
