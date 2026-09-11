"""
Observability and reporting package for the Vega Quant Trading Engine.

Exports:
    - TradeBlotter: Trade execution record collection and metrics.
    - TradeBlotterEntry: Standardized trade execution record.
    - StructuredLogger: Machine-readable JSON event logger and alert dispatcher.
    - LogEvent: Structured event record.
    - LogEventType: Standardized domain event types.
"""

from vega.reporting.blotter import TradeBlotter, TradeBlotterEntry
from vega.reporting.logger import LogEvent, LogEventType, StructuredLogger

__all__ = [
    "TradeBlotter",
    "TradeBlotterEntry",
    "StructuredLogger",
    "LogEvent",
    "LogEventType",
]
