"""
Event models for the real-time streaming pipeline.

Defines:
    - TickEvent: Metadata wrapper around a raw Tick with symbol and receipt time.
    - BarEvent: Emitted when a real-time bar aggregation bucket completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from vega.data.models import Bar, Tick


@dataclass(frozen=True)
class TickEvent:
    """
    Market tick event received from simulated or real-time streaming feed.

    Attributes:
        symbol: Ticker symbol (e.g. 'NIFTY50').
        tick: The underlying Tick object (timestamp, price, volume).
        received_at: Local engine arrival timestamp.
    """

    symbol: str
    tick: Tick
    received_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True)
class BarEvent:
    """
    Completed OHLCV bar event emitted by the real-time aggregator.

    Attributes:
        symbol: Ticker symbol.
        bar: Aggregated OHLCV Bar.
        emitted_at: Aggregation completion timestamp.
    """

    symbol: str
    bar: Bar
    emitted_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
