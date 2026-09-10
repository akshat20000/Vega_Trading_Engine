"""
Core data models for the Vega Quant Trading Engine.

Two models live here:
  Bar  — one OHLCV candle (daily, 1-minute, etc.)
  Tick — one raw trade event from tick data

Using named dataclasses instead of plain lists prevents mistakes like
accidentally swapping 'high' and 'low', or passing raw floats to an indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Bar:
    """
    One OHLCV bar covering a fixed time period (e.g. one trading day).

    Attributes:
        timestamp: The opening time of this bar's period.
                   For daily bars, this is midnight of that date.
                   For intraday bars, this is the bucket start time.
        open:      Price at the start of the period.
        high:      Highest price during the period.
        low:       Lowest price during the period.
        close:     Price at the end of the period.
        volume:    Total units traded during the period.

    Example:
        bar = Bar(
            timestamp=datetime(2023, 1, 2),
            open=17807.35,
            high=17866.35,
            low=17684.15,
            close=17859.45,
            volume=258_497,
        )
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """
        Validate that bar values are internally consistent.
        Called automatically by Python after __init__.
        """
        if self.high < self.low:
            raise ValueError(
                f"Bar at {self.timestamp}: high ({self.high}) cannot be "
                f"less than low ({self.low})."
            )
        if self.high < self.open:
            raise ValueError(
                f"Bar at {self.timestamp}: high ({self.high}) must be >= "
                f"open ({self.open})."
            )
        if self.high < self.close:
            raise ValueError(
                f"Bar at {self.timestamp}: high ({self.high}) must be >= "
                f"close ({self.close})."
            )
        if self.low > self.open:
            raise ValueError(
                f"Bar at {self.timestamp}: low ({self.low}) must be <= "
                f"open ({self.open})."
            )
        if self.low > self.close:
            raise ValueError(
                f"Bar at {self.timestamp}: low ({self.low}) must be <= "
                f"close ({self.close})."
            )
        if self.volume < 0:
            raise ValueError(
                f"Bar at {self.timestamp}: volume ({self.volume}) cannot be negative."
            )


@dataclass
class Tick:
    """
    One raw trade event — the most granular form of market data.

    Tick data shows every individual transaction that occurred on an exchange.
    We aggregate ticks into Bars using tick_aggregator.py.

    Attributes:
        timestamp: The exact date and time of this trade.
        price:     The price at which this trade occurred.
        volume:    The number of units traded in this single event.

    Example:
        tick = Tick(
            timestamp=datetime(2023, 1, 2, 9, 15, 0),
            price=17807.00,
            volume=200,
        )
    """

    timestamp: datetime
    price: float
    volume: float

    def __post_init__(self) -> None:
        """Validate tick values."""
        if self.price <= 0:
            raise ValueError(
                f"Tick at {self.timestamp}: price ({self.price}) must be positive."
            )
        if self.volume < 0:
            raise ValueError(
                f"Tick at {self.timestamp}: volume ({self.volume}) cannot be negative."
            )
