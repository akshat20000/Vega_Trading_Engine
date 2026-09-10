"""
Shared test fixtures for the Vega Quant Trading Engine test suite.

All fixtures here use synthetic, hand-crafted data — no CSV files are read.
This keeps tests:
  - Fast (no I/O)
  - Portable (no dependency on file paths)
  - Self-documenting (expected values are visible in the test itself)

Any test file in this directory can use these fixtures by declaring them
as function parameters — pytest injects them automatically.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from vega.config import VegaConfig
from vega.data.models import Bar, Tick


# ── Helper functions (not fixtures — call directly in tests) ──────────────────

def make_bar(
    timestamp: datetime,
    open_: float = 100.0,
    high: float  = 110.0,
    low: float   = 90.0,
    close: float = 105.0,
    volume: float = 1_000.0,
) -> Bar:
    """
    Convenience factory for creating a single Bar in tests.

    Named 'open_' (with underscore) because 'open' is a Python built-in.

    Example:
        bar = make_bar(datetime(2023, 1, 2), close=17_859.0)
    """
    return Bar(
        timestamp=timestamp,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
    )


def make_tick(timestamp: datetime, price: float = 100.0, volume: float = 100.0) -> Tick:
    """Convenience factory for creating a single Tick in tests."""
    return Tick(timestamp=timestamp, price=price, volume=volume)


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> VegaConfig:
    """
    A VegaConfig populated with all default values.
    Safe to use in any test that needs a config but doesn't care about
    specific parameter values.
    """
    return VegaConfig()


@pytest.fixture
def sample_bars() -> list[Bar]:
    """
    20 synthetic daily bars starting 2023-01-02.

    Prices form a gentle uptrend (price increases by ~30 each bar).
    OHLCV values are always internally consistent (high >= open/close,
    low <= open/close).

    Use this fixture when you need a realistic-looking list[Bar] but
    the exact prices don't matter for the test.
    """
    base  = datetime(2023, 1, 2)
    price = 17_500.0
    bars  = []
    for i in range(20):
        bars.append(
            Bar(
                timestamp = base + timedelta(days=i),
                open      = price,
                high      = price + 100.0,
                low       = price - 80.0,
                close     = price + 30.0,
                volume    = 300_000.0,
            )
        )
        price += 30.0
    return bars


@pytest.fixture
def sample_ticks() -> list[Tick]:
    """
    10 synthetic ticks across a single trading session (2023-01-02 09:15–09:24).

    Each tick is 1 minute apart, starting at price 17_500 and rising by 5
    per tick. Volume is fixed at 100 per tick.

    Use this fixture when you need a basic list[Tick] for loading/aggregation
    tests where exact OHLCV verification is not required.
    """
    base  = datetime(2023, 1, 2, 9, 15, 0)
    ticks = [
        Tick(
            timestamp = base + timedelta(minutes=i),
            price     = 17_500.0 + i * 5.0,
            volume    = 100.0,
        )
        for i in range(10)
    ]
    return ticks
