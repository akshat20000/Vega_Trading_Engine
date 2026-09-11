"""
Unit and regression tests for StopAndReverseStrategy.

Covers all 10 test requirements:
1. Strategy initialization
2. No signal without crossover
3. Bullish crossover from flat -> BUY 1
4. Bearish crossover from flat -> SELL 1
5. LONG + bearish crossover -> reversal (SELL 2)
6. SHORT + bullish crossover -> reversal (BUY 2)
7. Deterministic client_order_id
8. No duplicate signal on same crossover
9. Kill switch blocks entries and flattens
10. Insufficient history handled safely
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from vega.data.models import Bar
from vega.macro.models import Regime
from vega.orders.models import Fill, OrderSide, OrderStatus
from vega.strategy.stop_and_reverse import SARState, StopAndReverseStrategy


def make_bar(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_price: float | None = None,
    dt: datetime | None = None,
) -> Bar:
    """Helper to construct a valid Bar object."""
    c = float(close)
    o = float(open_price if open_price is not None else c)
    h = float(high if high is not None else max(c, o))
    l = float(low if low is not None else min(c, o))
    timestamp = dt or datetime(2026, 9, 11, 9, 15, 0, tzinfo=timezone.utc)
    return Bar(timestamp=timestamp, open=o, high=h, low=l, close=c, volume=1000.0)


def build_warmup_bars(count: int = 22, price: float = 100.0) -> list[Bar]:
    """Generate a series of stable bars for EMA warmup."""
    base_dt = datetime(2026, 1, 1, 9, 15, 0, tzinfo=timezone.utc)
    return [
        make_bar(close=price, dt=base_dt + timedelta(days=i))
        for i in range(count)
    ]


# ── Test 1: Strategy Initialization ──────────────────────────────────────────


def test_sar_initialization() -> None:
    """Test 1: Verify SAR initialization with default and explicit parameters."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21, quantity=1)
    assert sar.symbol == "NIFTY50"
    assert sar.fast_period == 9
    assert sar.slow_period == 21
    assert sar.quantity == 1
    assert sar.state == SARState.FLAT
    assert sar.position == 0
    assert sar.kill_switch is False


# ── Test 2: No Signal Without Crossover ──────────────────────────────────────


def test_sar_no_signal_without_crossover() -> None:
    """Test 2: Constant price or persistent trend produces no new crossover signals."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)

    # 30 bars of constant price 100.0 (fast == slow, no crossover)
    bars = build_warmup_bars(count=30, price=100.0)
    for bar in bars:
        orders = sar.on_bar(bar)
        assert orders == []


# ── Test 3: Bullish Crossover From Flat -> BUY ────────────────────────────────


def test_sar_bullish_crossover_from_flat_buy() -> None:
    """Test 3: Bullish crossover from FLAT state generates BUY 1 order."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21, quantity=1)

    # 22 bars of flat 100.0
    for bar in build_warmup_bars(count=22, price=100.0):
        sar.on_bar(bar)

    # Bar 23: Sharp rally to 120.0 -> Fast EMA crosses above Slow EMA
    crossover_bar = make_bar(
        close=120.0,
        dt=datetime(2026, 2, 1, 9, 15, 0, tzinfo=timezone.utc),
    )
    orders = sar.on_bar(crossover_bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.BUY
    assert order.quantity == 1
    assert order.price == 120.0
    assert order.client_order_id == "SAR-NIFTY50-1-BUY"

    # on_bar does NOT mutate position state
    assert sar.state == SARState.FLAT
    assert sar.position == 0

    # on_fill updates position state
    fill = Fill(
        order_id=order.client_order_id,
        filled_price=120.0,
        filled_qty=1,
        timestamp=crossover_bar.timestamp,
        side=OrderSide.BUY,
    )
    sar.on_fill(fill)
    assert sar.state == SARState.LONG
    assert sar.position == 1


# ── Test 4: Bearish Crossover From Flat -> SELL ───────────────────────────────


def test_sar_bearish_crossover_from_flat_sell() -> None:
    """Test 4: Bearish crossover from FLAT state generates SELL 1 order."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21, quantity=1)

    for bar in build_warmup_bars(count=22, price=100.0):
        sar.on_bar(bar)

    # Bar 23: Sharp drop to 80.0 -> Fast EMA crosses below Slow EMA
    crossover_bar = make_bar(
        close=80.0,
        dt=datetime(2026, 2, 1, 9, 15, 0, tzinfo=timezone.utc),
    )
    orders = sar.on_bar(crossover_bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.SELL
    assert order.quantity == 1
    assert order.price == 80.0
    assert order.client_order_id == "SAR-NIFTY50-1-SELL"

    fill = Fill(
        order_id=order.client_order_id,
        filled_price=80.0,
        filled_qty=1,
        timestamp=crossover_bar.timestamp,
        side=OrderSide.SELL,
    )
    sar.on_fill(fill)
    assert sar.state == SARState.SHORT
    assert sar.position == -1


# ── Test 5: LONG + Bearish Crossover -> Reversal (SELL 2) ─────────────────────


def test_sar_long_bearish_crossover_reversal() -> None:
    """Test 5: While in LONG state, bearish crossover emits reversal SELL 2 order."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21, quantity=1)
    sar.set_state(SARState.LONG, position=1)

    for bar in build_warmup_bars(count=22, price=100.0):
        sar.on_bar(bar)

    # Bearish drop
    crossover_bar = make_bar(
        close=80.0,
        dt=datetime(2026, 2, 1, 9, 15, 0, tzinfo=timezone.utc),
    )
    orders = sar.on_bar(crossover_bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.SELL
    # Fixed target position model: LONG (+1) -> SHORT (-1) requires SELL 2
    assert order.quantity == 2
    assert order.client_order_id == "SAR-NIFTY50-1-SELL"

    # Fill arrives: position changes from +1 by -2 -> -1
    fill = Fill(
        order_id=order.client_order_id,
        filled_price=80.0,
        filled_qty=2,
        timestamp=crossover_bar.timestamp,
        side=OrderSide.SELL,
    )
    sar.on_fill(fill)
    assert sar.state == SARState.SHORT
    assert sar.position == -1


# ── Test 6: SHORT + Bullish Crossover -> Reversal (BUY 2) ─────────────────────


def test_sar_short_bullish_crossover_reversal() -> None:
    """Test 6: While in SHORT state, bullish crossover emits reversal BUY 2 order."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21, quantity=1)
    sar.set_state(SARState.SHORT, position=-1)

    for bar in build_warmup_bars(count=22, price=100.0):
        sar.on_bar(bar)

    # Bullish rally
    crossover_bar = make_bar(
        close=120.0,
        dt=datetime(2026, 2, 1, 9, 15, 0, tzinfo=timezone.utc),
    )
    orders = sar.on_bar(crossover_bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.BUY
    # Fixed target position model: SHORT (-1) -> LONG (+1) requires BUY 2
    assert order.quantity == 2
    assert order.client_order_id == "SAR-NIFTY50-1-BUY"

    # Fill arrives: position changes from -1 by +2 -> +1
    fill = Fill(
        order_id=order.client_order_id,
        filled_price=120.0,
        filled_qty=2,
        timestamp=crossover_bar.timestamp,
        side=OrderSide.BUY,
    )
    sar.on_fill(fill)
    assert sar.state == SARState.LONG
    assert sar.position == 1


# ── Test 7: Deterministic Client Order ID ─────────────────────────────────────


def test_sar_deterministic_client_order_id() -> None:
    """Test 7: Client order IDs are deterministic and follow SAR-{symbol}-{count}-{side}."""
    sar1 = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)
    sar2 = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)

    for bar in build_warmup_bars(count=22, price=100.0):
        sar1.on_bar(bar)
        sar2.on_bar(bar)

    cross_bar = make_bar(close=120.0)
    o1 = sar1.on_bar(cross_bar)
    o2 = sar2.on_bar(cross_bar)

    assert o1[0].client_order_id == "SAR-NIFTY50-1-BUY"
    assert o1[0].client_order_id == o2[0].client_order_id


# ── Test 8: No Duplicate Signal on Same Crossover ─────────────────────────────


def test_sar_no_duplicate_signal_on_same_crossover() -> None:
    """Test 8: Consecutive bars while trend persists generate no duplicate orders."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)

    for bar in build_warmup_bars(count=22, price=100.0):
        sar.on_bar(bar)

    # Initial bullish crossover
    bar23 = make_bar(close=120.0, dt=datetime(2026, 2, 1, tzinfo=timezone.utc))
    orders23 = sar.on_bar(bar23)
    assert len(orders23) == 1

    # Next bar continues upward (fast still > slow, no new crossover)
    bar24 = make_bar(close=125.0, dt=datetime(2026, 2, 2, tzinfo=timezone.utc))
    orders24 = sar.on_bar(bar24)
    assert len(orders24) == 0

    bar25 = make_bar(close=128.0, dt=datetime(2026, 2, 3, tzinfo=timezone.utc))
    orders25 = sar.on_bar(bar25)
    assert len(orders25) == 0


# ── Test 9: Kill Switch ───────────────────────────────────────────────────────


def test_sar_kill_switch() -> None:
    """Test 9: Kill switch blocks crossover signals and emits flatten order when positioned."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)
    sar.set_state(SARState.LONG, position=1)
    sar.enable_kill_switch()

    # Positioned LONG -> produces flatten SELL 1
    bar = make_bar(close=100.0)
    orders = sar.on_bar(bar)
    assert len(orders) == 1
    assert orders[0].side == OrderSide.SELL
    assert orders[0].quantity == 1
    assert orders[0].client_order_id == "SAR-NIFTY50-FLATTEN"

    # When flat, produces no orders even under crossover
    sar_flat = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)
    sar_flat.enable_kill_switch()
    for b in build_warmup_bars(count=25, price=100.0):
        sar_flat.on_bar(b)
    crossover_bar = make_bar(close=130.0)
    assert sar_flat.on_bar(crossover_bar) == []


# ── Test 10: Insufficient History Handled Safely ──────────────────────────────


def test_sar_insufficient_history_handled_safely() -> None:
    """Test 10: Insufficient history during warmup returns empty order list without error."""
    sar = StopAndReverseStrategy("NIFTY50", fast_period=9, slow_period=21)

    # 10 bars is less than slow_period (21) + 1 = 22 bars
    for bar in build_warmup_bars(count=10, price=100.0):
        orders = sar.on_bar(bar)
        assert orders == []
