"""
Unit and regression tests for ATRGridStrategy.

Covers all 19 test requirements:
1. Strategy initialization
2. Grid creation
3. Correct ATR x multiplier spacing
4. Correct levels above/below anchor
5. First lower level produces BUY
6. First upper level produces SELL
7. Multiple crossed levels are deterministic (lower BUY descending, upper SELL ascending)
8. Duplicate level does not produce duplicate entry
9. Pyramiding across multiple levels
10. Position cap respected
11. Re-anchor only when flat
12. Re-anchor creates fresh grid
13. LONG pyramid exit (entry_price + spacing)
14. SHORT pyramid exit (entry_price - spacing)
15. Exit is not immediately reused as entry on same bar
16. Kill switch blocks entries
17. Kill switch produces flatten signal when positioned
18. Bearish macro blocks new ATR Grid long entries
19. Deterministic client_order_id
"""

from __future__ import annotations

from datetime import datetime, timezone
import pytest

from vega.data.models import Bar
from vega.macro.models import Regime
from vega.orders.models import Fill, OrderSide, OrderStatus
from vega.strategy.atr_grid import ATRGridStrategy, GridLevel, GridPosition


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


# ── Test 1: Strategy Initialization ──────────────────────────────────────────


def test_atr_grid_initialization() -> None:
    """Test 1: Verify strategy initializes with correct explicit parameters."""
    strategy = ATRGridStrategy(
        symbol="NIFTY50",
        atr_period=14,
        grid_multiplier=1.5,
        levels=3,
        position_cap=5,
        quantity_per_level=1,
    )
    assert strategy.symbol == "NIFTY50"
    assert strategy.atr_period == 14
    assert strategy.grid_multiplier == 1.5
    assert strategy.levels == 3
    assert strategy.position_cap == 5
    assert strategy.quantity_per_level == 1
    assert strategy.net_position == 0
    assert strategy.kill_switch is False
    assert strategy.anchor is None
    assert strategy.spacing is None
    assert strategy.epoch == 0


# ── Test 2: Grid Creation ────────────────────────────────────────────────────


def test_grid_creation() -> None:
    """Test 2: Verify grid creation establishes anchor, spacing, and levels."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    assert strategy.anchor == 22000.0
    assert strategy.spacing == 300.0
    assert strategy.epoch == 1
    assert len(strategy.buy_levels) == 3
    assert len(strategy.sell_levels) == 3


# ── Test 3: Correct ATR x Multiplier Spacing ──────────────────────────────────


def test_correct_atr_multiplier_spacing() -> None:
    """Test 3: Verify grid spacing is exactly ATR x grid_multiplier."""
    strategy = ATRGridStrategy("NIFTY50", atr_period=14, grid_multiplier=1.5)
    # With ATR = 200, spacing should be 200 * 1.5 = 300.0
    strategy.establish_grid(anchor=22000.0, atr_value=200.0)
    assert strategy.spacing == 300.0

    # With multiplier 2.0 and ATR = 150 -> 300.0
    strat2 = ATRGridStrategy("NIFTY50", grid_multiplier=2.0)
    strat2.establish_grid(anchor=22000.0, atr_value=150.0)
    assert strat2.spacing == 300.0


# ── Test 4: Correct Levels Above/Below Anchor ────────────────────────────────


def test_correct_levels_above_and_below_anchor() -> None:
    """Test 4: Verify buy levels below and sell levels above anchor match specification."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Buy levels below anchor: 21700, 21400, 21100
    buy_prices = [lvl.price for lvl in strategy.buy_levels]
    assert buy_prices == [21700.0, 21400.0, 21100.0]
    for i, lvl in enumerate(strategy.buy_levels, start=1):
        assert lvl.level_id == i
        assert lvl.side == OrderSide.BUY

    # Sell levels above anchor: 22300, 22600, 22900
    sell_prices = [lvl.price for lvl in strategy.sell_levels]
    assert sell_prices == [22300.0, 22600.0, 22900.0]
    for i, lvl in enumerate(strategy.sell_levels, start=1):
        assert lvl.level_id == i
        assert lvl.side == OrderSide.SELL


# ── Test 5: First Lower Level Produces BUY ───────────────────────────────────


def test_first_lower_level_produces_buy() -> None:
    """Test 5: Price crossing first lower level produces a BUY order."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar drops through 21700 (low=21650)
    bar = make_bar(close=21680.0, high=22000.0, low=21650.0)
    orders = strategy.on_bar(bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.BUY
    assert order.price == 21700.0
    assert order.quantity == 1
    assert order.client_order_id == "GRID-NIFTY50-E1-BUY-L1"
    # on_bar does NOT mutate filled position state
    assert strategy.net_position == 0


# ── Test 6: First Upper Level Produces SELL ──────────────────────────────────


def test_first_upper_level_produces_sell() -> None:
    """Test 6: Price crossing first upper level produces a SELL order."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar rises through 22300 (high=22350)
    bar = make_bar(close=22320.0, high=22350.0, low=22000.0)
    orders = strategy.on_bar(bar)

    assert len(orders) == 1
    order = orders[0]
    assert order.side == OrderSide.SELL
    assert order.price == 22300.0
    assert order.quantity == 1
    assert order.client_order_id == "GRID-NIFTY50-E1-SELL-S1"
    assert strategy.net_position == 0


# ── Test 7: Multiple Crossed Levels are Deterministic ─────────────────────────


def test_multiple_crossed_levels_are_deterministic() -> None:
    """
    Test 7: Multiple crossed levels generate orders in deterministic convention:
    Lower BUY levels first (nearest-to-anchor to furthest: L1, L2...),
    then upper SELL levels (nearest-to-anchor to furthest: S1, S2...).
    """
    strategy = ATRGridStrategy("NIFTY50", levels=3, position_cap=5)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar drops deeply: crosses Level 1 (21700) and Level 2 (21400)
    bar = make_bar(close=21350.0, high=21900.0, low=21350.0)
    orders = strategy.on_bar(bar)

    assert len(orders) == 2
    assert orders[0].client_order_id == "GRID-NIFTY50-E1-BUY-L1"
    assert orders[0].price == 21700.0
    assert orders[1].client_order_id == "GRID-NIFTY50-E1-BUY-L2"
    assert orders[1].price == 21400.0

    # Wide bar crossing both lower BUY and upper SELL levels in a single bar
    strat_wide = ATRGridStrategy("NIFTY50", levels=3, position_cap=5)
    strat_wide.establish_grid(anchor=22000.0, spacing=300.0)
    wide_bar = make_bar(close=22000.0, high=22400.0, low=21600.0)
    wide_orders = strat_wide.on_bar(wide_bar)

    # Order must be: BUY L1 first, then SELL S1
    assert len(wide_orders) == 2
    assert wide_orders[0].side == OrderSide.BUY
    assert wide_orders[0].price == 21700.0
    assert wide_orders[1].side == OrderSide.SELL
    assert wide_orders[1].price == 22300.0


# ── Test 8: Duplicate Level Does Not Produce Duplicate Entry ─────────────────


def test_duplicate_level_does_not_produce_duplicate_entry() -> None:
    """Test 8: An already-triggered level does not create duplicate orders in same epoch."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # First bar crosses Level 1 (21700)
    bar1 = make_bar(close=21680.0, high=21900.0, low=21650.0)
    orders1 = strategy.on_bar(bar1)
    assert len(orders1) == 1
    assert orders1[0].client_order_id == "GRID-NIFTY50-E1-BUY-L1"

    # Second bar also remains below/crosses Level 1
    bar2 = make_bar(close=21660.0, high=21690.0, low=21640.0)
    orders2 = strategy.on_bar(bar2)
    # Must NOT produce duplicate order for L1
    assert len(orders2) == 0


# ── Test 9: Pyramiding Across Multiple Levels ────────────────────────────────


def test_pyramiding_across_multiple_levels() -> None:
    """Test 9: Strategy allows pyramiding entries as additional levels are crossed."""
    strategy = ATRGridStrategy("NIFTY50", levels=3, position_cap=5)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar 1 crosses Level 1
    bar1 = make_bar(close=21650.0, high=21900.0, low=21650.0)
    orders1 = strategy.on_bar(bar1)
    assert len(orders1) == 1
    # Fill arrives
    fill1 = Fill(
        order_id=orders1[0].client_order_id,
        filled_price=21700.0,
        filled_qty=1,
        timestamp=bar1.timestamp,
        side=OrderSide.BUY,
    )
    strategy.on_fill(fill1)
    assert strategy.net_position == 1

    # Bar 2 crosses Level 2 (21400)
    bar2 = make_bar(close=21350.0, high=21600.0, low=21350.0)
    orders2 = strategy.on_bar(bar2)
    assert len(orders2) == 1
    assert orders2[0].client_order_id == "GRID-NIFTY50-E1-BUY-L2"
    fill2 = Fill(
        order_id=orders2[0].client_order_id,
        filled_price=21400.0,
        filled_qty=1,
        timestamp=bar2.timestamp,
        side=OrderSide.BUY,
    )
    strategy.on_fill(fill2)
    assert strategy.net_position == 2
    assert len(strategy.active_entries) == 2


# ── Test 10: Position Cap Respected ──────────────────────────────────────────


def test_position_cap_respected() -> None:
    """Test 10: Strategy never exceeds configured strategy position cap."""
    # Cap position at 2
    strategy = ATRGridStrategy("NIFTY50", levels=4, position_cap=2)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar drops sharply crossing L1 (21700), L2 (21400), L3 (21100), L4 (20800)
    bar = make_bar(close=20700.0, high=22000.0, low=20700.0)
    orders = strategy.on_bar(bar)

    # Only 2 orders allowed (L1 and L2), L3 and L4 blocked by cap
    assert len(orders) == 2
    assert orders[0].client_order_id == "GRID-NIFTY50-E1-BUY-L1"
    assert orders[1].client_order_id == "GRID-NIFTY50-E1-BUY-L2"


# ── Test 11: Re-Anchor Only When Flat ────────────────────────────────────────


def test_reanchor_only_when_flat() -> None:
    """Test 11: Grid may re-anchor ONLY when net position is flat."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # When flat, can_reanchor is True
    assert strategy.can_reanchor() is True

    # Register an entry -> position becomes 1
    strategy.register_entry(level_id=1, side=OrderSide.BUY, price=21700.0, quantity=1)
    assert strategy.net_position == 1
    assert strategy.can_reanchor() is False

    new_bar = make_bar(close=22100.0)
    with pytest.raises(ValueError, match="Cannot re-anchor grid: net position is not flat"):
        strategy.reanchor(new_bar)

    # Flatten position
    fill_exit = Fill(
        order_id="EXIT-ORDER",
        filled_price=22000.0,
        filled_qty=1,
        timestamp=new_bar.timestamp,
        side=OrderSide.SELL,
    )
    strategy.on_fill(fill_exit)
    assert strategy.net_position == 0
    assert strategy.can_reanchor() is True

    # Now reanchor succeeds
    strategy.reanchor(new_bar, spacing=300.0)
    assert strategy.anchor == 22100.0


# ── Test 12: Re-Anchor Creates Fresh Grid ────────────────────────────────────


def test_reanchor_creates_fresh_grid() -> None:
    """Test 12: Re-anchoring establishes new anchor, new spacing, clears triggered levels."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Trigger L1
    bar1 = make_bar(close=21650.0, low=21650.0)
    strategy.on_bar(bar1)
    assert (OrderSide.BUY, 1) in strategy.triggered_levels
    assert strategy.epoch == 1

    # Reanchor to 22500 with spacing 250
    reanchor_bar = make_bar(close=22500.0)
    strategy.reanchor(reanchor_bar, spacing=250.0)

    assert strategy.anchor == 22500.0
    assert strategy.spacing == 250.0
    assert strategy.epoch == 2
    assert len(strategy.triggered_levels) == 0

    buy_prices = [lvl.price for lvl in strategy.buy_levels]
    assert buy_prices == [22250.0, 22000.0, 21750.0]


# ── Test 13: LONG Pyramid Exit ───────────────────────────────────────────────


def test_long_pyramid_exit() -> None:
    """Test 13: LONG entry exits exactly one spacing above its entry price."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Trigger and fill Level 1 (21700)
    entry_bar = make_bar(close=21680.0, low=21650.0)
    entry_orders = strategy.on_bar(entry_bar)
    fill = Fill(
        order_id=entry_orders[0].client_order_id,
        filled_price=21700.0,
        filled_qty=1,
        timestamp=entry_bar.timestamp,
        side=OrderSide.BUY,
    )
    strategy.on_fill(fill)
    assert strategy.net_position == 1

    # Exit price for LONG is entry_price + spacing = 21700 + 300 = 22000
    exit_bar = make_bar(close=22010.0, high=22020.0, low=21750.0)
    exit_orders = strategy.on_bar(exit_bar)

    assert len(exit_orders) == 1
    exit_order = exit_orders[0]
    assert exit_order.side == OrderSide.SELL
    assert exit_order.price == 22000.0
    assert exit_order.quantity == 1
    assert exit_order.client_order_id == "GRID-NIFTY50-E1-EXIT-L1"


# ── Test 14: SHORT Pyramid Exit ──────────────────────────────────────────────


def test_short_pyramid_exit() -> None:
    """Test 14: SHORT entry exits exactly one spacing below its entry price."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Trigger and fill Level 1 SELL (22300)
    entry_bar = make_bar(close=22320.0, high=22350.0, low=22000.0)
    entry_orders = strategy.on_bar(entry_bar)
    fill = Fill(
        order_id=entry_orders[0].client_order_id,
        filled_price=22300.0,
        filled_qty=1,
        timestamp=entry_bar.timestamp,
        side=OrderSide.SELL,
    )
    strategy.on_fill(fill)
    assert strategy.net_position == -1

    # Exit price for SHORT is entry_price - spacing = 22300 - 300 = 22000
    exit_bar = make_bar(close=21990.0, high=22250.0, low=21980.0)
    exit_orders = strategy.on_bar(exit_bar)

    assert len(exit_orders) == 1
    exit_order = exit_orders[0]
    assert exit_order.side == OrderSide.BUY
    assert exit_order.price == 22000.0
    assert exit_order.quantity == 1
    assert exit_order.client_order_id == "GRID-NIFTY50-E1-EXIT-S1"


# ── Test 15: Exit is Not Immediately Reused as Entry on Same Bar ─────────────


def test_exit_not_immediately_reused_as_entry_on_same_bar() -> None:
    """
    Test 15: If an exit occurs at a level on a bar, that level cannot be immediately
    triggered as a new entry on the same bar.
    """
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Position from Level 2 (21400)
    strategy.register_entry(level_id=2, side=OrderSide.BUY, price=21400.0, quantity=1)
    # Target exit for Level 2 is 21400 + 300 = 21700 (which is also Buy Level 1!)

    # On this bar, price rises to 21720, triggering the exit at 21700
    bar = make_bar(close=21710.0, high=21720.0, low=21450.0)
    orders = strategy.on_bar(bar)

    # Must contain exit order
    assert any("EXIT" in o.client_order_id for o in orders)
    # Must NOT contain a new BUY order for L1 on this same bar
    assert not any(o.client_order_id == "GRID-NIFTY50-E1-BUY-L1" for o in orders)


# ── Test 16: Kill Switch Blocks Entries ───────────────────────────────────────


def test_kill_switch_blocks_entries() -> None:
    """Test 16: When kill switch is engaged, new entry orders are completely blocked."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)
    strategy.enable_kill_switch()

    # Price drops crossing L1 and L2
    bar = make_bar(close=21350.0, low=21350.0)
    orders = strategy.on_bar(bar)

    assert len(orders) == 0


# ── Test 17: Kill Switch Produces Flatten Signal When Positioned ──────────────


def test_kill_switch_produces_flatten_signal_when_positioned() -> None:
    """Test 17: When kill switch is engaged and strategy is positioned, returns flatten order."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Positioned LONG 2 lots
    strategy.register_entry(level_id=1, side=OrderSide.BUY, price=21700.0, quantity=1)
    strategy.register_entry(level_id=2, side=OrderSide.BUY, price=21400.0, quantity=1)
    assert strategy.net_position == 2

    strategy.enable_kill_switch()
    bar = make_bar(close=21500.0)
    orders = strategy.on_bar(bar)

    assert len(orders) == 1
    flatten_order = orders[0]
    assert flatten_order.side == OrderSide.SELL
    assert flatten_order.quantity == 2
    assert flatten_order.client_order_id == "GRID-NIFTY50-E1-FLATTEN"


# ── Test 18: Bearish Macro Blocks New ATR Grid Long Entries ───────────────────


def test_bearish_macro_blocks_new_atr_grid_long_entries() -> None:
    """Test 18: In Regime.BEARISH, new BUY entries are blocked while SELL entries operate."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    # Bar crosses lower level (BUY L1 at 21700) under BEARISH regime
    buy_bar = make_bar(close=21650.0, low=21650.0)
    buy_orders = strategy.on_bar(buy_bar, regime=Regime.BEARISH)
    # Long entry must be blocked!
    assert len(buy_orders) == 0

    # Bar crosses upper level (SELL S1 at 22300) under BEARISH regime
    sell_bar = make_bar(close=22350.0, high=22350.0)
    sell_orders = strategy.on_bar(sell_bar, regime=Regime.BEARISH)
    # Short entry is allowed
    assert len(sell_orders) == 1
    assert sell_orders[0].side == OrderSide.SELL
    assert sell_orders[0].price == 22300.0


# ── Test 19: Deterministic Client Order ID ───────────────────────────────────


def test_deterministic_client_order_id() -> None:
    """Test 19: Strategy produces deterministic, predictable client_order_id strings."""
    strategy = ATRGridStrategy("NIFTY50", levels=3)
    strategy.establish_grid(anchor=22000.0, spacing=300.0)

    bar = make_bar(close=21650.0, low=21650.0)
    orders = strategy.on_bar(bar)
    assert orders[0].client_order_id == "GRID-NIFTY50-E1-BUY-L1"

    # Replay on fresh instance gives identical ID
    strategy2 = ATRGridStrategy("NIFTY50", levels=3)
    strategy2.establish_grid(anchor=22000.0, spacing=300.0)
    orders2 = strategy2.on_bar(bar)
    assert orders2[0].client_order_id == orders[0].client_order_id
