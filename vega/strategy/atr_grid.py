"""
ATR Grid Strategy for the Vega Quant Trading Engine.

Implements a deterministic ATR-based mean-reversion grid trading strategy.

Grid Definition:
    anchor  = current bar close (at grid establishment)
    spacing = ATR(atr_period) * grid_multiplier

    Levels below anchor:
        L_k = anchor - k * spacing (k = 1 .. levels) -> BUY
    Levels above anchor:
        S_k = anchor + k * spacing (k = 1 .. levels) -> SELL

Deterministic Bar Crossing Convention:
    OHLC data does not disclose the true intrabar price trajectory.
    When a single bar crosses multiple grid levels, we apply the following
    deterministic simulation convention (NOT actual intrabar sequencing):
        1. Process lower BUY levels first, from nearest-to-anchor to furthest
           (descending price order: L_1, L_2, L_3, ...).
        2. Process upper SELL levels next, from nearest-to-anchor to furthest
           (ascending price order: S_1, S_2, S_3, ...).
    If both sides are crossed in a single bar, orders are generated in this order.

Core Invariants:
    - on_bar() generates Order objects; it does NOT mutate filled position state.
    - on_fill() updates actual strategy position state upon execution.
    - Re-anchoring is allowed ONLY when net position is flat (net_position == 0).
    - Long pyramid entries exit at entry_price + spacing.
    - Short pyramid entries exit at entry_price - spacing.
    - An exited level cannot be reused as an entry within the same bar.
    - Kill switch blocks all new entries and emits a flatten signal if positioned.
    - Bearish macro regime blocks new BUY entries (exits and shorts continue).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vega.config import VegaConfig
from vega.data.models import Bar
from vega.indicators.atr import atr
from vega.macro.models import Regime
from vega.orders.models import Fill, Order, OrderSide, OrderStatus
from vega.strategy.base import BaseStrategy


@dataclass(frozen=True)
class GridLevel:
    """
    Represents one price level on the grid.

    Attributes:
        level_id: 1-indexed distance from anchor (1, 2, 3, ...).
        side: OrderSide.BUY for levels below anchor, OrderSide.SELL for above.
        price: Price threshold of the level.
    """

    level_id: int
    side: OrderSide
    price: float


@dataclass
class GridPosition:
    """
    Represents an active filled pyramid entry.

    Attributes:
        epoch: Grid epoch index during which entry was placed.
        level_id: Grid level index (1, 2, 3, ...).
        side: OrderSide of entry (BUY for long, SELL for short).
        entry_price: Executed fill price.
        quantity: Units filled.
        client_order_id: Idempotency identifier of the entry order.
    """

    epoch: int
    level_id: int
    side: OrderSide
    entry_price: float
    quantity: int
    client_order_id: str


class ATRGridStrategy(BaseStrategy):
    """
    Mean-reversion ATR Grid Strategy.

    Generates BUY orders below anchor and SELL orders above anchor, with
    pyramiding up to position cap, deterministic crossing evaluation,
    and 1-spacing profit-taking exits.
    """

    def __init__(
        self,
        symbol: str,
        atr_period: int = 14,
        grid_multiplier: float = 1.5,
        levels: int = 3,
        position_cap: int = 5,
        quantity_per_level: int = 1,
        config: VegaConfig | None = None,
    ) -> None:
        """
        Initialize ATR Grid Strategy with explicit configuration.

        Args:
            symbol: Ticker symbol (e.g. 'NIFTY50').
            atr_period: Lookback period for ATR volatility calculation.
            grid_multiplier: Multiplier applied to ATR to determine grid spacing.
            levels: Number of grid levels above and below the anchor.
            position_cap: Maximum absolute net position (in lots) allowed.
            quantity_per_level: Lot size per pyramid entry.
            config: Optional VegaConfig to provide baseline defaults if needed.
        """
        super().__init__(symbol=symbol)

        if config is not None:
            atr_period = config.atr_period
            grid_multiplier = config.atr_grid_multiplier
            position_cap = config.atr_position_cap

        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if grid_multiplier <= 0.0:
            raise ValueError(f"grid_multiplier must be positive, got {grid_multiplier}")
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if position_cap < 1:
            raise ValueError(f"position_cap must be >= 1, got {position_cap}")
        if quantity_per_level < 1:
            raise ValueError(f"quantity_per_level must be >= 1, got {quantity_per_level}")

        self.atr_period: int = atr_period
        self.grid_multiplier: float = float(grid_multiplier)
        self.levels: int = levels
        self.position_cap: int = position_cap
        self.quantity_per_level: int = quantity_per_level

        # Grid state
        self.epoch: int = 0
        self.anchor: float | None = None
        self.spacing: float | None = None
        self.buy_levels: list[GridLevel] = []
        self.sell_levels: list[GridLevel] = []

        # Tracking triggered entries per grid epoch (to avoid duplicate signals)
        self.triggered_levels: set[tuple[OrderSide, int]] = set()

        # Filled position state (MUTATED ONLY IN on_fill)
        self.net_position: int = 0
        self.active_entries: list[GridPosition] = []

        # Order tracking (for matching fills to entries/exits)
        self.pending_entry_orders: dict[str, tuple[int, OrderSide, float, int]] = {}
        self.pending_exit_orders: dict[str, str] = {}

        # Bar history for ATR calculation
        self.history: list[Bar] = []

    # ── Grid Management ─────────────────────────────────────────────────────────

    def establish_grid(
        self,
        anchor: float,
        spacing: float | None = None,
        atr_value: float | None = None,
    ) -> None:
        """
        Establish a new grid centered at `anchor` with given spacing or ATR.

        Clears the previous grid's triggered levels and increments the epoch.

        Args:
            anchor: Price anchor (typically current bar close).
            spacing: Precomputed grid spacing. If None, calculated from atr_value.
            atr_value: ATR value used to calculate spacing = atr_value * grid_multiplier.
        """
        if spacing is not None:
            calc_spacing = float(spacing)
        elif atr_value is not None:
            calc_spacing = float(atr_value) * self.grid_multiplier
        else:
            raise ValueError("Either spacing or atr_value must be provided to establish_grid.")

        if calc_spacing <= 0.0:
            raise ValueError(f"Grid spacing must be positive, got {calc_spacing}")

        self.anchor = float(anchor)
        self.spacing = calc_spacing
        self.epoch += 1
        self.triggered_levels.clear()

        # Buy levels below anchor: L_k = anchor - k * spacing
        # nearest to anchor is level_id=1
        self.buy_levels = [
            GridLevel(level_id=k, side=OrderSide.BUY, price=self.anchor - k * self.spacing)
            for k in range(1, self.levels + 1)
        ]

        # Sell levels above anchor: S_k = anchor + k * spacing
        # nearest to anchor is level_id=1
        self.sell_levels = [
            GridLevel(level_id=k, side=OrderSide.SELL, price=self.anchor + k * self.spacing)
            for k in range(1, self.levels + 1)
        ]

    def can_reanchor(self) -> bool:
        """Return True if grid is allowed to re-anchor (only when net position is flat)."""
        return self.net_position == 0

    def reanchor(
        self,
        bar: Bar,
        spacing: float | None = None,
        atr_value: float | None = None,
    ) -> None:
        """
        Re-anchor the grid to a new bar close.

        Strictly enforces Rule 8: The grid may re-anchor ONLY when net position is flat.

        Args:
            bar: The current Bar providing the new anchor price (bar.close).
            spacing: Optional explicit spacing. If None, calculated from ATR.
            atr_value: Optional explicit ATR value.

        Raises:
            ValueError: If net_position is not flat (net_position != 0).
        """
        if not self.can_reanchor():
            raise ValueError(
                f"Cannot re-anchor grid: net position is not flat (net_position={self.net_position})."
            )

        calc_spacing = spacing
        if calc_spacing is None:
            if atr_value is not None:
                calc_spacing = atr_value * self.grid_multiplier
            elif len(self.history) >= self.atr_period:
                current_atr = atr(self.history, period=self.atr_period)[-1]
                calc_spacing = current_atr * self.grid_multiplier
            elif self.spacing is not None:
                calc_spacing = self.spacing
            else:
                raise ValueError("Cannot calculate spacing: insufficient bar history for ATR.")

        self.establish_grid(anchor=bar.close, spacing=calc_spacing)

    # ── Signal Generation (on_bar) ─────────────────────────────────────────────

    def on_bar(
        self,
        bar: Bar,
        regime: Regime | None = None,
    ) -> list[Order]:
        """
        Process a bar, evaluate exits, and evaluate entry crossings.

        Does NOT mutate filled position state (mutated only in on_fill).

        Args:
            bar: Current OHLCV Bar.
            regime: Current macro regime classification.

        Returns:
            List of generated Order objects in deterministic order.
        """
        self.history.append(bar)

        # 1. Automatic initial grid establishment if not established yet
        if self.anchor is None:
            if len(self.history) >= self.atr_period:
                atr_vals = atr(self.history, period=self.atr_period)
                current_atr = atr_vals[-1]
                self.establish_grid(anchor=bar.close, atr_value=current_atr)
            else:
                # Still in ATR warmup period
                return []

        orders: list[Order] = []

        # 2. Kill switch evaluation
        if self.kill_switch:
            if self.net_position != 0:
                flatten_side = OrderSide.SELL if self.net_position > 0 else OrderSide.BUY
                flatten_qty = abs(self.net_position)
                flatten_order = Order(
                    client_order_id=f"GRID-{self.symbol}-E{self.epoch}-FLATTEN",
                    symbol=self.symbol,
                    side=flatten_side,
                    quantity=flatten_qty,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                orders.append(flatten_order)
            # Kill switch blocks any new entries
            return orders

        # 3. Profit-Taking Exits Evaluation
        # LONG entry exits one spacing above entry_price
        # SHORT entry exits one spacing below entry_price
        exited_level_ids: set[tuple[OrderSide, int]] = set()
        exited_prices: set[float] = set()

        if self.spacing is not None:
            for entry in list(self.active_entries):
                if entry.side == OrderSide.BUY:
                    exit_price = entry.entry_price + self.spacing
                    if bar.high >= exit_price:
                        exit_order = Order(
                            client_order_id=f"GRID-{self.symbol}-E{entry.epoch}-EXIT-L{entry.level_id}",
                            symbol=self.symbol,
                            side=OrderSide.SELL,
                            quantity=entry.quantity,
                            price=exit_price,
                            status=OrderStatus.PENDING,
                        )
                        orders.append(exit_order)
                        self.pending_exit_orders[exit_order.client_order_id] = entry.client_order_id
                        exited_level_ids.add((OrderSide.BUY, entry.level_id))
                        exited_prices.add(exit_price)

                elif entry.side == OrderSide.SELL:
                    exit_price = entry.entry_price - self.spacing
                    if bar.low <= exit_price:
                        exit_order = Order(
                            client_order_id=f"GRID-{self.symbol}-E{entry.epoch}-EXIT-S{entry.level_id}",
                            symbol=self.symbol,
                            side=OrderSide.BUY,
                            quantity=entry.quantity,
                            price=exit_price,
                            status=OrderStatus.PENDING,
                        )
                        orders.append(exit_order)
                        self.pending_exit_orders[exit_order.client_order_id] = entry.client_order_id
                        exited_level_ids.add((OrderSide.SELL, entry.level_id))
                        exited_prices.add(exit_price)

        # 4. Entry Crossing Evaluation
        # Deterministic simulation convention:
        # Step A: Lower BUY levels first (nearest to anchor to furthest: L_1, L_2, ...)
        # Step B: Upper SELL levels next (nearest to anchor to furthest: S_1, S_2, ...)
        
        # Track pending orders generated in this bar so we do not exceed position cap
        pending_order_delta = 0

        # Step A: Lower BUY levels (descending price order)
        # Rule 18: BEARISH macro regime blocks new BUY entries
        if regime != Regime.BEARISH:
            # Sorted by level_id ascending -> price descending (nearest to anchor first)
            sorted_buy_levels = sorted(self.buy_levels, key=lambda lvl: lvl.level_id)
            for lvl in sorted_buy_levels:
                key = (OrderSide.BUY, lvl.level_id)
                # Check crossing
                if bar.low <= lvl.price:
                    # Check not already triggered in this epoch
                    if key in self.triggered_levels:
                        continue
                    # Rule 10/15: Exited level cannot be reused as entry on same bar
                    if key in exited_level_ids or any(abs(lvl.price - p) < 1e-4 for p in exited_prices):
                        continue
                    # Position cap check
                    projected_pos = self.net_position + pending_order_delta + self.quantity_per_level
                    if projected_pos > self.position_cap:
                        continue

                    buy_order = Order(
                        client_order_id=f"GRID-{self.symbol}-E{self.epoch}-BUY-L{lvl.level_id}",
                        symbol=self.symbol,
                        side=OrderSide.BUY,
                        quantity=self.quantity_per_level,
                        price=lvl.price,
                        status=OrderStatus.PENDING,
                    )
                    orders.append(buy_order)
                    self.triggered_levels.add(key)
                    self.pending_entry_orders[buy_order.client_order_id] = (
                        lvl.level_id,
                        OrderSide.BUY,
                        lvl.price,
                        self.quantity_per_level,
                    )
                    pending_order_delta += self.quantity_per_level

        # Step B: Upper SELL levels (ascending price order)
        # Sorted by level_id ascending -> price ascending (nearest to anchor first)
        sorted_sell_levels = sorted(self.sell_levels, key=lambda lvl: lvl.level_id)
        for lvl in sorted_sell_levels:
            key = (OrderSide.SELL, lvl.level_id)
            # Check crossing
            if bar.high >= lvl.price:
                # Check not already triggered in this epoch
                if key in self.triggered_levels:
                    continue
                # Exited level cannot be reused as entry on same bar
                if key in exited_level_ids or any(abs(lvl.price - p) < 1e-4 for p in exited_prices):
                    continue
                # Position cap check
                projected_pos = self.net_position + pending_order_delta - self.quantity_per_level
                if abs(projected_pos) > self.position_cap:
                    continue

                sell_order = Order(
                    client_order_id=f"GRID-{self.symbol}-E{self.epoch}-SELL-S{lvl.level_id}",
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=self.quantity_per_level,
                    price=lvl.price,
                    status=OrderStatus.PENDING,
                )
                orders.append(sell_order)
                self.triggered_levels.add(key)
                self.pending_entry_orders[sell_order.client_order_id] = (
                    lvl.level_id,
                    OrderSide.SELL,
                    lvl.price,
                    self.quantity_per_level,
                )
                pending_order_delta -= self.quantity_per_level

        return orders

    # ── Position State Mutator (on_fill) ───────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        """
        Update actual strategy position state upon receiving confirmed fill.

        Strictly adheres to:
            Strategy position state represents ACTUAL FILLED POSITION only.

        Args:
            fill: Execution Fill object.
        """
        # Case 1: Entry order fill
        if fill.order_id in self.pending_entry_orders:
            level_id, side, price, qty = self.pending_entry_orders.pop(fill.order_id)
            entry = GridPosition(
                epoch=self.epoch,
                level_id=level_id,
                side=side,
                entry_price=fill.filled_price,
                quantity=fill.filled_qty,
                client_order_id=fill.order_id,
            )
            self.active_entries.append(entry)
            if side == OrderSide.BUY:
                self.net_position += fill.filled_qty
            else:
                self.net_position -= fill.filled_qty
            return

        # Case 2: Exit order fill
        if fill.order_id in self.pending_exit_orders:
            entry_order_id = self.pending_exit_orders.pop(fill.order_id)
            # Remove matching active entry
            self.active_entries = [
                e for e in self.active_entries if e.client_order_id != entry_order_id
            ]
            if fill.side == OrderSide.SELL:
                self.net_position -= fill.filled_qty
            else:
                self.net_position += fill.filled_qty
            return

        # Case 3: Direct or Flatten fill
        if "FLATTEN" in fill.order_id:
            self.active_entries.clear()
            self.net_position = 0
            return

        # Fallback for generic / direct fills (e.g. in standalone tests)
        fill_side = fill.side or (OrderSide.BUY if "BUY" in fill.order_id else OrderSide.SELL)
        if fill_side == OrderSide.BUY:
            self.net_position += fill.filled_qty
        else:
            self.net_position -= fill.filled_qty

    def register_entry(
        self,
        level_id: int,
        side: OrderSide,
        price: float,
        quantity: int = 1,
    ) -> None:
        """
        Direct helper to register a confirmed entry (useful in test fixtures).

        Args:
            level_id: Grid level index (1, 2, ...).
            side: OrderSide (BUY or SELL).
            price: Executed price.
            quantity: Executed quantity.
        """
        client_order_id = (
            f"GRID-{self.symbol}-E{self.epoch}-{'BUY' if side == OrderSide.BUY else 'SELL'}-"
            f"{'L' if side == OrderSide.BUY else 'S'}{level_id}"
        )
        entry = GridPosition(
            epoch=self.epoch,
            level_id=level_id,
            side=side,
            entry_price=float(price),
            quantity=int(quantity),
            client_order_id=client_order_id,
        )
        self.active_entries.append(entry)
        self.triggered_levels.add((side, level_id))
        if side == OrderSide.BUY:
            self.net_position += quantity
        else:
            self.net_position -= quantity

    def get_pyramid_count(self) -> int:
        """Return the current count of active filled pyramid entry legs."""
        return len(self.active_entries)

