"""
Stop-and-Reverse (SAR) Strategy for the Vega Quant Trading Engine.

Implements an Exponential Moving Average (EMA 9 / 21) crossover trend-following
strategy with a fixed target position model:
    FLAT  = 0
    LONG  = +1
    SHORT = -1

Transitions and Orders:
    FLAT  -> LONG  : BUY  1
    FLAT  -> SHORT : SELL 1
    LONG  -> SHORT : SELL 2 (reversal: close 1 + open 1)
    SHORT -> LONG  : BUY  2 (reversal: close 1 + open 1)

Core Invariants:
    - on_bar() evaluates EMA crossover and returns Order objects;
      it does NOT mutate filled position state.
    - on_fill() updates actual strategy position state upon execution.
    - No signals generated during EMA warmup or when no crossover occurs.
    - No duplicate orders generated on subsequent bars while in the same crossover state.
    - Kill switch blocks all entries and emits a flatten signal if positioned.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

from vega.config import VegaConfig
from vega.data.models import Bar
from vega.indicators.ema import ema
from vega.macro.models import Regime
from vega.orders.models import Fill, Order, OrderSide, OrderStatus
from vega.strategy.base import BaseStrategy


class SARState(Enum):
    """Target position states for Stop-and-Reverse strategy."""

    FLAT = "FLAT"
    LONG = "LONG"
    SHORT = "SHORT"


class StopAndReverseStrategy(BaseStrategy):
    """
    EMA 9/21 Stop-and-Reverse Strategy.

    Always seeks to be in the direction of the EMA trend, reversing position
    upon opposite crossover.
    """

    def __init__(
        self,
        symbol: str,
        fast_period: int = 9,
        slow_period: int = 21,
        quantity: int = 1,
        config: VegaConfig | None = None,
    ) -> None:
        """
        Initialize Stop-and-Reverse Strategy with explicit configuration.

        Args:
            symbol: Ticker symbol (e.g. 'NIFTY50').
            fast_period: Fast EMA lookback period (default: 9).
            slow_period: Slow EMA lookback period (default: 21).
            quantity: Fixed lot size (default: 1).
            config: Optional VegaConfig to provide baseline defaults.
        """
        super().__init__(symbol=symbol)

        if config is not None:
            fast_period = config.sar_fast_ema
            slow_period = config.sar_slow_ema

        if fast_period < 1:
            raise ValueError(f"fast_period must be >= 1, got {fast_period}")
        if slow_period <= fast_period:
            raise ValueError(
                f"slow_period ({slow_period}) must be strictly greater than fast_period ({fast_period})"
            )
        if quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {quantity}")

        self.fast_period: int = fast_period
        self.slow_period: int = slow_period
        self.quantity: int = quantity

        # Filled position state (MUTATED ONLY IN on_fill)
        self.state: SARState = SARState.FLAT
        self.position: int = 0

        # Bar history for EMA calculation
        self.history: list[Bar] = []

        # Crossover tracking
        self._crossover_count: int = 0
        self._last_crossover_direction: str | None = None

    def set_state(self, state: SARState, position: int | None = None) -> None:
        """
        Explicitly configure state and position (primarily for unit test fixtures).

        Args:
            state: SARState (FLAT, LONG, SHORT).
            position: Optional position quantity (defaults to 0, +quantity, or -quantity).
        """
        self.state = state
        if position is not None:
            self.position = position
        else:
            if state == SARState.FLAT:
                self.position = 0
            elif state == SARState.LONG:
                self.position = self.quantity
            elif state == SARState.SHORT:
                self.position = -self.quantity

    # ── Signal Generation (on_bar) ─────────────────────────────────────────────

    def on_bar(
        self,
        bar: Bar,
        regime: Regime | None = None,
    ) -> list[Order]:
        """
        Process a new bar, compute EMA 9/21, and detect crossovers.

        Does NOT mutate position state (position mutated only in on_fill).

        Args:
            bar: Current OHLCV Bar.
            regime: Current macro regime classification (consumed, but SAR
                    operates normally per specifications).

        Returns:
            List of generated Order objects (0 or 1 order).
        """
        self.history.append(bar)

        # 1. Kill Switch Evaluation
        if self.kill_switch:
            if self.position != 0 or self.state != SARState.FLAT:
                flatten_side = OrderSide.SELL if self.position > 0 else OrderSide.BUY
                flatten_qty = abs(self.position) if self.position != 0 else self.quantity
                flatten_order = Order(
                    client_order_id=f"SAR-{self.symbol}-FLATTEN",
                    symbol=self.symbol,
                    side=flatten_side,
                    quantity=flatten_qty,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                return [flatten_order]
            return []

        # 2. History Check (need at least slow_period + 1 bars to detect crossover)
        if len(self.history) < self.slow_period + 1:
            return []

        # 3. Calculate EMAs
        closes = [b.close for b in self.history]
        fast_ema_series = ema(closes, self.fast_period)
        slow_ema_series = ema(closes, self.slow_period)

        fast_prev = fast_ema_series[-2]
        slow_prev = slow_ema_series[-2]
        fast_curr = fast_ema_series[-1]
        slow_curr = slow_ema_series[-1]

        # 4. Detect Crossovers
        bullish_crossover = (fast_prev <= slow_prev) and (fast_curr > slow_curr)
        bearish_crossover = (fast_prev >= slow_prev) and (fast_curr < slow_curr)

        if not (bullish_crossover or bearish_crossover):
            return []

        orders: list[Order] = []

        # Case A: Bullish Crossover
        if bullish_crossover:
            if self._last_crossover_direction == "BULLISH":
                # No duplicate signal on same crossover
                return []
            self._last_crossover_direction = "BULLISH"
            self._crossover_count += 1

            if self.state == SARState.FLAT:
                # FLAT -> LONG = BUY 1
                order = Order(
                    client_order_id=f"SAR-{self.symbol}-{self._crossover_count}-BUY",
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=self.quantity,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                orders.append(order)

            elif self.state == SARState.SHORT:
                # SHORT -> LONG = BUY 2 (reversal)
                order = Order(
                    client_order_id=f"SAR-{self.symbol}-{self._crossover_count}-BUY",
                    symbol=self.symbol,
                    side=OrderSide.BUY,
                    quantity=2 * self.quantity,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                orders.append(order)

            # If already LONG, no order needed

        # Case B: Bearish Crossover
        elif bearish_crossover:
            if self._last_crossover_direction == "BEARISH":
                # No duplicate signal on same crossover
                return []
            self._last_crossover_direction = "BEARISH"
            self._crossover_count += 1

            if self.state == SARState.FLAT:
                # FLAT -> SHORT = SELL 1
                order = Order(
                    client_order_id=f"SAR-{self.symbol}-{self._crossover_count}-SELL",
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=self.quantity,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                orders.append(order)

            elif self.state == SARState.LONG:
                # LONG -> SHORT = SELL 2 (reversal)
                order = Order(
                    client_order_id=f"SAR-{self.symbol}-{self._crossover_count}-SELL",
                    symbol=self.symbol,
                    side=OrderSide.SELL,
                    quantity=2 * self.quantity,
                    price=bar.close,
                    status=OrderStatus.PENDING,
                )
                orders.append(order)

            # If already SHORT, no order needed

        return orders

    # ── Position State Mutator (on_fill) ───────────────────────────────────────

    def on_fill(self, fill: Fill) -> None:
        """
        Update actual strategy position state upon confirmed fill execution.

        Strictly adheres to:
            Strategy position state represents ACTUAL FILLED POSITION only.

        Args:
            fill: Execution Fill object.
        """
        if "FLATTEN" in fill.order_id:
            self.position = 0
            self.state = SARState.FLAT
            return

        fill_side = fill.side or (OrderSide.BUY if "BUY" in fill.order_id else OrderSide.SELL)

        if fill_side == OrderSide.BUY:
            self.position += fill.filled_qty
        else:
            self.position -= fill.filled_qty

        # Derive state from actual net position
        if self.position == 0:
            self.state = SARState.FLAT
        elif self.position > 0:
            self.state = SARState.LONG
        else:
            self.state = SARState.SHORT
