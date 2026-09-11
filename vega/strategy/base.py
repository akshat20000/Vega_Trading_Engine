"""
Base strategy interface for the Vega Quant Trading Engine.

Defines:
    - BaseStrategy: Abstract base class for all trading strategies.

Principles:
    - Strategies receive market information and produce Order objects.
    - Strategies do NOT execute orders directly through Broker.
    - Strategies do NOT modify Portfolio directly.
    - Strategies do NOT call RiskManager.
    - on_bar() generates orders/signals; it does NOT mutate position state.
    - on_fill() updates actual strategy position state upon execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from vega.data.models import Bar
from vega.macro.models import Regime
from vega.orders.models import Fill, Order


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Attributes:
        symbol: The instrument identifier traded by this strategy.
        kill_switch: Flag indicating if the strategy kill switch is engaged.
    """

    def __init__(self, symbol: str) -> None:
        """
        Initialize base strategy attributes.

        Args:
            symbol: Ticker symbol (e.g. 'NIFTY50').
        """
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")
        self.symbol: str = symbol.strip()
        self.kill_switch: bool = False

    def enable_kill_switch(self) -> None:
        """Engage the emergency kill switch."""
        self.kill_switch = True

    def disable_kill_switch(self) -> None:
        """Disengage the emergency kill switch."""
        self.kill_switch = False

    @abstractmethod
    def on_bar(
        self,
        bar: Bar,
        regime: Regime | None = None,
    ) -> list[Order]:
        """
        Process a new market data bar and return zero or more Order objects.

        This method must NOT mutate the strategy's filled position state.
        Actual position state is mutated only upon receiving confirmed fills
        in on_fill().

        Args:
            bar: The current OHLCV Bar.
            regime: Current macro regime classification (optional).

        Returns:
            A list of proposed Order objects (may be empty).
        """
        pass

    @abstractmethod
    def on_fill(self, fill: Fill) -> None:
        """
        Process a confirmed execution fill and update actual position state.

        Args:
            fill: The execution Fill returned by the broker/matching engine.
        """
        pass

    def on_bars(
        self,
        bars: Sequence[Bar],
        regime: Regime | None = None,
    ) -> list[Order]:
        """
        Convenience helper to process a sequence of bars in chronological order.

        Args:
            bars: Chronological sequence of Bar objects.
            regime: Current macro regime classification (optional).

        Returns:
            Combined list of all generated orders across bars.
        """
        all_orders: list[Order] = []
        for bar in bars:
            orders = self.on_bar(bar, regime=regime)
            all_orders.extend(orders)
        return all_orders

    def get_pyramid_count(self) -> int:
        """
        Return the count of active same-direction entry legs (pyramids).

        Default implementation returns 0. Subclasses with pyramiding
        support should override this to return their active entry count.
        """
        return 0

