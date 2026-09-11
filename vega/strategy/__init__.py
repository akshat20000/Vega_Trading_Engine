"""
Strategy layer for the Vega Quant Trading Engine.

Exports:
    - BaseStrategy: Abstract base strategy interface
    - ATRGridStrategy: Mean-reversion ATR Grid strategy
    - GridLevel: Dataclass representing a price level on the grid
    - GridPosition: Dataclass representing an active filled pyramid entry
    - StopAndReverseStrategy: EMA 9/21 trend-following reversal strategy
    - SARState: Target position state enum (FLAT, LONG, SHORT)
"""

from vega.strategy.atr_grid import ATRGridStrategy, GridLevel, GridPosition
from vega.strategy.base import BaseStrategy
from vega.strategy.stop_and_reverse import SARState, StopAndReverseStrategy

__all__ = [
    "BaseStrategy",
    "ATRGridStrategy",
    "GridLevel",
    "GridPosition",
    "StopAndReverseStrategy",
    "SARState",
]
