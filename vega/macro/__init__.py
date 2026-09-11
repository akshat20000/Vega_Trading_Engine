"""
Macro regime engine package.

Exports:
    - Regime: Enum of market states (BULLISH, NEUTRAL, BEARISH)
    - MacroSnapshot: Dataclass capturing macro proxy observations
    - load_macro_snapshots: CSV loader for historical macro data
    - MacroRegimeEngine: Deterministic rule-based regime classifier and parameter adapter
"""

from vega.macro.csv_loader import load_macro_snapshots
from vega.macro.engine import MacroRegimeEngine
from vega.macro.models import MacroSnapshot, Regime

__all__ = [
    "Regime",
    "MacroSnapshot",
    "load_macro_snapshots",
    "MacroRegimeEngine",
]
