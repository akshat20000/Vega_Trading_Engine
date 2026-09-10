"""
Technical analysis indicators implemented from first principles.

This package provides pure functions for core trading indicators:
- ema: Exponential Moving Average
- rsi: Relative Strength Index (Wilder)
- atr: Average True Range (Wilder) and true_range
- obv: On-Balance Volume (Granville)
"""

from vega.indicators.atr import atr, true_range
from vega.indicators.ema import ema
from vega.indicators.obv import obv
from vega.indicators.rsi import rsi

__all__ = [
    "ema",
    "rsi",
    "atr",
    "true_range",
    "obv",
]
