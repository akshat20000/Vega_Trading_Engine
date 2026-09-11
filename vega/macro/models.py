"""
Data models for the Macro Regime Engine.

Defines:
    - Regime: Enum representing the three market macro states (BULLISH, NEUTRAL, BEARISH).
    - MacroSnapshot: Dataclass capturing macro proxy observations for a given date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum


class Regime(Enum):
    """
    Macro market regime classification.

    - BULLISH: Favorable macro conditions (low volatility, upward trend, stable currency).
    - NEUTRAL: Mixed or transitional macro conditions; default baseline parameters.
    - BEARISH: Adverse macro conditions (elevated volatility, downward trend, currency stress).
    """

    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"


@dataclass
class MacroSnapshot:
    """
    A single snapshot of macro indicators for a specific trading day.

    Attributes:
        nifty_trend: Precomputed normalized trend signal representing the NIFTY
                     relationship to its 200-day EMA:
                         nifty_trend = (NIFTY close - NIFTY 200-day EMA) / NIFTY 200-day EMA
                     Convention:
                         nifty_trend >= 0.0 -> NIFTY is at/above its 200-day EMA
                         nifty_trend < 0.0  -> NIFTY is below its 200-day EMA
                     Note: This is a precomputed input; the engine does NOT calculate
                     a 200-day EMA from the 120 synthetic macro rows.
        india_vix: India VIX index level (annualised implied volatility).
        usdinr: USD/INR spot exchange rate.
        score: Composite macro score computed by MacroRegimeEngine (-4.0 to +4.0).
        regime: Regime classified by MacroRegimeEngine based on composite score.
    """

    date: date | datetime
    nifty_trend: float
    india_vix: float
    usdinr: float
    score: float = 0.0
    regime: Regime = Regime.NEUTRAL

    def __post_init__(self) -> None:
        """Validate that macro inputs are realistic and well-formed."""
        if not isinstance(self.date, (date, datetime)):
            raise ValueError(
                f"date must be a datetime.date or datetime.datetime, got {type(self.date).__name__}"
            )
        if self.india_vix <= 0.0:
            raise ValueError(
                f"india_vix must be positive, got {self.india_vix}"
            )
        if self.usdinr <= 0.0:
            raise ValueError(
                f"usdinr must be positive, got {self.usdinr}"
            )
        if not isinstance(self.regime, Regime):
            raise ValueError(
                f"regime must be an instance of Regime enum, got {type(self.regime).__name__}"
            )

    @property
    def is_nifty_above_200ema(self) -> bool:
        """
        Return True if NIFTY is above its 200-day EMA.

        Convention:
            nifty_trend represents (Close - EMA_200) / EMA_200.
            nifty_trend >= 0.0 -> True (at or above 200-day EMA).
            nifty_trend < 0.0  -> False (below 200-day EMA).
        """
        return self.nifty_trend >= 0.0
