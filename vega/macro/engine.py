"""
Deterministic Macro Regime Engine.

Evaluates macro market proxy conditions (India VIX, NIFTY 200-day trend, USDINR),
computes an objective composite score, classifies the market regime
(BULLISH, NEUTRAL, BEARISH), provides regime-specific parameter overrides,
and evaluates circuit-breaker conditions.

Scoring Rules:
    1. India VIX (Implied Volatility):
       - VIX < vix_low_threshold (14.0):              +2.0  (complacent/bullish)
       - vix_low_threshold <= VIX <= vix_high_threshold (14.0 .. 20.0): 0.0  (normal)
       - VIX > vix_high_threshold (20.0):             -2.0  (elevated volatility/fear)

    2. NIFTY 200-Day Trend (Medium-Term Trend):
       - NIFTY above 200-day EMA (nifty_trend >= 0):  +1.0  (structural uptrend)
       - NIFTY below 200-day EMA (nifty_trend < 0):   -1.0  (structural downtrend)

    3. USD/INR (Currency Stress):
       - USDINR < usdinr_stress_level (84.0):         +1.0  (stable currency)
       - USDINR >= usdinr_stress_level (84.0):        -1.0  (currency stress / capital flight)

Regime Classification:
    - Composite Score >= +2.0: BULLISH
    - Composite Score <= -2.0: BEARISH
    - Otherwise:              NEUTRAL

Parameter Overrides:
    - BULLISH: Standard position sizing and grid spacing.
    - NEUTRAL: Unchanged default configuration parameters.
    - BEARISH: Reduced position cap (bearish_position_cap) and wider grid spacing
               (bearish_grid_multiplier) to protect capital.
    Original configuration objects are never mutated.

Circuit Breaker:
    - Triggers when the macro regime is BEARISH AND India VIX exceeds the
      high-volatility threshold (circuit_breaker_vix_threshold).
    - Blocks new order entries under extreme stress.
"""

from __future__ import annotations

import dataclasses

from vega.config import VegaConfig
from vega.macro.models import MacroSnapshot, Regime


class MacroRegimeEngine:
    """
    Deterministic rule-based Macro Regime Engine.

    All thresholds and overrides are driven by VegaConfig to avoid magic numbers.
    """

    def __init__(self, config: VegaConfig | None = None) -> None:
        """
        Initialize the engine with an optional VegaConfig.

        Args:
            config: Configuration object containing macro thresholds and overrides.
                    If None, a default VegaConfig is used.
        """
        self.config = config if config is not None else VegaConfig()

    def _get_config(self, config: VegaConfig | None = None) -> VegaConfig:
        """Return the provided config or fallback to the instance config."""
        return config if config is not None else self.config

    # ── Component Scoring Methods ─────────────────────────────────────────────

    def score_vix(self, india_vix: float, config: VegaConfig | None = None) -> float:
        """
        Score India VIX.

        Rules:
            VIX < 14:          +2.0
            14 <= VIX <= 20:    0.0
            VIX > 20:          -2.0
        """
        cfg = self._get_config(config)
        if india_vix < cfg.vix_low_threshold:
            return 2.0
        if india_vix <= cfg.vix_high_threshold:
            return 0.0
        return -2.0

    def score_nifty_trend(self, nifty_trend: float | bool) -> float:
        """
        Score NIFTY relative to its 200-day EMA.

        The `nifty_trend` field is a precomputed normalized trend signal representing
        the NIFTY relationship to its 200-day EMA:
            nifty_trend = (NIFTY close - NIFTY 200-day EMA) / NIFTY 200-day EMA

        Rules:
            nifty_trend >= 0 (or True)  -> +1.0 (at or above 200-day EMA)
            nifty_trend < 0  (or False) -> -1.0 (below 200-day EMA)

        Note:
            We do NOT calculate a 200-day EMA from the 120 synthetic macro rows.
            The input represents this precomputed relationship directly.
        """
        if isinstance(nifty_trend, bool):
            return 1.0 if nifty_trend else -1.0
        return 1.0 if nifty_trend >= 0.0 else -1.0

    def score_usdinr(self, usdinr: float, config: VegaConfig | None = None) -> float:
        """
        Score USD/INR exchange rate against stress level.

        Rules:
            USDINR < stress_level (84.0):  +1.0
            USDINR >= stress_level (84.0): -1.0
        """
        cfg = self._get_config(config)
        if usdinr < cfg.usdinr_stress_level:
            return 1.0
        return -1.0

    # ── Composite Evaluation Methods ──────────────────────────────────────────

    def calculate_score(
        self, snapshot: MacroSnapshot, config: VegaConfig | None = None
    ) -> float:
        """
        Calculate total composite macro score for a snapshot.

        Score range is -4.0 to +4.0.
        """
        cfg = self._get_config(config)
        vix_score = self.score_vix(snapshot.india_vix, cfg)
        trend_score = self.score_nifty_trend(snapshot.nifty_trend)
        usdinr_score = self.score_usdinr(snapshot.usdinr, cfg)
        return float(vix_score + trend_score + usdinr_score)

    def classify(
        self, snapshot: MacroSnapshot, config: VegaConfig | None = None
    ) -> Regime:
        """
        Classify a snapshot into BULLISH, NEUTRAL, or BEARISH regime.

        Rules:
            score >= bullish_score_threshold (2.0):  BULLISH
            score <= bearish_score_threshold (-2.0): BEARISH
            otherwise:                               NEUTRAL
        """
        cfg = self._get_config(config)
        score = self.calculate_score(snapshot, cfg)

        if score >= cfg.bullish_score_threshold:
            return Regime.BULLISH
        if score <= cfg.bearish_score_threshold:
            return Regime.BEARISH
        return Regime.NEUTRAL

    def evaluate(
        self, snapshot: MacroSnapshot, config: VegaConfig | None = None
    ) -> MacroSnapshot:
        """
        Evaluate a snapshot and return a new MacroSnapshot with score and regime populated.

        The original snapshot is not mutated.
        """
        cfg = self._get_config(config)
        score = self.calculate_score(snapshot, cfg)
        regime = self.classify(snapshot, cfg)
        return dataclasses.replace(snapshot, score=score, regime=regime)

    # ── Parameter Overrides ───────────────────────────────────────────────────

    def adjust_params(
        self, regime: Regime, config: VegaConfig | None = None
    ) -> VegaConfig:
        """
        Return a new VegaConfig with regime-specific parameter overrides applied.

        Rules:
            BULLISH: Standard position sizing and grid spacing.
            NEUTRAL: Unchanged default configuration.
            BEARISH: Reduced position cap and wider grid spacing to limit drawdowns.

        The input configuration is NEVER mutated.

        Args:
            regime: Current macro regime.
            config: Baseline configuration. If None, instance config is used.

        Returns:
            A new VegaConfig instance with overrides applied.
        """
        base_cfg = self._get_config(config)

        if regime == Regime.BEARISH:
            return dataclasses.replace(
                base_cfg,
                atr_position_cap=base_cfg.bearish_position_cap,
                atr_grid_multiplier=base_cfg.bearish_grid_multiplier,
            )

        # BULLISH and NEUTRAL use standard configured parameters
        return dataclasses.replace(base_cfg)

    # ── Circuit Breaker ───────────────────────────────────────────────────────

    def is_circuit_breaker_triggered(
        self, snapshot: MacroSnapshot, config: VegaConfig | None = None
    ) -> bool:
        """
        Evaluate whether the macro circuit breaker is triggered.

        The circuit breaker triggers when:
            regime is BEARISH AND India VIX > circuit_breaker_vix_threshold (20.0).

        Args:
            snapshot: Current macro snapshot.
            config: Configuration object.

        Returns:
            True if circuit breaker is active (block trading), False otherwise.
        """
        cfg = self._get_config(config)
        regime = self.classify(snapshot, cfg)
        is_bearish = regime == Regime.BEARISH
        is_high_vix = snapshot.india_vix > cfg.circuit_breaker_vix_threshold

        return bool(is_bearish and is_high_vix)
