"""
Unit tests for the Macro Regime Engine and Macro CSV loader.

Covers:
    1. VIX scoring (<14 -> +2, 14..20 -> 0, >20 -> -2) with boundary values (14.0, 20.0, 20.01)
    2. NIFTY trend scoring (above 200 EMA -> +1, below -> -1)
    3. USDINR scoring (<84 -> +1, >=84 -> -1) with boundary values (84.0)
    4. Regime classification (score >= 2 -> BULLISH, <= -2 -> BEARISH, otherwise NEUTRAL)
    5. Hand-calculated composite scores across multiple combinations
    6. Regime parameter overrides (BULLISH, NEUTRAL, BEARISH)
    7. Immutability of baseline VegaConfig
    8. Circuit breaker activation and non-activation conditions
    9. Determinism of engine outputs
    10. Macro CSV loader: validation, error handling, duplicate dates, chronological sorting
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from vega.config import VegaConfig
from vega.macro.csv_loader import load_macro_snapshots
from vega.macro.engine import MacroRegimeEngine
from vega.macro.models import MacroSnapshot, Regime


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config() -> VegaConfig:
    """Standard VegaConfig instance for testing."""
    return VegaConfig()


@pytest.fixture
def engine(config: VegaConfig) -> MacroRegimeEngine:
    """MacroRegimeEngine instance with default config."""
    return MacroRegimeEngine(config)


def make_snapshot(
    nifty_trend: float = 0.02,
    india_vix: float = 15.0,
    usdinr: float = 82.5,
    d: date | None = None,
) -> MacroSnapshot:
    """Helper to create valid MacroSnapshot objects."""
    return MacroSnapshot(
        date=d if d is not None else date(2023, 1, 2),
        nifty_trend=float(nifty_trend),
        india_vix=float(india_vix),
        usdinr=float(usdinr),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. India VIX Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestVIXScoring:
    def test_vix_below_14_contributes_plus_2(self, engine: MacroRegimeEngine) -> None:
        """VIX < 14.0 contributes +2.0."""
        assert engine.score_vix(10.0) == 2.0
        assert engine.score_vix(12.5) == 2.0
        assert engine.score_vix(13.99) == 2.0

    def test_vix_between_14_and_20_contributes_zero(self, engine: MacroRegimeEngine) -> None:
        """14.0 <= VIX <= 20.0 contributes 0.0 (boundary values inclusive)."""
        assert engine.score_vix(14.0) == 0.0    # Lower boundary inclusive
        assert engine.score_vix(17.5) == 0.0    # Midpoint
        assert engine.score_vix(20.0) == 0.0    # Upper boundary inclusive

    def test_vix_above_20_contributes_minus_2(self, engine: MacroRegimeEngine) -> None:
        """VIX > 20.0 contributes -2.0."""
        assert engine.score_vix(20.01) == -2.0  # Just above boundary
        assert engine.score_vix(24.5) == -2.0
        assert engine.score_vix(35.0) == -2.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. NIFTY Trend Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNiftyTrendScoring:
    def test_nifty_above_200_ema_contributes_plus_1(self, engine: MacroRegimeEngine) -> None:
        """NIFTY above 200 EMA (nifty_trend >= 0) contributes +1.0."""
        assert engine.score_nifty_trend(0.05) == 1.0
        assert engine.score_nifty_trend(0.0) == 1.0     # Exactly at EMA
        assert engine.score_nifty_trend(True) == 1.0    # Boolean input

    def test_nifty_below_200_ema_contributes_minus_1(self, engine: MacroRegimeEngine) -> None:
        """NIFTY below 200 EMA (nifty_trend < 0) contributes -1.0."""
        assert engine.score_nifty_trend(-0.02) == -1.0
        assert engine.score_nifty_trend(-0.0001) == -1.0
        assert engine.score_nifty_trend(False) == -1.0  # Boolean input


# ─────────────────────────────────────────────────────────────────────────────
# 3. USD/INR Scoring Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUSDINRScoring:
    def test_usdinr_below_stress_threshold_contributes_plus_1(self, engine: MacroRegimeEngine) -> None:
        """USDINR < 84.0 contributes +1.0."""
        assert engine.score_usdinr(82.0) == 1.0
        assert engine.score_usdinr(83.99) == 1.0

    def test_usdinr_at_or_above_stress_threshold_contributes_minus_1(self, engine: MacroRegimeEngine) -> None:
        """USDINR >= 84.0 contributes -1.0 (boundary value 84.0 inclusive)."""
        assert engine.score_usdinr(84.0) == -1.0    # Boundary value
        assert engine.score_usdinr(84.5) == -1.0
        assert engine.score_usdinr(87.0) == -1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Composite Scoring & Regime Classification Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeRegimeClassification:
    def test_hand_calculated_bullish_maximum(self, engine: MacroRegimeEngine) -> None:
        """
        VIX = 12.0  -> +2
        Trend = +2% -> +1
        USDINR = 82 -> +1
        Total score = +4.0 -> BULLISH
        """
        snap = make_snapshot(nifty_trend=0.02, india_vix=12.0, usdinr=82.0)
        assert engine.calculate_score(snap) == 4.0
        assert engine.classify(snap) == Regime.BULLISH

    def test_hand_calculated_bullish_boundary(self, engine: MacroRegimeEngine) -> None:
        """
        VIX = 15.0 (normal) ->  0
        Trend = +2%         -> +1
        USDINR = 82.0       -> +1
        Total score = +2.0  -> BULLISH (boundary: score >= 2.0)
        """
        snap = make_snapshot(nifty_trend=0.02, india_vix=15.0, usdinr=82.0)
        assert engine.calculate_score(snap) == 2.0
        assert engine.classify(snap) == Regime.BULLISH

    def test_hand_calculated_neutral_zero(self, engine: MacroRegimeEngine) -> None:
        """
        VIX = 15.0 (normal) ->  0
        Trend = +2%         -> +1
        USDINR = 85.0 (stress) -> -1
        Total score = 0.0   -> NEUTRAL
        """
        snap = make_snapshot(nifty_trend=0.02, india_vix=15.0, usdinr=85.0)
        assert engine.calculate_score(snap) == 0.0
        assert engine.classify(snap) == Regime.NEUTRAL

    def test_hand_calculated_bearish_boundary(self, engine: MacroRegimeEngine) -> None:
        """
        VIX = 15.0 (normal)  ->  0
        Trend = -2% (down)   -> -1
        USDINR = 85.0 (stress)-> -1
        Total score = -2.0   -> BEARISH (boundary: score <= -2.0)
        """
        snap = make_snapshot(nifty_trend=-0.02, india_vix=15.0, usdinr=85.0)
        assert engine.calculate_score(snap) == -2.0
        assert engine.classify(snap) == Regime.BEARISH

    def test_hand_calculated_bearish_maximum(self, engine: MacroRegimeEngine) -> None:
        """
        VIX = 24.0 (high)    -> -2
        Trend = -2% (down)   -> -1
        USDINR = 85.0 (stress)-> -1
        Total score = -4.0   -> BEARISH
        """
        snap = make_snapshot(nifty_trend=-0.02, india_vix=24.0, usdinr=85.0)
        assert engine.calculate_score(snap) == -4.0
        assert engine.classify(snap) == Regime.BEARISH

    def test_evaluate_populates_snapshot_immutably(self, engine: MacroRegimeEngine) -> None:
        """evaluate() returns a new MacroSnapshot with score and regime populated."""
        raw = make_snapshot(nifty_trend=0.02, india_vix=12.0, usdinr=82.0)
        assert raw.score == 0.0
        assert raw.regime == Regime.NEUTRAL

        evaluated = engine.evaluate(raw)
        assert evaluated.score == 4.0
        assert evaluated.regime == Regime.BULLISH
        # Original snapshot must remain unmutated
        assert raw.score == 0.0
        assert raw.regime == Regime.NEUTRAL


# ─────────────────────────────────────────────────────────────────────────────
# 5. Parameter Override Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestParameterOverrides:
    def test_bullish_parameter_override(self, engine: MacroRegimeEngine, config: VegaConfig) -> None:
        """BULLISH regime maintains standard position sizing and grid spacing."""
        adjusted = engine.adjust_params(Regime.BULLISH, config)
        assert adjusted.atr_position_cap == config.atr_position_cap
        assert adjusted.atr_grid_multiplier == config.atr_grid_multiplier

    def test_neutral_parameter_override(self, engine: MacroRegimeEngine, config: VegaConfig) -> None:
        """NEUTRAL regime preserves baseline defaults."""
        adjusted = engine.adjust_params(Regime.NEUTRAL, config)
        assert adjusted.atr_position_cap == config.atr_position_cap
        assert adjusted.atr_grid_multiplier == config.atr_grid_multiplier

    def test_bearish_parameter_override(self, engine: MacroRegimeEngine, config: VegaConfig) -> None:
        """BEARISH regime enforces reduced position cap and wider grid spacing."""
        adjusted = engine.adjust_params(Regime.BEARISH, config)
        assert adjusted.atr_position_cap == config.bearish_position_cap
        assert adjusted.atr_grid_multiplier == config.bearish_grid_multiplier
        # Ensure values actually represent a reduction in risk
        assert adjusted.atr_position_cap < config.atr_position_cap
        assert adjusted.atr_grid_multiplier > config.atr_grid_multiplier

    def test_global_configuration_is_not_mutated(self, engine: MacroRegimeEngine, config: VegaConfig) -> None:
        """Calling adjust_params must never mutate the original configuration object."""
        orig_cap = config.atr_position_cap
        orig_mult = config.atr_grid_multiplier

        _ = engine.adjust_params(Regime.BEARISH, config)

        assert config.atr_position_cap == orig_cap
        assert config.atr_grid_multiplier == orig_mult


# ─────────────────────────────────────────────────────────────────────────────
# 6. Circuit Breaker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def test_circuit_breaker_activates_under_bearish_and_high_vix(self, engine: MacroRegimeEngine) -> None:
        """Circuit breaker triggers when regime is BEARISH AND VIX > 20.0."""
        # VIX=22 (-2), Trend=-0.02 (-1), USDINR=82 (+1) -> Score = -2 (BEARISH), VIX=22 > 20
        snap = make_snapshot(nifty_trend=-0.02, india_vix=22.0, usdinr=82.0)
        assert engine.classify(snap) == Regime.BEARISH
        assert engine.is_circuit_breaker_triggered(snap) is True

    def test_circuit_breaker_does_not_activate_when_bearish_but_normal_vix(
        self, engine: MacroRegimeEngine
    ) -> None:
        """Circuit breaker does NOT trigger when regime is BEARISH but VIX is normal (<= 20)."""
        # VIX=16 (0), Trend=-0.02 (-1), USDINR=85 (-1) -> Score = -2 (BEARISH), but VIX 16 <= 20
        snap = make_snapshot(nifty_trend=-0.02, india_vix=16.0, usdinr=85.0)
        assert engine.classify(snap) == Regime.BEARISH
        assert engine.is_circuit_breaker_triggered(snap) is False

    def test_circuit_breaker_does_not_activate_when_high_vix_but_neutral(
        self, engine: MacroRegimeEngine
    ) -> None:
        """Circuit breaker does NOT trigger when VIX is elevated but regime is NEUTRAL."""
        # VIX=22 (-2), Trend=0.02 (+1), USDINR=82 (+1) -> Score = 0 (NEUTRAL), VIX 22 > 20
        snap = make_snapshot(nifty_trend=0.02, india_vix=22.0, usdinr=82.0)
        assert engine.classify(snap) == Regime.NEUTRAL
        assert engine.is_circuit_breaker_triggered(snap) is False

    def test_circuit_breaker_does_not_activate_in_bullish_regime(
        self, engine: MacroRegimeEngine
    ) -> None:
        """Circuit breaker does NOT trigger in BULLISH regime."""
        snap = make_snapshot(nifty_trend=0.02, india_vix=12.0, usdinr=82.0)
        assert engine.classify(snap) == Regime.BULLISH
        assert engine.is_circuit_breaker_triggered(snap) is False


# ─────────────────────────────────────────────────────────────────────────────
# 7. Determinism Test
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_engine_is_strictly_deterministic(self, engine: MacroRegimeEngine) -> None:
        """Identical inputs must always produce identical scores, regimes, and decisions."""
        snap = make_snapshot(nifty_trend=0.015, india_vix=14.5, usdinr=82.7)

        score_1 = engine.calculate_score(snap)
        score_2 = engine.calculate_score(snap)
        regime_1 = engine.classify(snap)
        regime_2 = engine.classify(snap)
        cb_1 = engine.is_circuit_breaker_triggered(snap)
        cb_2 = engine.is_circuit_breaker_triggered(snap)

        assert score_1 == score_2
        assert regime_1 == regime_2
        assert cb_1 == cb_2


# ─────────────────────────────────────────────────────────────────────────────
# 8. Macro CSV Loader & Error Handling Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroCSVLoader:
    def test_load_real_macro_data_csv(self) -> None:
        """Existing data/macro/macro_data.csv must load successfully with 120 rows."""
        filepath = Path("data/macro/macro_data.csv")
        snapshots = load_macro_snapshots(filepath)
        assert len(snapshots) == 120
        assert snapshots[0].date == date(2023, 1, 2)
        assert snapshots[0].india_vix == pytest.approx(14.51)
        assert snapshots[0].usdinr == pytest.approx(82.58)

    def test_chronological_ordering(self, tmp_path: Path) -> None:
        """Shuffled CSV rows must be returned chronologically sorted."""
        csv_content = (
            "date,nifty_trend,india_vix,usdinr\n"
            "2023-01-05,0.03,15.4,82.7\n"
            "2023-01-02,0.02,14.5,82.5\n"
            "2023-01-03,0.02,14.8,82.6\n"
        )
        csv_file = tmp_path / "shuffled.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        snapshots = load_macro_snapshots(csv_file)
        assert len(snapshots) == 3
        assert snapshots[0].date == date(2023, 1, 2)
        assert snapshots[1].date == date(2023, 1, 3)
        assert snapshots[2].date == date(2023, 1, 5)

    def test_duplicate_dates_rejected(self, tmp_path: Path) -> None:
        """Duplicate dates must raise ValueError."""
        csv_content = (
            "date,nifty_trend,india_vix,usdinr\n"
            "2023-01-02,0.02,14.5,82.5\n"
            "2023-01-02,0.03,14.6,82.6\n"
        )
        csv_file = tmp_path / "dup.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(ValueError, match="Duplicate date"):
            load_macro_snapshots(csv_file)

    def test_missing_column_rejected(self, tmp_path: Path) -> None:
        """Missing required column raises ValueError."""
        csv_content = (
            "date,nifty_trend,india_vix\n"  # missing usdinr
            "2023-01-02,0.02,14.5\n"
        )
        csv_file = tmp_path / "missing_col.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(ValueError, match="missing required columns"):
            load_macro_snapshots(csv_file)

    def test_malformed_numeric_row_rejected(self, tmp_path: Path) -> None:
        """Non-numeric values in numeric fields raise ValueError with line number."""
        csv_content = (
            "date,nifty_trend,india_vix,usdinr\n"
            "2023-01-02,0.02,INVALID_VIX,82.5\n"
        )
        csv_file = tmp_path / "malformed.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(ValueError, match="line 2"):
            load_macro_snapshots(csv_file)

    def test_negative_vix_rejected(self, tmp_path: Path) -> None:
        """Impossible non-positive VIX raises ValueError."""
        csv_content = (
            "date,nifty_trend,india_vix,usdinr\n"
            "2023-01-02,0.02,-5.0,82.5\n"
        )
        csv_file = tmp_path / "neg_vix.csv"
        csv_file.write_text(csv_content, encoding="utf-8")

        with pytest.raises(ValueError, match="india_vix must be positive"):
            load_macro_snapshots(csv_file)

    def test_nonexistent_file_raises_file_not_found(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_macro_snapshots("nonexistent_macro_file.csv")
