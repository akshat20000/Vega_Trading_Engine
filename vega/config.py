"""
Configuration for the Vega Quant Trading Engine.

All tunable parameters are stored in this single dataclass.
No magic numbers should appear anywhere else in the codebase.

Usage:
    from vega.config import load_config
    config = load_config("config.yaml")   # loads from file
    config = load_config()                # uses all defaults (useful in tests)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class VegaConfig:
    """
    Central configuration object.

    All fields have sensible defaults so the engine can run in tests
    without a config file. In production, call load_config("config.yaml")
    to read values from the YAML file.
    """

    # ── Backtest ─────────────────────────────────────────────────────────────
    initial_capital: float = 1_000_000.0   # Starting capital in INR
    symbol: str = "NIFTY50"

    # ── Transaction Costs ─────────────────────────────────────────────────────
    slippage_pct: float = 0.001            # 0.1% slippage on fill price (t+1 open)
    brokerage_pct: float = 0.0003          # 0.03% brokerage per trade

    # ── Risk ──────────────────────────────────────────────────────────────────
    max_position_size: int = 10            # Maximum lots at any point in time
    max_pyramids: int = 3                  # Maximum pyramid entries per grid epoch
    daily_loss_limit: float = 50_000.0    # INR — triggers kill switch if breached

    # ── ATR Grid Strategy ─────────────────────────────────────────────────────
    atr_period: int = 14                   # Lookback period for ATR calculation
    atr_grid_multiplier: float = 1.5       # Grid spacing = ATR × this multiplier
    atr_position_cap: int = 5             # Max lots from ATR Grid strategy

    # ── Stop-and-Reverse Strategy ─────────────────────────────────────────────
    sar_fast_ema: int = 9                  # Fast EMA period for crossover signal
    sar_slow_ema: int = 21                 # Slow EMA period for crossover signal
    sar_atr_period: int = 14              # ATR period used inside SAR (if needed)

    # ── Macro Regime Engine ───────────────────────────────────────────────────
    vix_high_threshold: float = 20.0             # VIX above this contributes -2 (bearish)
    vix_low_threshold: float = 14.0              # VIX below this contributes +2 (bullish)
    usdinr_stress_level: float = 84.0            # USDINR at/above this contributes -1 (bearish)
    bullish_score_threshold: float = 2.0         # Macro score >= this triggers BULLISH regime
    bearish_score_threshold: float = -2.0        # Macro score <= this triggers BEARISH regime
    bearish_position_cap: int = 2                # Reduced max lots in BEARISH regime
    bearish_grid_multiplier: float = 2.0         # Wider grid spacing multiplier in BEARISH regime
    circuit_breaker_vix_threshold: float = 20.0  # High-volatility VIX threshold for circuit breaker

    # ── Rolling Walk-Forward Evaluation ───────────────────────────────────────
    train_bars: int = 252                  # Training window length (~1 year)
    test_bars: int = 63                    # Test window length (~1 quarter)

    # ── Operational & Broker Settings ─────────────────────────────────────────
    broker: str = "paper"                  # "paper" | "kite"
    kite_api_key: str | None = None        # Zerodha Kite Connect API key
    kite_api_secret: str | None = None     # Zerodha Kite Connect API secret
    kite_access_token: str | None = None   # Daily session access token
    redis_url: str | None = None           # Redis connection URL


def load_config(path: str | Path = "config.yaml") -> VegaConfig:
    """
    Load VegaConfig from a YAML file.

    If the file does not exist, all default values from VegaConfig are used.
    This makes it safe to call load_config() in tests without creating a file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A VegaConfig instance populated from the file (or defaults).

    Example:
        config = load_config("config.yaml")
        print(config.initial_capital)  # 1000000.0
    """
    import os

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f) or {}
    except FileNotFoundError:
        # No config file — use all dataclass defaults.
        raw = {}

    # Extract nested sections, defaulting to empty dict if section is absent.
    b   = raw.get("backtest", {})
    c   = raw.get("costs", {})
    r   = raw.get("risk", {})
    ag  = raw.get("strategies", {}).get("atr_grid", {})
    sar = raw.get("strategies", {}).get("stop_and_reverse", {})
    m   = raw.get("macro", {})
    wf  = raw.get("walk_forward", {})

    # Operational settings with environment-variable precedence
    env_broker = os.getenv("BROKER", raw.get("broker", "paper"))
    env_kite_key = os.getenv("KITE_API_KEY", raw.get("kite", {}).get("api_key"))
    env_kite_secret = os.getenv("KITE_API_SECRET", raw.get("kite", {}).get("api_secret"))
    env_kite_token = os.getenv("KITE_ACCESS_TOKEN", raw.get("kite", {}).get("access_token"))
    env_redis_url = os.getenv("REDIS_URL", raw.get("redis", {}).get("url"))

    return VegaConfig(
        # Backtest
        initial_capital   = float(b.get("initial_capital", 1_000_000.0)),
        symbol            = str(b.get("symbol", "NIFTY50")),
        # Costs
        slippage_pct      = float(c.get("slippage_pct", 0.001)),
        brokerage_pct     = float(c.get("brokerage_pct", 0.0003)),
        # Risk
        max_position_size = int(r.get("max_position_size", 10)),
        max_pyramids      = int(r.get("max_pyramids", 3)),
        daily_loss_limit  = float(r.get("daily_loss_limit", 50_000.0)),
        # ATR Grid
        atr_period        = int(ag.get("atr_period", 14)),
        atr_grid_multiplier = float(ag.get("grid_multiplier", 1.5)),
        atr_position_cap  = int(ag.get("position_cap", 5)),
        # SAR
        sar_fast_ema      = int(sar.get("fast_ema", 9)),
        sar_slow_ema      = int(sar.get("slow_ema", 21)),
        sar_atr_period    = int(sar.get("atr_period", 14)),
        # Macro
        vix_high_threshold  = float(m.get("vix_high_threshold", 20.0)),
        vix_low_threshold   = float(m.get("vix_low_threshold", 14.0)),
        usdinr_stress_level = float(m.get("usdinr_stress_level", 84.0)),
        bullish_score_threshold = float(m.get("bullish_score_threshold", 2.0)),
        bearish_score_threshold = float(m.get("bearish_score_threshold", -2.0)),
        bearish_position_cap    = int(m.get("bearish_position_cap", 2)),
        bearish_grid_multiplier = float(m.get("bearish_grid_multiplier", 2.0)),
        circuit_breaker_vix_threshold = float(m.get("circuit_breaker_vix_threshold", 20.0)),
        # Walk-forward
        train_bars = int(wf.get("train_bars", 252)),
        test_bars  = int(wf.get("test_bars", 63)),
        # Operational
        broker            = env_broker,
        kite_api_key      = env_kite_key,
        kite_api_secret   = env_kite_secret,
        kite_access_token = env_kite_token,
        redis_url         = env_redis_url,
    )
