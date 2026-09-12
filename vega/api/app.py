"""
FastAPI REST Application for the Vega Quant Trading Engine.

Architectural Guarantees:
    - Pure read-only presentation and telemetry surface.
    - Does not own or execute trading, strategy, or order placement logic.
    - Decoupled from runtime authority:
        FastAPI ──> Redis State Cache (read model)
        FastAPI ──> TradeBlotter (historical executed trades & metrics)
        FastAPI ──> Portfolio (authoritative fallback when cache is unpopulated or severed)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from vega.cache.redis_cache import RedisStateCache
from vega.portfolio.portfolio import Portfolio
from vega.reporting.blotter import TradeBlotter


# ─── Pydantic Response Models ────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field("ok", description="Process liveness status")
    timestamp: str = Field(..., description="UTC ISO-8601 timestamp")
    version: str = Field("1.0.0", description="API version")


class StatusResponse(BaseModel):
    """Engine operational and risk status."""

    status: str = Field(..., description="Engine lifecycle state (e.g. RUNNING, STOPPED)")
    kill_switch: bool = Field(False, description="Emergency risk gate status")
    source: str = Field(..., description="Data origin ('cache' or 'domain')")
    timestamp: str = Field(..., description="UTC timestamp of the observation")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata")


class PortfolioResponse(BaseModel):
    """Top-level portfolio valuation and cash metrics."""

    equity: float = Field(..., description="Total portfolio equity in INR")
    cash: float = Field(..., description="Available cash balance in INR")
    realized_pnl: float = Field(0.0, description="Cumulative realized P&L in INR")
    unrealized_pnl: float = Field(0.0, description="Open mark-to-market P&L in INR")
    drawdown_pct: float = Field(0.0, description="Current drawdown percentage from peak")
    source: str = Field(..., description="Data origin ('cache' or 'domain')")


class PositionResponse(BaseModel):
    """Individual instrument position holding."""

    symbol: str = Field(..., description="Ticker symbol")
    quantity: int = Field(..., description="Net units (>0 long, <0 short, 0 flat)")
    avg_entry_price: float = Field(..., description="Average cost basis per unit")
    realized_pnl: float = Field(0.0, description="Cumulative realized P&L for this symbol")
    unrealized_pnl: float = Field(0.0, description="Mark-to-market unrealized P&L")


class PositionsListResponse(BaseModel):
    """Collection of open/tracked instrument positions."""

    positions: list[PositionResponse] = Field(default_factory=list)
    count: int = Field(0, description="Number of tracked positions")


class TradeResponse(BaseModel):
    """Standardized trade execution blotter record (11 fields)."""

    timestamp: str = Field(..., description="Execution timestamp")
    symbol: str = Field(..., description="Instrument symbol")
    side: str = Field(..., description="Execution side ('BUY' or 'SELL')")
    quantity: int = Field(..., description="Executed quantity")
    signal_price: float | None = Field(None, description="Signal price or None if unrecorded")
    execution_price: float = Field(..., description="Actual fill price including slippage")
    slippage: float = Field(..., description="Per-unit slippage applied")
    brokerage: float = Field(..., description="Commission charged")
    realized_pnl: float = Field(..., description="Realized P&L (0.0 for entries)")
    strategy: str = Field("", description="Strategy name")
    reason: str = Field("", description="Execution or signal rationale")


class TradesListResponse(BaseModel):
    """Collection of executed trade records."""

    trades: list[TradeResponse] = Field(default_factory=list)
    total_trades: int = Field(0, description="Total executed trades on blotter")


class MetricsResponse(BaseModel):
    """Engine trading performance metrics."""

    win_rate: float = Field(..., description="Proportion of winning closed trades")
    profit_factor: float | None = Field(..., description="Gross profit / abs(loss) or None if infinite")
    gross_realized_pnl: float = Field(..., description="Sum of realized P&L across all closed trades")
    total_brokerage: float = Field(..., description="Total commissions paid")
    net_pnl: float = Field(..., description="Gross Realized P&L minus Total Brokerage")
    total_slippage_cost: float = Field(..., description="Informational slippage cost impact")
    total_trades: int = Field(..., description="Total executed orders")
    closed_trades: int = Field(..., description="Trades with non-zero realized P&L")


# ─── FastAPI Application Factory ─────────────────────────────────────────────


def create_app(
    cache: RedisStateCache | None = None,
    portfolio: Portfolio | None = None,
    blotter: TradeBlotter | None = None,
    system_status: str = "RUNNING",
    kill_switch: bool = False,
    version: str = "1.0.0",
) -> FastAPI:
    """
    Construct a FastAPI instance with injected read dependencies.

    Args:
        cache: Optional RedisStateCache adapter (defaults to an in-memory cache).
        portfolio: Optional domain Portfolio authority (defaults to 1M INR cash).
        blotter: Optional TradeBlotter reporting instance (defaults to empty blotter).
        system_status: Baseline engine status string.
        kill_switch: Baseline kill-switch state.
        version: API semantic version string.

    Returns:
        Configured FastAPI application instance.
    """
    _cache = cache if cache is not None else RedisStateCache.create_in_memory()
    _portfolio = portfolio if portfolio is not None else Portfolio(initial_cash=1_000_000.0)
    _blotter = blotter if blotter is not None else TradeBlotter()

    app = FastAPI(
        title="Vega Quant Trading Engine API",
        description="Low-latency telemetry and reporting surface for the Vega Quant Trading Engine.",
        version=version,
    )

    @app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
    def get_health() -> HealthResponse:
        """Process liveness probe; always succeeds if the HTTP server is responsive."""
        return HealthResponse(
            status="ok",
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=version,
        )

    @app.get("/status", response_model=StatusResponse, tags=["Telemetry"])
    def get_status() -> StatusResponse:
        """
        Engine operational and risk status.
        Reads from Redis cache if available, falling back to domain state.
        """
        cached = _cache.get_system_status()
        if cached is not None and isinstance(cached, dict):
            return StatusResponse(
                status=str(cached.get("status", system_status)),
                kill_switch=bool(cached.get("kill_switch", kill_switch)),
                source="cache",
                timestamp=str(cached.get("timestamp", datetime.now(timezone.utc).isoformat())),
                details={k: v for k, v in cached.items() if k not in ("status", "kill_switch", "timestamp")},
            )

        # Fallback to domain state
        return StatusResponse(
            status=system_status,
            kill_switch=kill_switch,
            source="domain",
            timestamp=datetime.now(timezone.utc).isoformat(),
            details={},
        )

    @app.get("/portfolio", response_model=PortfolioResponse, tags=["Portfolio"])
    def get_portfolio() -> PortfolioResponse:
        """
        Top-level portfolio equity and cash metrics.
        Attempts cache read first, gracefully falling back to domain state.
        """
        cached = _cache.get_portfolio()
        if cached is not None and isinstance(cached, dict):
            return PortfolioResponse(
                equity=float(cached.get("equity", 0.0)),
                cash=float(cached.get("cash", 0.0)),
                realized_pnl=float(cached.get("realized_pnl", 0.0)),
                unrealized_pnl=float(cached.get("unrealized_pnl", 0.0)),
                drawdown_pct=float(cached.get("drawdown_pct", 0.0)),
                source="cache",
            )

        # Authoritative domain fallback
        equity = float(_portfolio.get_equity())
        cash = float(_portfolio.get_cash())
        realized_pnl = float(_portfolio.get_realized_pnl())
        unrealized_pnl = float(_portfolio.get_total_unrealized_pnl()) if hasattr(_portfolio, "get_total_unrealized_pnl") else 0.0
        drawdown_pct = float(_portfolio.get_drawdown_pct())

        return PortfolioResponse(
            equity=equity,
            cash=cash,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            drawdown_pct=drawdown_pct,
            source="domain",
        )

    @app.get("/positions", response_model=PositionsListResponse, tags=["Portfolio"])
    def get_positions() -> PositionsListResponse:
        """Return active portfolio positions directly from the domain authority."""
        positions_dict = _portfolio.get_all_positions()
        items: list[PositionResponse] = []
        for pos in positions_dict.values():
            items.append(
                PositionResponse(
                    symbol=pos.symbol,
                    quantity=pos.quantity,
                    avg_entry_price=float(pos.average_entry_price),
                    realized_pnl=float(pos.realized_pnl),
                    unrealized_pnl=float(pos.unrealized_pnl),
                )
            )
        return PositionsListResponse(positions=items, count=len(items))

    @app.get("/trades", response_model=TradesListResponse, tags=["Reporting"])
    def get_trades() -> TradesListResponse:
        """Return standardized trade blotter execution records."""
        entries = _blotter.entries
        items: list[TradeResponse] = []
        for e in entries:
            ts_str = e.timestamp.isoformat() if isinstance(e.timestamp, datetime) else str(e.timestamp)
            items.append(
                TradeResponse(
                    timestamp=ts_str,
                    symbol=e.symbol,
                    side=e.side,
                    quantity=e.quantity,
                    signal_price=e.signal_price,
                    execution_price=e.execution_price,
                    slippage=e.slippage,
                    brokerage=e.brokerage,
                    realized_pnl=e.realized_pnl,
                    strategy=e.strategy,
                    reason=e.reason,
                )
            )
        return TradesListResponse(trades=items, total_trades=len(items))

    @app.get("/metrics", response_model=MetricsResponse, tags=["Reporting"])
    def get_metrics() -> MetricsResponse:
        """Return key trading performance metrics calculated by TradeBlotter."""
        pf = _blotter.profit_factor
        pf_val = None if pf == float("inf") else pf

        return MetricsResponse(
            win_rate=_blotter.win_rate,
            profit_factor=pf_val,
            gross_realized_pnl=_blotter.gross_realized_pnl,
            total_brokerage=_blotter.total_brokerage,
            net_pnl=_blotter.net_pnl,
            total_slippage_cost=_blotter.total_slippage_cost,
            total_trades=_blotter.total_trades,
            closed_trades=_blotter.closed_trades_count,
        )

    return app


# Default module-level application instance for 'uvicorn vega.api.app:app'
app = create_app()
