"""
Streamlit Dashboard for the Vega Quant Trading Engine.

Architectural Guarantees:
    - Strict Presentation Layer: contains ZERO trading rules, calculations, or position mutations.
    - Consumes telemetry exclusively via the FastAPI REST surface.
    - Truthful Error State: if FastAPI is unreachable, displays a prominent 'API OFFLINE'
      diagnostic rather than inventing fallback data or silently bypassing the API.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


class DashboardAPIClient:
    """HTTP client querying the Vega FastAPI backend."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 2.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_health(self) -> dict[str, Any] | None:
        """Fetch liveness health check."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_status(self) -> dict[str, Any] | None:
        """Fetch engine operational status."""
        try:
            r = requests.get(f"{self.base_url}/status", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_portfolio(self) -> dict[str, Any] | None:
        """Fetch portfolio equity and cash metrics."""
        try:
            r = requests.get(f"{self.base_url}/portfolio", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def get_positions(self) -> list[dict[str, Any]] | None:
        """Fetch active positions."""
        try:
            r = requests.get(f"{self.base_url}/positions", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data.get("positions", [])
            return None
        except Exception:
            return None

    def get_trades(self) -> list[dict[str, Any]] | None:
        """Fetch trade blotter records."""
        try:
            r = requests.get(f"{self.base_url}/trades", timeout=self.timeout)
            if r.status_code == 200:
                data = r.json()
                return data.get("trades", [])
            return None
        except Exception:
            return None

    def get_metrics(self) -> dict[str, Any] | None:
        """Fetch performance metrics."""
        try:
            r = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None


def format_trades_dataframe(trades: list[dict[str, Any]]) -> pd.DataFrame:
    """Format trade blotter entries into a presentation-ready DataFrame."""
    if not trades:
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "Symbol",
                "Side",
                "Quantity",
                "Signal Price",
                "Execution Price",
                "Slippage",
                "Brokerage",
                "Realized PnL",
                "Strategy",
                "Reason",
            ]
        )

    rows = []
    for t in trades:
        rows.append(
            {
                "Timestamp": t.get("timestamp", ""),
                "Symbol": t.get("symbol", ""),
                "Side": t.get("side", ""),
                "Quantity": t.get("quantity", 0),
                "Signal Price": f"₹{t['signal_price']:,.2f}" if t.get("signal_price") is not None else "-",
                "Execution Price": f"₹{t.get('execution_price', 0.0):,.2f}",
                "Slippage": f"₹{t.get('slippage', 0.0):,.2f}",
                "Brokerage": f"₹{t.get('brokerage', 0.0):,.2f}",
                "Realized PnL": f"₹{t.get('realized_pnl', 0.0):+,.2f}",
                "Strategy": t.get("strategy", ""),
                "Reason": t.get("reason", ""),
            }
        )
    return pd.DataFrame(rows)


def render_dashboard() -> None:
    """Render the primary Streamlit user interface."""
    st.set_page_config(
        page_title="Vega Quant Trading Engine",
        page_icon="⚡",
        layout="wide",
    )

    # ── Sidebar Controls & Status ─────────────────────────────────────────────
    st.sidebar.title("Vega Engine")
    api_url = st.sidebar.text_input(
        "API Base URL",
        value=os.getenv("VEGA_API_URL", "http://localhost:8000"),
    )
    client = DashboardAPIClient(base_url=api_url)

    health = client.get_health()
    if health is None:
        st.sidebar.error("🔴 API OFFLINE")
        st.error(
            f"⚠️ **API OFFLINE**: Unable to connect to Vega backend at `{api_url}`.\n\n"
            "Please ensure the FastAPI service is running:\n"
            "```bash\nuvicorn vega.api.app:app --port 8000\n```"
        )
        st.info("The dashboard is strictly a presentation layer and does not fabricate data when the API is unreachable.")
        return

    st.sidebar.success(f"🟢 API ONLINE (v{health.get('version', '1.0.0')})")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    st.title("⚡ Vega Quant Trading Engine")
    st.caption("Standardized Execution Blotter & Real-Time Telemetry")

    # ── Fetch Telemetry ───────────────────────────────────────────────────────
    status = client.get_status() or {}
    portfolio = client.get_portfolio() or {}
    positions = client.get_positions() or []
    trades = client.get_trades() or []
    metrics = client.get_metrics() or {}

    # ── Tab Navigation ────────────────────────────────────────────────────────
    tab_overview, tab_positions, tab_trades, tab_performance, tab_system = st.tabs(
        ["Overview", "Positions", "Trades", "Performance", "System Status"]
    )

    # ── 1. Overview ───────────────────────────────────────────────────────────
    with tab_overview:
        st.subheader("Portfolio Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        equity = portfolio.get("equity", 0.0)
        cash = portfolio.get("cash", 0.0)
        realized = portfolio.get("realized_pnl", 0.0)
        unrealized = portfolio.get("unrealized_pnl", 0.0)
        drawdown_pct = portfolio.get("drawdown_pct", 0.0)

        col1.metric("Total Equity", f"₹{equity:,.2f}")
        col2.metric("Available Cash", f"₹{cash:,.2f}")
        col3.metric("Realized P&L", f"₹{realized:+,.2f}", delta=f"₹{realized:+,.2f}")
        col4.metric("Unrealized P&L", f"₹{unrealized:+,.2f}")
        col5.metric("Current Drawdown", f"{drawdown_pct:.2f}%")

        st.divider()
        st.subheader("System Telemetry")
        scol1, scol2, scol3 = st.columns(3)
        scol1.info(f"**Status**: `{status.get('status', 'UNKNOWN')}`")
        kill_switch_active = status.get("kill_switch", False)
        if kill_switch_active:
            scol2.error("**Kill Switch**: `ACTIVATED`")
        else:
            scol2.success("**Kill Switch**: `DISENGAGED`")
        scol3.caption(f"**Data Provenance**: `{portfolio.get('source', status.get('source', 'domain'))}`")

    # ── 2. Positions ──────────────────────────────────────────────────────────
    with tab_positions:
        st.subheader(f"Active Positions ({len(positions)})")
        if positions:
            pos_rows = []
            for p in positions:
                pos_rows.append(
                    {
                        "Symbol": p.get("symbol", ""),
                        "Side": "LONG" if p.get("quantity", 0) > 0 else ("SHORT" if p.get("quantity", 0) < 0 else "FLAT"),
                        "Quantity": p.get("quantity", 0),
                        "Avg Entry Price": f"₹{p.get('avg_entry_price', 0.0):,.2f}",
                        "Realized P&L": f"₹{p.get('realized_pnl', 0.0):+,.2f}",
                        "Unrealized P&L": f"₹{p.get('unrealized_pnl', 0.0):+,.2f}",
                    }
                )
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)
        else:
            st.info("No open positions. Portfolio is entirely in cash.")

    # ── 3. Trades (Trade Blotter) ─────────────────────────────────────────────
    with tab_trades:
        st.subheader(f"Execution Trade Blotter ({len(trades)} Trades)")
        df_trades = format_trades_dataframe(trades)
        st.dataframe(df_trades, use_container_width=True)

        if not df_trades.empty:
            csv_data = df_trades.to_csv(index=False)
            st.download_button(
                label="📥 Export Trade Blotter to CSV",
                data=csv_data,
                file_name="vega_trade_blotter.csv",
                mime="text/csv",
            )

    # ── 4. Performance Metrics ────────────────────────────────────────────────
    with tab_performance:
        st.subheader("Quantitative Performance Metrics")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        win_rate = metrics.get("win_rate", 0.0) * 100
        profit_factor = metrics.get("profit_factor")
        pf_str = f"{profit_factor:.2f}" if profit_factor is not None else "∞"
        gross_pnl = metrics.get("gross_realized_pnl", 0.0)
        brokerage = metrics.get("total_brokerage", 0.0)
        net_pnl = metrics.get("net_pnl", 0.0)
        slippage = metrics.get("total_slippage_cost", 0.0)

        mcol1.metric("Win Rate", f"{win_rate:.1f}%")
        mcol2.metric("Profit Factor", pf_str)
        mcol3.metric("Gross Realized P&L", f"₹{gross_pnl:,.2f}")
        mcol4.metric("Brokerage Paid", f"₹{brokerage:,.2f}")

        mcol5, mcol6, mcol7, mcol8 = st.columns(4)
        mcol5.metric("Net Economic P&L", f"₹{net_pnl:+,.2f}", delta=f"₹{net_pnl:+,.2f}")
        mcol6.metric("Informational Slippage", f"₹{slippage:,.2f}")
        mcol7.metric("Total Executed Trades", metrics.get("total_trades", 0))
        mcol8.metric("Closed Trades (with P&L)", metrics.get("closed_trades", 0))

    # ── 5. System Status ──────────────────────────────────────────────────────
    with tab_system:
        st.subheader("Engine Diagnostics & Telemetry")
        st.json(
            {
                "engine_status": status.get("status", "UNKNOWN"),
                "kill_switch": status.get("kill_switch", False),
                "data_source": status.get("source", "UNKNOWN"),
                "timestamp": status.get("timestamp", ""),
                "diagnostic_details": status.get("details", {}),
                "api_endpoint": api_url,
                "api_version": health.get("version", "1.0.0"),
            }
        )


if __name__ == "__main__":
    render_dashboard()
