"""
Dashboard package for the Vega Quant Trading Engine.

Exports:
    - DashboardAPIClient: REST client fetching telemetry for visualization.
    - format_trades_dataframe: Standardized Trade Blotter DataFrame formatter.
    - render_dashboard: Streamlit presentation entrypoint.
"""

from vega.dashboard.app import (
    DashboardAPIClient,
    format_trades_dataframe,
    render_dashboard,
)

__all__ = [
    "DashboardAPIClient",
    "format_trades_dataframe",
    "render_dashboard",
]
