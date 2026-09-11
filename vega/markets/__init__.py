"""
Indian market plumbing package for the Vega Quant Trading Engine.

Exports:
    - Contract specifications: ContractSpec, InstrumentType, Exchange, StaticContractRegistry.
    - Statutory cost engine: CostSchedule, CostBreakdown, CostEngine, MarginEstimate, estimate_margin,
      and published 2026 tariff schedules (NSE_FUTURES_SCHEDULE_2026, MCX_FUTURES_SCHEDULE_2026,
      NSE_EQUITY_INTRADAY_SCHEDULE_2026).
    - Expiry & Rollover: TradingCalendar, ExpiryRule, RolloverDecision, RolloverManager.
"""

from vega.markets.contract_master import (
    ContractSpec,
    Exchange,
    InstrumentType,
    StaticContractRegistry,
)
from vega.markets.costs import (
    CostBreakdown,
    CostEngine,
    CostSchedule,
    MarginEstimate,
    MCX_FUTURES_SCHEDULE_2026,
    NSE_EQUITY_INTRADAY_SCHEDULE_2026,
    NSE_FUTURES_SCHEDULE_2026,
    estimate_margin,
)
from vega.markets.rollover import (
    CANONICAL_INDIAN_HOLIDAYS_2026,
    ExpiryRule,
    RolloverDecision,
    RolloverManager,
    TradingCalendar,
)

__all__ = [
    # Contract master
    "ContractSpec",
    "InstrumentType",
    "Exchange",
    "StaticContractRegistry",
    # Costs & margins
    "CostBreakdown",
    "CostEngine",
    "CostSchedule",
    "MarginEstimate",
    "estimate_margin",
    "NSE_FUTURES_SCHEDULE_2026",
    "MCX_FUTURES_SCHEDULE_2026",
    "NSE_EQUITY_INTRADAY_SCHEDULE_2026",
    # Rollover & Expiry
    "TradingCalendar",
    "ExpiryRule",
    "RolloverDecision",
    "RolloverManager",
    "CANONICAL_INDIAN_HOLIDAYS_2026",
]
