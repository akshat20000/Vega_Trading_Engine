"""
Indian Market Statutory Cost & Margin Engine.

Provides:
    - CostSchedule: Configurable fee & statutory tax schedule (versioned for 2026 rules).
    - CostBreakdown: Itemized breakdown of brokerage, STT/CTT, exchange charges,
      SEBI fees, stamp duty, and GST, matching to the paisa.
    - CostEngine: Deterministic calculation engine using Python Decimal arithmetic.
    - MarginEstimate & estimate_margin(): Indicative margin requirement estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from vega.markets.contract_master import ContractSpec

# Standard 2-decimal places quantizer for currency (paisa precision)
PAISA = Decimal("0.01")


@dataclass(frozen=True)
class CostSchedule:
    """
    Versioned fee and tax schedule for Indian financial markets.

    Attributes:
        name: Human-readable identifier of the schedule.
        effective_date: ISO date string when this tariff became effective.
        brokerage_rate: Broker commission percentage (e.g. 0.0003 for 0.03%).
        brokerage_cap: Maximum broker commission cap per executed order (e.g. ₹20.00).
        stt_sell_rate: Securities Transaction Tax on SELL side (e.g. 0.0005 for 0.05%).
        ctt_sell_rate: Commodities Transaction Tax on SELL side (e.g. 0.0001 for 0.01%).
        exchange_turnover_rate: Exchange transaction levy rate.
        sebi_rate: SEBI regulatory turnover charge (e.g. ₹10 per crore = 0.0001%).
        gst_rate: Goods and Services Tax (18% on brokerage + exchange + SEBI).
        stamp_buy_rate: State stamp duty rate applied to BUY side only.
    """

    name: str
    effective_date: str
    brokerage_rate: Decimal
    brokerage_cap: Decimal
    stt_sell_rate: Decimal
    ctt_sell_rate: Decimal
    exchange_turnover_rate: Decimal
    sebi_rate: Decimal
    gst_rate: Decimal
    stamp_buy_rate: Decimal


# ─── Published 2026 Indian Tariff Schedules ──────────────────────────────────

# NSE Futures (effective 2026-04-01 per Union Budget / NSE levy revisions)
NSE_FUTURES_SCHEDULE_2026 = CostSchedule(
    name="NSE Equity / Index Futures (2026)",
    effective_date="2026-04-01",
    brokerage_rate=Decimal("0.0003"),  # 0.03%
    brokerage_cap=Decimal("20.00"),   # Max ₹20 per order
    stt_sell_rate=Decimal("0.0005"),   # 0.05% on sell side (revised 2026 rate)
    ctt_sell_rate=Decimal("0.0"),
    exchange_turnover_rate=Decimal("0.0000183"),  # 0.00183%
    sebi_rate=Decimal("0.000001"),                 # ₹10 / crore (0.0001%)
    gst_rate=Decimal("0.18"),                     # 18%
    stamp_buy_rate=Decimal("0.00002"),            # 0.002% on buy side
)

# MCX Commodity Futures (2026 published rates)
MCX_FUTURES_SCHEDULE_2026 = CostSchedule(
    name="MCX Commodity Futures (2026)",
    effective_date="2026-04-01",
    brokerage_rate=Decimal("0.0003"),  # 0.03%
    brokerage_cap=Decimal("20.00"),
    stt_sell_rate=Decimal("0.0"),
    ctt_sell_rate=Decimal("0.0001"),   # 0.01% on sell side for non-agri commodities
    exchange_turnover_rate=Decimal("0.000021"),   # 0.0021%
    sebi_rate=Decimal("0.000001"),                 # ₹10 / crore
    gst_rate=Decimal("0.18"),
    stamp_buy_rate=Decimal("0.00002"),            # 0.002% on buy side
)

# NSE Equity Intraday (2026 published rates)
NSE_EQUITY_INTRADAY_SCHEDULE_2026 = CostSchedule(
    name="NSE Equity Intraday (2026)",
    effective_date="2026-04-01",
    brokerage_rate=Decimal("0.0003"),  # 0.03%
    brokerage_cap=Decimal("20.00"),
    stt_sell_rate=Decimal("0.00025"),  # 0.025% on sell side
    ctt_sell_rate=Decimal("0.0"),
    exchange_turnover_rate=Decimal("0.0000307"),  # 0.00307%
    sebi_rate=Decimal("0.000001"),
    gst_rate=Decimal("0.18"),
    stamp_buy_rate=Decimal("0.00003"),            # 0.003% on buy side
)


@dataclass(frozen=True)
class CostBreakdown:
    """
    Itemized deterministic cost breakdown matching to the paisa.

    Attributes:
        turnover: Total trade turnover (price * quantity).
        brokerage: Broker execution commission (capped).
        stt: Securities Transaction Tax (sell-side only).
        ctt: Commodities Transaction Tax (sell-side only).
        exchange_charges: Exchange transaction levy.
        sebi_charges: SEBI regulatory turnover charges.
        stamp_duty: State stamp duty (buy-side only).
        gst: Goods and Services Tax (18% on brokerage + exchange + SEBI).
        total: Combined total cost.
    """

    turnover: Decimal
    brokerage: Decimal
    stt: Decimal
    ctt: Decimal
    exchange_charges: Decimal
    sebi_charges: Decimal
    stamp_duty: Decimal
    gst: Decimal
    total: Decimal

    @property
    def statutory_total(self) -> Decimal:
        """Total statutory & regulatory charges excluding broker commission."""
        return (
            self.stt
            + self.ctt
            + self.exchange_charges
            + self.sebi_charges
            + self.stamp_duty
            + self.gst
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize cost breakdown to float values for JSON / reporting."""
        return {
            "turnover": float(self.turnover),
            "brokerage": float(self.brokerage),
            "stt": float(self.stt),
            "ctt": float(self.ctt),
            "exchange_charges": float(self.exchange_charges),
            "sebi_charges": float(self.sebi_charges),
            "stamp_duty": float(self.stamp_duty),
            "gst": float(self.gst),
            "statutory_total": float(self.statutory_total),
            "total": float(self.total),
        }


class CostEngine:
    """
    Deterministic calculation engine for Indian market execution costs.

    Guarantees:
        - Uses Python Decimal arithmetic throughout.
        - Quantizes every component to ₹0.01 (paisa) using ROUND_HALF_UP.
        - Strictly applies STT/CTT only on SELL orders.
        - Strictly applies Stamp Duty only on BUY orders.
        - Calculates GST as 18% of (Brokerage + Exchange Turnover Charges + SEBI Fees).
    """

    @staticmethod
    def calculate_costs(
        price: Decimal | float | int,
        quantity: int,
        side: str,
        schedule: CostSchedule,
    ) -> CostBreakdown:
        """
        Calculate complete itemized transaction costs for a trade.

        Args:
            price: Execution price per unit.
            quantity: Number of executed units.
            side: Order side ('BUY' or 'SELL').
            schedule: CostSchedule defining statutory and brokerage rates.

        Returns:
            CostBreakdown with all items rounded to paisa precision.
        """
        p = Decimal(str(price))
        q = Decimal(str(quantity))
        side_upper = side.upper()

        # 1. Trade Turnover
        turnover = (p * q).quantize(PAISA, rounding=ROUND_HALF_UP)

        # 2. Brokerage (e.g. 0.03% or ₹20 cap, whichever is lower)
        raw_brokerage = turnover * schedule.brokerage_rate
        brokerage = min(raw_brokerage, schedule.brokerage_cap).quantize(
            PAISA, rounding=ROUND_HALF_UP
        )

        # 3. STT (Securities Transaction Tax - strictly SELL side)
        if side_upper == "SELL" and schedule.stt_sell_rate > Decimal("0"):
            stt = (turnover * schedule.stt_sell_rate).quantize(
                PAISA, rounding=ROUND_HALF_UP
            )
        else:
            stt = Decimal("0.00")

        # 4. CTT (Commodities Transaction Tax - strictly SELL side for MCX)
        if side_upper == "SELL" and schedule.ctt_sell_rate > Decimal("0"):
            ctt = (turnover * schedule.ctt_sell_rate).quantize(
                PAISA, rounding=ROUND_HALF_UP
            )
        else:
            ctt = Decimal("0.00")

        # 5. Exchange Turnover Charges
        exchange_charges = (turnover * schedule.exchange_turnover_rate).quantize(
            PAISA, rounding=ROUND_HALF_UP
        )

        # 6. SEBI Turnover Charges (₹10 / crore)
        sebi_charges = (turnover * schedule.sebi_rate).quantize(
            PAISA, rounding=ROUND_HALF_UP
        )

        # 7. Stamp Duty (strictly BUY side)
        if side_upper == "BUY" and schedule.stamp_buy_rate > Decimal("0"):
            stamp_duty = (turnover * schedule.stamp_buy_rate).quantize(
                PAISA, rounding=ROUND_HALF_UP
            )
        else:
            stamp_duty = Decimal("0.00")

        # 8. GST (18% on Brokerage + Exchange Charges + SEBI Charges)
        gst_base = brokerage + exchange_charges + sebi_charges
        gst = (gst_base * schedule.gst_rate).quantize(
            PAISA, rounding=ROUND_HALF_UP
        )

        # 9. Total Combined Transaction Cost
        total = (
            brokerage
            + stt
            + ctt
            + exchange_charges
            + sebi_charges
            + stamp_duty
            + gst
        ).quantize(PAISA, rounding=ROUND_HALF_UP)

        return CostBreakdown(
            turnover=turnover,
            brokerage=brokerage,
            stt=stt,
            ctt=ctt,
            exchange_charges=exchange_charges,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            gst=gst,
            total=total,
        )


# ─── Margin Requirement Estimation ───────────────────────────────────────────


@dataclass(frozen=True)
class MarginEstimate:
    """
    Indicative margin estimation for an Indian derivative position.

    Attributes:
        contract_value: Notional contract value (price * lot_size * lots).
        margin_rate: Applied percentage margin requirement.
        estimated_margin: Estimated INR capital required.
        label: Disclaiming descriptor indicating this is an approximation.
    """

    contract_value: Decimal
    margin_rate: Decimal
    estimated_margin: Decimal
    label: str = "Estimated margin — not exchange SPAN/Exposure margin"

    def to_dict(self) -> dict[str, Any]:
        """Serialize margin estimate to dictionary."""
        return {
            "contract_value": float(self.contract_value),
            "margin_rate": float(self.margin_rate),
            "estimated_margin": float(self.estimated_margin),
            "label": self.label,
        }


def estimate_margin(
    contract: ContractSpec,
    price: Decimal | float | int,
    lots: int = 1,
    margin_rate: Decimal | float | None = None,
) -> MarginEstimate:
    """
    Estimate initial margin requirement for an Indian derivative contract.

    Args:
        contract: Target ContractSpec.
        price: Current asset price.
        lots: Number of lots to trade.
        margin_rate: Optional custom margin rate (defaults to contract.margin_rate).

    Returns:
        MarginEstimate with notional contract value and required capital.
    """
    p = Decimal(str(price))
    rate = (
        Decimal(str(margin_rate))
        if margin_rate is not None
        else contract.margin_rate
    )
    total_quantity = Decimal(contract.lot_size * lots)
    contract_value = (p * total_quantity).quantize(PAISA, rounding=ROUND_HALF_UP)
    estimated_margin = (contract_value * rate).quantize(PAISA, rounding=ROUND_HALF_UP)

    return MarginEstimate(
        contract_value=contract_value,
        margin_rate=rate,
        estimated_margin=estimated_margin,
    )
