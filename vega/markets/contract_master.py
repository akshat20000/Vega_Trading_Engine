"""
Indian Market Contract Specifications and Static Registry.

Provides:
    - InstrumentType: Categorization of Indian financial instruments.
    - Exchange: Supported exchange identifiers (NSE, NFO, MCX).
    - ContractSpec: Contract specifications with lot size, tick size, and validation.
    - StaticContractRegistry: In-memory registry for canonical Indian contracts
      (Index Futures, Stock Futures, Commodity Futures).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum


class InstrumentType(str, Enum):
    """Financial instrument classification for Indian markets."""

    EQUITY = "EQUITY"
    INDEX_FUT = "INDEX_FUT"
    STOCK_FUT = "STOCK_FUT"
    INDEX_OPT = "INDEX_OPT"
    STOCK_OPT = "STOCK_OPT"
    COMMODITY_FUT = "COMMODITY_FUT"


class Exchange(str, Enum):
    """Trading venues."""

    NSE = "NSE"
    NFO = "NFO"
    MCX = "MCX"


@dataclass(frozen=True)
class ContractSpec:
    """
    Standardized specification for an Indian market derivative or cash instrument.

    Attributes:
        symbol: Ticker identifier (e.g. 'NIFTY26APRFUT', 'CRUDEOIL26APRFUT').
        underlying: Base instrument symbol (e.g. 'NIFTY', 'BANKNIFTY', 'CRUDEOIL').
        exchange: Exchange identifier ('NSE', 'NFO', 'MCX').
        instrument_type: Category of derivative / asset.
        lot_size: Minimum tradable unit multiplier (e.g. 50 or 25 for NIFTY, 100 for CRUDEOIL).
        tick_size: Minimum discrete price movement (e.g. 0.05 for NSE, 1.00 for CRUDEOIL).
        expiry_type: 'MONTHLY', 'WEEKLY', or 'NONE'.
        expiry_date: Explicit contract expiration date (or None for perpetual/cash).
        expiry_weekday: Nominal expiry weekday (1 = Tuesday for current NSE rules).
        strike_price: Strike price for option contracts (or None).
        option_type: 'CE', 'PE', or None.
        margin_rate: Configurable indicative margin requirement rate (e.g. 0.12 for 12%).
    """

    symbol: str
    underlying: str
    exchange: str
    instrument_type: InstrumentType
    lot_size: int
    tick_size: Decimal
    expiry_type: str = "MONTHLY"
    expiry_date: date | None = None
    expiry_weekday: int = 1
    strike_price: Decimal | None = None
    option_type: str | None = None
    margin_rate: Decimal = Decimal("0.12")

    def round_to_tick(self, price: Decimal | float | int) -> Decimal:
        """
        Round price to the nearest valid tick size using standard half-up rounding.
        """
        p = Decimal(str(price))
        ticks = (p / self.tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        rounded = ticks * self.tick_size
        return rounded.quantize(self.tick_size)

    def validate_price(self, price: Decimal | float | int) -> bool:
        """
        Validate whether a price is strictly positive and aligns with tick size.

        Returns:
            True if price > 0 and price is an integer multiple of tick_size, else False.
        """
        p = Decimal(str(price))
        if p <= Decimal("0"):
            return False
        quotient = p / self.tick_size
        return quotient == quotient.to_integral_value()

    def validate_quantity(self, quantity: int) -> bool:
        """
        Validate whether order quantity is positive and a valid multiple of lot_size.

        Rejects invalid quantities without silently rounding.
        """
        if not isinstance(quantity, int) or quantity <= 0:
            return False
        return (quantity % self.lot_size) == 0


class StaticContractRegistry:
    """
    In-memory static contract repository for Indian markets.

    Note:
        In production systems, this repository would be hydrated dynamically from
        the daily exchange contract master files (e.g., NSE bhavcopy / contract.txt).
        For deterministic backtesting and simulation, this provides verified canonical
        specifications.
    """

    def __init__(self) -> None:
        self._contracts: dict[str, ContractSpec] = {}

    def register(self, spec: ContractSpec) -> None:
        """Register or update a contract specification."""
        self._contracts[spec.symbol.upper()] = spec

    def get(self, symbol: str) -> ContractSpec:
        """
        Retrieve contract specification by symbol.

        Raises:
            KeyError: If the contract symbol is not registered.
        """
        sym = symbol.upper()
        if sym not in self._contracts:
            raise KeyError(
                f"Contract '{symbol}' not found in registry. "
                f"Available symbols: {list(self._contracts.keys())}"
            )
        return self._contracts[sym]

    def find_by_underlying(self, underlying: str) -> list[ContractSpec]:
        """Find all contracts matching an underlying symbol."""
        und = underlying.upper()
        return [c for c in self._contracts.values() if c.underlying.upper() == und]

    def all_contracts(self) -> list[ContractSpec]:
        """Return list of all registered contracts."""
        return list(self._contracts.values())

    @classmethod
    def create_default(cls) -> StaticContractRegistry:
        """
        Factory creating a registry pre-populated with canonical Indian contracts.
        """
        reg = cls()

        # ── NSE / NFO Index Futures ──────────────────────────────────────────
        reg.register(
            ContractSpec(
                symbol="NIFTY26APRFUT",
                underlying="NIFTY",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.INDEX_FUT,
                lot_size=25,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,  # Tuesday
                margin_rate=Decimal("0.12"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="NIFTY26MAYFUT",
                underlying="NIFTY",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.INDEX_FUT,
                lot_size=25,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 5, 26),
                expiry_weekday=1,
                margin_rate=Decimal("0.12"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="BANKNIFTY26APRFUT",
                underlying="BANKNIFTY",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.INDEX_FUT,
                lot_size=15,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,
                margin_rate=Decimal("0.15"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="FINNIFTY26APRFUT",
                underlying="FINNIFTY",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.INDEX_FUT,
                lot_size=25,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,
                margin_rate=Decimal("0.14"),
            )
        )

        # ── NSE Stock Futures ────────────────────────────────────────────────
        reg.register(
            ContractSpec(
                symbol="RELIANCE26APRFUT",
                underlying="RELIANCE",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.STOCK_FUT,
                lot_size=250,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,
                margin_rate=Decimal("0.20"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="TCS26APRFUT",
                underlying="TCS",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.STOCK_FUT,
                lot_size=175,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,
                margin_rate=Decimal("0.18"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="HDFCBANK26APRFUT",
                underlying="HDFCBANK",
                exchange=Exchange.NFO.value,
                instrument_type=InstrumentType.STOCK_FUT,
                lot_size=550,
                tick_size=Decimal("0.05"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 28),
                expiry_weekday=1,
                margin_rate=Decimal("0.18"),
            )
        )

        # ── MCX Commodity Futures ────────────────────────────────────────────
        reg.register(
            ContractSpec(
                symbol="CRUDEOIL26APRFUT",
                underlying="CRUDEOIL",
                exchange=Exchange.MCX.value,
                instrument_type=InstrumentType.COMMODITY_FUT,
                lot_size=100,
                tick_size=Decimal("1.00"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 17),
                margin_rate=Decimal("0.25"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="GOLD26JUNFUT",
                underlying="GOLD",
                exchange=Exchange.MCX.value,
                instrument_type=InstrumentType.COMMODITY_FUT,
                lot_size=100,
                tick_size=Decimal("1.00"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 6, 5),
                margin_rate=Decimal("0.10"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="SILVER26MAYFUT",
                underlying="SILVER",
                exchange=Exchange.MCX.value,
                instrument_type=InstrumentType.COMMODITY_FUT,
                lot_size=30,
                tick_size=Decimal("1.00"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 5, 5),
                margin_rate=Decimal("0.12"),
            )
        )
        reg.register(
            ContractSpec(
                symbol="NATURALGAS26APRFUT",
                underlying="NATURALGAS",
                exchange=Exchange.MCX.value,
                instrument_type=InstrumentType.COMMODITY_FUT,
                lot_size=1250,
                tick_size=Decimal("0.10"),
                expiry_type="MONTHLY",
                expiry_date=date(2026, 4, 24),
                margin_rate=Decimal("0.30"),
            )
        )

        return reg
