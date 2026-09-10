"""
Contract Master — static instrument definitions.

A "contract" in trading describes the exact specification of an instrument:
its exchange, type (equity/futures/options), lot size, tick size, and expiry.

In a production system, this data would be fetched daily from the exchange
(NSE publishes an instrument master CSV). For this backtesting engine we use
a small set of static sample data that matches real NSE/NFO specifications.

Typical usage:
    from vega.contracts.master import ContractMaster
    master = ContractMaster()
    contract = master.get_contract("NIFTY50")
    print(contract.lot_size)   # 50
    print(contract.tick_size)  # 0.05
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class ContractInfo:
    """
    Full specification of a tradeable instrument.

    Attributes:
        symbol:          Ticker / trading symbol, e.g. "NIFTY50".
        exchange:        Exchange where it trades: "NSE", "BSE", or "NFO".
        instrument_type: "EQ" = equity index/stock,
                         "FUT" = futures contract,
                         "OPT" = options contract.
        expiry:          Expiry date for derivatives (futures/options).
                         None for equities (they don't expire).
        lot_size:        Number of units per lot.
                         For NIFTY50 futures, 1 lot = 50 units.
        tick_size:       Minimum price movement, e.g. 0.05 means prices
                         move in multiples of ₹0.05.
    """

    symbol: str
    exchange: str
    instrument_type: str
    expiry: date | None
    lot_size: int
    tick_size: float


# ── Static instrument definitions ─────────────────────────────────────────────
# These values match actual NSE/NFO specifications as of 2024.
# In production, this would be refreshed from the NSE instrument master file.

_CONTRACTS: dict[str, ContractInfo] = {
    "NIFTY50": ContractInfo(
        symbol          = "NIFTY50",
        exchange        = "NSE",
        instrument_type = "EQ",
        expiry          = None,
        lot_size        = 50,
        tick_size       = 0.05,
    ),
    "NIFTY_FUT": ContractInfo(
        symbol          = "NIFTY_FUT",
        exchange        = "NFO",
        instrument_type = "FUT",
        expiry          = date(2024, 12, 26),   # December expiry
        lot_size        = 50,
        tick_size       = 0.05,
    ),
    "BANKNIFTY": ContractInfo(
        symbol          = "BANKNIFTY",
        exchange        = "NSE",
        instrument_type = "EQ",
        expiry          = None,
        lot_size        = 15,
        tick_size       = 0.05,
    ),
    "RELIANCE": ContractInfo(
        symbol          = "RELIANCE",
        exchange        = "NSE",
        instrument_type = "EQ",
        expiry          = None,
        lot_size        = 1,
        tick_size       = 0.05,
    ),
}


class ContractMaster:
    """
    Provides read-only access to the static instrument definitions.

    This is intentionally simple: a dictionary lookup.
    No network calls, no database queries, no caching needed.

    Example:
        master = ContractMaster()

        contract = master.get_contract("NIFTY50")
        print(contract.lot_size)   # 50
        print(contract.expiry)     # None (equities don't expire)

        all_symbols = master.get_all_symbols()
        # ["NIFTY50", "NIFTY_FUT", "BANKNIFTY", "RELIANCE"]
    """

    def get_contract(self, symbol: str) -> ContractInfo:
        """
        Return the ContractInfo for the given symbol.

        Args:
            symbol: Instrument symbol, e.g. "NIFTY50".

        Returns:
            The ContractInfo for this symbol.

        Raises:
            KeyError: If the symbol is not in the static master data.
        """
        if symbol not in _CONTRACTS:
            available = list(_CONTRACTS.keys())
            raise KeyError(
                f"Symbol '{symbol}' not found in the contract master.\n"
                f"Available symbols: {available}"
            )
        return _CONTRACTS[symbol]

    def get_all_symbols(self) -> list[str]:
        """Return a list of all symbols defined in the contract master."""
        return list(_CONTRACTS.keys())
