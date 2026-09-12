"""
Broker factory for the Vega Quant Trading Engine.

Centralizes broker instantiation based on configuration (paper vs. kite),
ensuring strategies and execution engines consume only the AbstractBroker contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vega.broker.base import AbstractBroker
    from vega.config import VegaConfig


def create_broker(config: VegaConfig | None = None) -> AbstractBroker:
    """
    Instantiate the configured broker implementation.

    Selection logic:
        - BROKER=paper (default): Instantiates deterministic PaperBroker.
        - BROKER=kite: Instantiates KiteBroker with configured credentials.

    Args:
        config: VegaConfig instance. If None, loaded via load_config().

    Returns:
        AbstractBroker instance (PaperBroker or KiteBroker).

    Raises:
        ValueError: If broker name is unknown or if Kite credentials are missing.
    """
    from vega.broker.kite import KiteBroker
    from vega.broker.paper import PaperBroker
    from vega.config import load_config

    cfg = config or load_config()
    broker_name = (cfg.broker or "paper").strip().lower()

    if broker_name == "paper":
        return PaperBroker(config=cfg)
    elif broker_name == "kite":
        if not cfg.kite_api_key or not cfg.kite_access_token:
            raise ValueError(
                "KiteBroker requires both 'KITE_API_KEY' and 'KITE_ACCESS_TOKEN' to be configured. "
                "Specify them in your .env file or pass them directly in VegaConfig."
            )
        return KiteBroker(api_key=cfg.kite_api_key, access_token=cfg.kite_access_token)
    else:
        raise ValueError(
            f"Unsupported broker '{cfg.broker}'. Supported options: 'paper', 'kite'."
        )
