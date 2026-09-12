"""
Tests for broker selection, configuration loading, and factory instantiation.
"""

from __future__ import annotations

import os
import pytest

from vega.broker.factory import create_broker
from vega.broker.kite import KiteBroker
from vega.broker.paper import PaperBroker
from vega.config import VegaConfig, load_config


def test_default_broker_selection_is_paper():
    """Default configuration instantiates PaperBroker with default config."""
    broker = create_broker()
    assert isinstance(broker, PaperBroker)
    assert broker.slippage_pct == 0.001


def test_custom_paper_broker_configuration():
    """Specifying broker='paper' returns PaperBroker with custom config."""
    cfg = VegaConfig(broker="paper", slippage_pct=0.0025)
    broker = create_broker(cfg)
    assert isinstance(broker, PaperBroker)
    assert broker.slippage_pct == 0.0025


def test_kite_broker_selection_with_credentials():
    """Configuring broker='kite' with valid credentials returns configured KiteBroker."""
    cfg = VegaConfig(
        broker="kite",
        kite_api_key="test_api_key_123",
        kite_access_token="test_access_token_456",
    )
    broker = create_broker(cfg)
    assert isinstance(broker, KiteBroker)
    assert broker.api_key == "test_api_key_123"
    assert broker.access_token == "test_access_token_456"


def test_kite_broker_selection_missing_credentials_raises_error():
    """Configuring broker='kite' without credentials raises clear ValueError."""
    cfg = VegaConfig(broker="kite", kite_api_key=None, kite_access_token=None)
    with pytest.raises(ValueError, match="KiteBroker requires both 'KITE_API_KEY' and 'KITE_ACCESS_TOKEN'"):
        create_broker(cfg)


def test_unsupported_broker_name_raises_error():
    """Specifying an unknown broker name raises ValueError."""
    cfg = VegaConfig(broker="binance")
    with pytest.raises(ValueError, match="Unsupported broker 'binance'"):
        create_broker(cfg)


def test_environment_variable_override_for_broker(monkeypatch):
    """Environment variables dynamically override config defaults."""
    monkeypatch.setenv("BROKER", "kite")
    monkeypatch.setenv("KITE_API_KEY", "env_key_abc")
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "env_token_xyz")
    monkeypatch.setenv("REDIS_URL", "redis://custom-redis:6379/1")

    cfg = load_config("non_existent_config.yaml")
    assert cfg.broker == "kite"
    assert cfg.kite_api_key == "env_key_abc"
    assert cfg.kite_access_token == "env_token_xyz"
    assert cfg.redis_url == "redis://custom-redis:6379/1"

    broker = create_broker(cfg)
    assert isinstance(broker, KiteBroker)
    assert broker.api_key == "env_key_abc"
    assert broker.access_token == "env_token_xyz"
