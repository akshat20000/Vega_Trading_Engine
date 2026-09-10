"""
Pytest integration for independent indicator cross-validation against reference pandas-ta.

These tests verify that:
1. EMA matches pandas-ta to within 1e-9 (bit-identical).
2. OBV matches pandas-ta to within 1e-9 (bit-identical).
3. RSI matches pandas-ta from index 14 onwards to within 1e-9.
4. ATR converges to pandas-ta within 0.005 after warmup.
"""

from __future__ import annotations

import pytest

from validation.validate_indicators import (
    generate_synthetic_bars,
    validate_atr,
    validate_ema,
    validate_obv,
    validate_rsi,
)


@pytest.fixture(scope="module")
def validation_bars():
    """Deterministic synthetic bars fixture for validation tests."""
    return generate_synthetic_bars(num_bars=150, seed=42)


def test_cross_validate_ema(validation_bars) -> None:
    """Verify EMA matches reference pandas-ta within tolerance."""
    result = validate_ema(validation_bars, period=20)
    assert result.passed, f"EMA validation failed: diff={result.max_absolute_diff} > tol={result.tolerance}"


def test_cross_validate_obv(validation_bars) -> None:
    """Verify OBV matches reference pandas-ta within tolerance."""
    result = validate_obv(validation_bars)
    assert result.passed, f"OBV validation failed: diff={result.max_absolute_diff} > tol={result.tolerance}"


def test_cross_validate_rsi(validation_bars) -> None:
    """Verify RSI matches reference pandas-ta from index 14 onwards."""
    result = validate_rsi(validation_bars, period=14)
    assert result.passed, f"RSI validation failed: diff={result.max_absolute_diff} > tol={result.tolerance}"


def test_cross_validate_atr(validation_bars) -> None:
    """Verify ATR converges to reference pandas-ta within tolerance."""
    result = validate_atr(validation_bars, period=14)
    assert result.passed, f"ATR validation failed: diff={result.max_absolute_diff} > tol={result.tolerance}"
